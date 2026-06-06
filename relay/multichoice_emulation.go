package relay

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/dto"
	"github.com/QuantumNous/new-api/logger"
	"github.com/QuantumNous/new-api/relay/helper"
	"github.com/QuantumNous/new-api/service"
	"github.com/QuantumNous/new-api/types"
	"github.com/gin-gonic/gin"
	"golang.org/x/sync/errgroup"
)

const (
	multiChoiceFanoutEnabledEnv    = "CHAT_COMPLETIONS_N_FANOUT_ENABLED"
	multiChoiceFanoutMaxChoicesEnv = "CHAT_COMPLETIONS_N_FANOUT_MAX_CHOICES"
	multiChoiceFanoutSelfBaseEnv   = "CHAT_COMPLETIONS_N_FANOUT_SELF_BASE_URL"
	legacyMultiChoiceSelfBaseEnv   = "MULTI_CHOICE_SELF_BASE_URL"
	defaultMaxEmulatedChoices      = 8
)

// maybeEmulateChatCompletionChoices implements OpenAI chat-completions `n` for
// upstreams that do not support it natively. It fans one incoming request out to
// N normal New API chat-completion requests with n=1, then merges their results
// back into a single OpenAI-compatible response.
//
// It intentionally calls New API's own public handler rather than talking to the
// selected channel directly. This keeps model mapping, format conversion,
// retries, quota accounting and channel-specific response handling in one place.
func multiChoiceFanoutEnabled() bool {
	return common.GetEnvOrDefaultBool(multiChoiceFanoutEnabledEnv, true)
}

func multiChoiceFanoutMaxChoices() int {
	maxChoices := common.GetEnvOrDefault(multiChoiceFanoutMaxChoicesEnv, defaultMaxEmulatedChoices)
	if maxChoices < 1 {
		return defaultMaxEmulatedChoices
	}
	return maxChoices
}

func maybeEmulateChatCompletionChoices(c *gin.Context, request *dto.GeneralOpenAIRequest) (bool, *types.NewAPIError) {
	if c == nil || request == nil || request.N == nil || *request.N <= 1 {
		return false, nil
	}
	if !multiChoiceFanoutEnabled() {
		return false, nil
	}
	if request.Tools != nil || request.ToolChoice != nil {
		// Multi-choice tool calling is ambiguous and SillyTavern disables tools for
		// multi-swipe on its side. Let the normal path handle it if a caller insists.
		return false, nil
	}

	n := *request.N
	maxChoices := multiChoiceFanoutMaxChoices()
	if n > maxChoices {
		logger.LogWarn(c, fmt.Sprintf("chat completions n fan-out choices clamped from %d to %d", n, maxChoices))
		n = maxChoices
	}

	stream := request.Stream != nil && *request.Stream
	logger.LogInfo(c, fmt.Sprintf("chat completions n fan-out enabled: choices=%d stream=%t model=%s", n, stream, request.Model))
	if stream {
		return true, emulateStreamChoices(c, request, n)
	}
	return true, emulateNonStreamChoices(c, request, n)
}

func emulateNonStreamChoices(c *gin.Context, request *dto.GeneralOpenAIRequest, n int) *types.NewAPIError {
	ctx, cancel := context.WithCancel(c.Request.Context())
	defer cancel()

	results := make([]dto.OpenAITextResponse, n)
	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(n)

	for i := 0; i < n; i++ {
		i := i
		g.Go(func() error {
			body, err := buildChoiceRequestBody(request, false, i)
			if err != nil {
				return err
			}

			resp, err := doSelfChatCompletionRequest(ctx, c, body)
			if err != nil {
				return err
			}
			defer service.CloseResponseBodyGracefully(resp)

			if resp.StatusCode != http.StatusOK {
				responseBody, _ := io.ReadAll(resp.Body)
				return fmt.Errorf("choice %d upstream status %d: %s", i, resp.StatusCode, strings.TrimSpace(string(responseBody)))
			}

			responseBody, err := io.ReadAll(resp.Body)
			if err != nil {
				return err
			}

			var out dto.OpenAITextResponse
			if err := common.Unmarshal(responseBody, &out); err != nil {
				return fmt.Errorf("choice %d bad response: %w", i, err)
			}
			if len(out.Choices) == 0 {
				return fmt.Errorf("choice %d returned no choices", i)
			}
			results[i] = out
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return types.NewOpenAIError(err, types.ErrorCodeBadResponse, http.StatusBadGateway)
	}

	merged := dto.OpenAITextResponse{
		Id:      helper.GetResponseID(c),
		Object:  "chat.completion",
		Created: common.GetTimestamp(),
		Model:   request.Model,
		Choices: make([]dto.OpenAITextResponseChoice, 0, n),
	}

	for i, result := range results {
		if merged.Model == "" {
			merged.Model = result.Model
		}
		choice := result.Choices[0]
		choice.Index = i
		merged.Choices = append(merged.Choices, choice)
		addUsage(&merged.Usage, result.Usage)
	}

	c.JSON(http.StatusOK, merged)
	return nil
}

func emulateStreamChoices(c *gin.Context, request *dto.GeneralOpenAIRequest, n int) *types.NewAPIError {
	ctx, cancel := context.WithCancel(c.Request.Context())
	defer cancel()

	type streamOpenResult struct {
		choiceIndex int
		resp        *http.Response
		err         error
	}

	openResults := make(chan streamOpenResult, n)
	for i := 0; i < n; i++ {
		i := i
		go func() {
			body, err := buildChoiceRequestBody(request, true, i)
			if err != nil {
				openResults <- streamOpenResult{choiceIndex: i, err: err}
				return
			}

			resp, err := doSelfChatCompletionRequest(ctx, c, body)
			if err != nil {
				openResults <- streamOpenResult{choiceIndex: i, err: err}
				return
			}
			if resp.StatusCode != http.StatusOK {
				responseBody, _ := io.ReadAll(resp.Body)
				service.CloseResponseBodyGracefully(resp)
				openResults <- streamOpenResult{choiceIndex: i, err: fmt.Errorf("choice %d upstream status %d: %s", i, resp.StatusCode, strings.TrimSpace(string(responseBody)))}
				return
			}
			openResults <- streamOpenResult{choiceIndex: i, resp: resp}
		}()
	}

	var writeMu sync.Mutex
	var headersOnce sync.Once
	var forwardWG sync.WaitGroup
	var firstErr error
	successes := 0

	for opened := 0; opened < n; opened++ {
		result := <-openResults
		if result.err != nil {
			if firstErr == nil {
				firstErr = result.err
			}
			logger.LogWarn(c, "multi-choice stream fan-out child failed: "+result.err.Error())
			continue
		}

		successes++
		forwardWG.Add(1)
		go func(result streamOpenResult) {
			defer forwardWG.Done()
			defer service.CloseResponseBodyGracefully(result.resp)
			headersOnce.Do(func() {
				writeMu.Lock()
				helper.SetEventStreamHeaders(c)
				writeMu.Unlock()
			})
			if err := forwardChoiceStream(c, result.resp.Body, result.choiceIndex, &writeMu); err != nil {
				logger.LogWarn(c, fmt.Sprintf("multi-choice stream fan-out choice %d forward error: %v", result.choiceIndex, err))
			}
		}(result)
	}

	if successes == 0 {
		if firstErr == nil {
			firstErr = fmt.Errorf("all multi-choice stream fan-out children failed")
		}
		return types.NewOpenAIError(firstErr, types.ErrorCodeBadResponse, http.StatusBadGateway)
	}

	forwardWG.Wait()

	writeMu.Lock()
	helper.Done(c)
	writeMu.Unlock()
	return nil
}

func buildChoiceRequestBody(request *dto.GeneralOpenAIRequest, stream bool, choiceIndex int) ([]byte, error) {
	child, err := common.DeepCopy(request)
	if err != nil {
		return nil, err
	}

	one := 1
	child.N = &one
	child.Stream = &stream
	if !stream {
		child.StreamOptions = nil
	}

	// If caller pins a seed, vary it per child. Otherwise several upstreams will
	// return identical candidates, which defeats SillyTavern multi-swipe.
	if child.Seed != nil {
		seed := *child.Seed + float64(choiceIndex)
		child.Seed = &seed
	}

	return common.Marshal(child)
}

func doSelfChatCompletionRequest(ctx context.Context, c *gin.Context, body []byte) (*http.Response, error) {
	url := selfChatCompletionURL(c)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	copySelfRequestHeaders(req.Header, c.Request.Header)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-NewAPI-Multi-Choice-Child", "1")

	return service.GetHttpClient().Do(req)
}

func selfChatCompletionURL(c *gin.Context) string {
	if baseURL := strings.TrimRight(common.GetEnvOrDefaultString(multiChoiceFanoutSelfBaseEnv, ""), "/"); baseURL != "" {
		uri := c.Request.URL.RequestURI()
		if uri == "" {
			uri = c.Request.URL.Path
		}
		return baseURL + uri
	}
	if baseURL := strings.TrimRight(common.GetEnvOrDefaultString(legacyMultiChoiceSelfBaseEnv, ""), "/"); baseURL != "" {
		uri := c.Request.URL.RequestURI()
		if uri == "" {
			uri = c.Request.URL.Path
		}
		return baseURL + uri
	}

	scheme := c.GetHeader("X-Forwarded-Proto")
	if scheme == "" {
		if c.Request.TLS != nil {
			scheme = "https"
		} else {
			scheme = "http"
		}
	}

	host := c.GetHeader("X-Forwarded-Host")
	if host == "" {
		host = c.Request.Host
	}

	uri := c.Request.URL.RequestURI()
	if uri == "" {
		uri = c.Request.URL.Path
	}
	return scheme + "://" + host + uri
}

func copySelfRequestHeaders(dst, src http.Header) {
	for key, values := range src {
		if strings.EqualFold(key, "Content-Length") || strings.EqualFold(key, "Host") {
			continue
		}
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func forwardChoiceStream(c *gin.Context, body io.Reader, choiceIndex int, writeMu *sync.Mutex) error {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, helper.InitialScannerBufferSize), helper.DefaultMaxScannerBufferSize)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}

		var chunk dto.ChatCompletionsStreamResponse
		if err := common.Unmarshal([]byte(data), &chunk); err != nil {
			return err
		}
		if len(chunk.Choices) == 0 {
			// Ignore per-child usage-only chunks. The children are already billed by
			// their own normal relay path, and SillyTavern does not need these chunks.
			continue
		}
		for i := range chunk.Choices {
			chunk.Choices[i].Index = choiceIndex
		}
		chunk.Id = helper.GetResponseID(c)
		chunk.Model = ""

		payload, err := common.Marshal(chunk)
		if err != nil {
			return err
		}

		writeMu.Lock()
		err = helper.StringData(c, string(payload))
		writeMu.Unlock()
		if err != nil {
			return err
		}
	}
	return scanner.Err()
}

func closeResponses(responses []*http.Response) {
	for _, resp := range responses {
		if resp != nil {
			service.CloseResponseBodyGracefully(resp)
		}
	}
}

func addUsage(total *dto.Usage, usage dto.Usage) {
	total.PromptTokens += usage.PromptTokens
	total.CompletionTokens += usage.CompletionTokens
	total.TotalTokens += usage.TotalTokens
	total.PromptCacheHitTokens += usage.PromptCacheHitTokens
	total.InputTokens += usage.InputTokens
	total.OutputTokens += usage.OutputTokens
	total.PromptTokensDetails.CachedTokens += usage.PromptTokensDetails.CachedTokens
	total.PromptTokensDetails.CachedCreationTokens += usage.PromptTokensDetails.CachedCreationTokens
	total.PromptTokensDetails.TextTokens += usage.PromptTokensDetails.TextTokens
	total.PromptTokensDetails.AudioTokens += usage.PromptTokensDetails.AudioTokens
	total.PromptTokensDetails.ImageTokens += usage.PromptTokensDetails.ImageTokens
	total.CompletionTokenDetails.TextTokens += usage.CompletionTokenDetails.TextTokens
	total.CompletionTokenDetails.AudioTokens += usage.CompletionTokenDetails.AudioTokens
	total.CompletionTokenDetails.ImageTokens += usage.CompletionTokenDetails.ImageTokens
	total.CompletionTokenDetails.ReasoningTokens += usage.CompletionTokenDetails.ReasoningTokens
	total.ClaudeCacheCreation5mTokens += usage.ClaudeCacheCreation5mTokens
	total.ClaudeCacheCreation1hTokens += usage.ClaudeCacheCreation1hTokens
}
