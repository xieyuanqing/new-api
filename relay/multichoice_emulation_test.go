package relay

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/dto"
	"github.com/QuantumNous/new-api/service"
	"github.com/gin-gonic/gin"
)

func TestBuildChoiceRequestBodySetsSingleChoiceAndVariesSeed(t *testing.T) {
	n := 3
	stream := false
	seed := 42.0
	req := &dto.GeneralOpenAIRequest{
		Model:  "test-model",
		N:      &n,
		Stream: &stream,
		Seed:   &seed,
	}

	body, err := buildChoiceRequestBody(req, true, 2)
	if err != nil {
		t.Fatalf("buildChoiceRequestBody returned error: %v", err)
	}

	var child dto.GeneralOpenAIRequest
	if err := common.Unmarshal(body, &child); err != nil {
		t.Fatalf("unmarshal child request: %v", err)
	}

	if child.N == nil || *child.N != 1 {
		t.Fatalf("child n = %v, want 1", child.N)
	}
	if child.Stream == nil || !*child.Stream {
		t.Fatalf("child stream = %v, want true", child.Stream)
	}
	if child.Seed == nil || *child.Seed != 44 {
		t.Fatalf("child seed = %v, want 44", child.Seed)
	}
}

func TestSelfChatCompletionURLUsesConfiguredBaseURL(t *testing.T) {
	t.Setenv(multiChoiceFanoutSelfBaseEnv, "http://127.0.0.1:3000/")
	t.Setenv(legacyMultiChoiceSelfBaseEnv, "")

	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "https://public.example/v1/chat/completions?x=1", nil)

	got := selfChatCompletionURL(c)
	want := "http://127.0.0.1:3000/v1/chat/completions?x=1"
	if got != want {
		t.Fatalf("selfChatCompletionURL() = %q, want %q", got, want)
	}
}

func TestForwardChoiceStreamRewritesChoiceIndex(t *testing.T) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)

	input := strings.Join([]string{
		`data: {"id":"chatcmpl-test","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}`,
		`data: [DONE]`,
		``,
	}, "\n")

	var mu sync.Mutex
	if err := forwardChoiceStream(c, strings.NewReader(input), 2, &mu); err != nil {
		t.Fatalf("forwardChoiceStream returned error: %v", err)
	}

	body := w.Body.String()
	if !strings.Contains(body, `"index":2`) {
		t.Fatalf("stream output does not contain rewritten index: %s", body)
	}
	if strings.Contains(body, `"index":0`) {
		t.Fatalf("stream output still contains original index: %s", body)
	}
	if !strings.Contains(body, "hello") {
		t.Fatalf("stream output lost content: %s", body)
	}
}

func TestEmulateStreamChoicesStreamsFastChoiceBeforeSlowFailingChoice(t *testing.T) {
	gin.SetMode(gin.TestMode)
	service.InitHttpClient()

	childServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read child body: %v", err)
		}
		var child dto.GeneralOpenAIRequest
		if err := common.Unmarshal(raw, &child); err != nil {
			t.Fatalf("unmarshal child body: %v", err)
		}
		seed := 0.0
		if child.Seed != nil {
			seed = *child.Seed
		}

		if seed == 43 {
			time.Sleep(350 * time.Millisecond)
			http.Error(w, "slow choice failed", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, `data: {"id":"child-fast","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"fast"},"finish_reason":null}]}`)
		fmt.Fprintln(w)
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		fmt.Fprintln(w, `data: [DONE]`)
		fmt.Fprintln(w)
	}))
	defer childServer.Close()
	t.Setenv(multiChoiceFanoutSelfBaseEnv, childServer.URL)
	t.Setenv(legacyMultiChoiceSelfBaseEnv, "")

	router := gin.New()
	router.POST("/v1/chat/completions", func(c *gin.Context) {
		n := 2
		stream := true
		seed := 42.0
		err := emulateStreamChoices(c, &dto.GeneralOpenAIRequest{
			Model:  "test-model",
			N:      &n,
			Stream: &stream,
			Seed:   &seed,
		}, n)
		if err != nil {
			status := err.StatusCode
			if status == 0 {
				status = http.StatusBadGateway
			}
			c.String(status, err.Error())
		}
	})
	outerServer := httptest.NewServer(router)
	defer outerServer.Close()

	start := time.Now()
	resp, err := http.Post(outerServer.URL+"/v1/chat/completions", "application/json", bytes.NewReader([]byte(`{}`)))
	if err != nil {
		t.Fatalf("post outer stream: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("outer status = %d, want 200, body=%s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	line, err := bufio.NewReader(resp.Body).ReadString('\n')
	if err != nil {
		t.Fatalf("read first stream line: %v", err)
	}
	elapsed := time.Since(start)
	if elapsed > 200*time.Millisecond {
		t.Fatalf("first stream line took %s, want under 200ms; line=%s", elapsed, line)
	}
	if !strings.Contains(line, `"index":0`) || !strings.Contains(line, "fast") {
		t.Fatalf("unexpected first stream line: %s", line)
	}
}

func TestAddUsageSumsTokenFields(t *testing.T) {
	total := dto.Usage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3}
	addUsage(&total, dto.Usage{
		PromptTokens:     4,
		CompletionTokens: 5,
		TotalTokens:      9,
		PromptTokensDetails: dto.InputTokenDetails{
			CachedTokens: 7,
			TextTokens:   8,
		},
		CompletionTokenDetails: dto.OutputTokenDetails{
			ReasoningTokens: 11,
		},
	})

	if total.PromptTokens != 5 || total.CompletionTokens != 7 || total.TotalTokens != 12 {
		t.Fatalf("summed usage = %+v, token totals mismatch", total)
	}
	if total.PromptTokensDetails.CachedTokens != 7 || total.PromptTokensDetails.TextTokens != 8 {
		t.Fatalf("prompt token details not summed: %+v", total.PromptTokensDetails)
	}
	if total.CompletionTokenDetails.ReasoningTokens != 11 {
		t.Fatalf("completion token details not summed: %+v", total.CompletionTokenDetails)
	}
}
