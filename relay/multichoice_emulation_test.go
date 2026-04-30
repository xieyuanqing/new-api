package relay

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/dto"
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
