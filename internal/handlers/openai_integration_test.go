package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/k0in/openai-ollama-proxy/internal/config"
	"github.com/k0in/openai-ollama-proxy/internal/stats"
	"github.com/k0in/openai-ollama-proxy/internal/types"
)

// TestOpenAIChat_NonStream verifies a complete non-streaming OpenAI chat
// completion request and response through the proxy.
func TestOpenAIChat_NonStream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer openai-key" {
			t.Errorf("Authorization = %q", r.Header.Get("Authorization"))
		}

		var req types.OpenAIChatRequest
		json.NewDecoder(r.Body).Decode(&req)
		if req.Model != "gpt-4o-upstream" {
			t.Errorf("upstream model = %q, want %q", req.Model, "gpt-4o-upstream")
		}
		if req.Stream {
			t.Error("upstream stream should be false for non-stream request")
		}
		if len(req.Messages) != 1 {
			t.Errorf("len(messages) = %d, want 1", len(req.Messages))
		}

		content := "Hello from GPT-4o!"
		stop := "stop"
		json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			ID:      "chatcmpl-456",
			Object:  "chat.completion",
			Model:   "gpt-4o-upstream",
			Created: 1700000001,
			Choices: []types.OpenAIChoice{{
				Index:        0,
				Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
				FinishReason: &stop,
			}},
			Usage: &types.OpenAIUsage{PromptTokens: 8, CompletionTokens: 4, TotalTokens: 12},
		})
	}))
	defer upstream.Close()

	router, err := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL:    upstream.URL,
			APIKey: "openai-key",
			Models: []config.ModelMapping{
				{Upstream: "gpt-4o-upstream", Local: "gpt-4o", ContextLength: 128000},
			},
		},
	}, 128000)
	if err != nil {
		t.Fatalf("BuildRoutingTable: %v", err)
	}

	cfg := config.Config{
		ListenAddr:            ":11434",
		ModelContextLength:    128000,
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	server := NewWithClients(cfg, router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, nil)

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"Hello!"}],"stream":false}`

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleOpenAIChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}

	contentType := w.Result().Header.Get("Content-Type")
	if !strings.HasPrefix(contentType, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", contentType)
	}

	var resp types.OpenAIChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	// The proxy normalizes the model name back to the local name
	if resp.Model != "gpt-4o" {
		t.Errorf("model = %q, want %q", resp.Model, "gpt-4o")
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("len(choices) = %d, want 1", len(resp.Choices))
	}
	if resp.Choices[0].Message == nil {
		t.Fatal("choice.Message is nil")
	}
	if resp.Choices[0].Message.Content == nil {
		t.Fatal("choice.Message.Content is nil")
	}
	if *resp.Choices[0].Message.Content != "Hello from GPT-4o!" {
		t.Errorf("content = %q, want %q", *resp.Choices[0].Message.Content, "Hello from GPT-4o!")
	}
	if resp.Choices[0].FinishReason == nil || *resp.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %v, want 'stop'", resp.Choices[0].FinishReason)
	}
	if resp.Usage == nil {
		t.Fatal("usage is nil")
	}
	if resp.Usage.PromptTokens != 8 {
		t.Errorf("prompt_tokens = %d, want 8", resp.Usage.PromptTokens)
	}
	if resp.Usage.CompletionTokens != 4 {
		t.Errorf("completion_tokens = %d, want 4", resp.Usage.CompletionTokens)
	}
}

// TestOpenAIChat_Stream verifies streaming chat completions are proxied
// correctly through the OpenAI-compatible endpoint.
func TestOpenAIChat_Stream(t *testing.T) {
	sseData := strings.Join([]string{
		`data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}`,
		``,
		`data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"}}]}`,
		``,
		`data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" world"}}]}`,
		``,
		`data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(sseData))
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: upstream.URL,
			Models: []config.ModelMapping{
				{Upstream: "gpt-upstream", Local: "gpt-4o-mini"},
			},
		},
	}, 65536)

	cfg := config.Config{
		ListenAddr:            ":11434",
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	server := NewWithClients(cfg, router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, nil)

	body := `{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hi"}],"stream":true}`

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleOpenAIChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}

	contentType := w.Result().Header.Get("Content-Type")
	if !strings.HasPrefix(contentType, "text/event-stream") {
		t.Fatalf("Content-Type = %q, want text/event-stream", contentType)
	}

	lines := strings.Split(strings.TrimSpace(w.Body.String()), "\n")
	if len(lines) < 3 {
		t.Fatalf("expected at least 3 SSE lines, got %d", len(lines))
	}

	// The proxy normalizes model names in the stream. Check a content chunk.
	var foundContent bool
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") && !strings.Contains(line, "[DONE]") {
			data := strings.TrimPrefix(line, "data: ")
			var chunk types.OpenAIChatResponse
			if err := json.Unmarshal([]byte(data), &chunk); err == nil {
				if len(chunk.Choices) > 0 && chunk.Choices[0].Delta != nil && chunk.Choices[0].Delta.Content != nil {
					foundContent = true
					break
				}
			}
		}
	}
	if !foundContent {
		t.Error("no content delta found in stream")
	}
}

// TestOpenAIChat_MethodNotAllowed verifies non-POST requests return 405.
func TestOpenAIChat_MethodNotAllowed(t *testing.T) {
	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: "http://localhost:8000", Models: []config.ModelMapping{{Upstream: "m", Local: "gpt-4o"}}},
	}, 65536)
	server := New(config.Config{ListenAddr: ":11434"}, router, &http.Client{Timeout: 5 * time.Second}, stats.New())

	req := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	server.handleOpenAIChat(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", w.Code)
	}
}

// TestOpenAIModels_List verifies /v1/models returns all models from the router.
func TestOpenAIModels_List(t *testing.T) {
	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: "http://localhost:8000",
			Models: []config.ModelMapping{
				{Upstream: "gpt-4o-upstream", Local: "gpt-4o", ContextLength: 128000},
				{Upstream: "gpt-4o-mini-upstream", Local: "gpt-4o-mini", ContextLength: 128000},
			},
		},
	}, 65536)

	server := New(config.Config{ListenAddr: ":11434", ModelContextLength: 65536}, router, &http.Client{Timeout: 5 * time.Second}, stats.New())

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	server.handleOpenAIModels(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var resp types.OpenAIModelListResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if resp.Object != "list" {
		t.Errorf("object = %q, want %q", resp.Object, "list")
	}

	gotModels := make(map[string]bool)
	for _, m := range resp.Data {
		gotModels[m.ID] = true
		if m.Object != "model" {
			t.Errorf("model %s object = %q, want %q", m.ID, m.Object, "model")
		}
		if m.MaxModelLen <= 0 {
			t.Errorf("model %s max_model_len = %d, want > 0", m.ID, m.MaxModelLen)
		}
	}
	for _, want := range []string{"gpt-4o", "gpt-4o-mini"} {
		if !gotModels[want] {
			t.Errorf("model %q missing from /v1/models", want)
		}
	}
	if len(resp.Data) != 2 {
		t.Errorf("len(models) = %d, want 2", len(resp.Data))
	}
}

// TestOpenAIModels_MethodNotAllowed verifies non-GET requests return 405.
func TestOpenAIModels_MethodNotAllowed(t *testing.T) {
	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: "http://localhost:8000", Models: []config.ModelMapping{{Upstream: "m", Local: "gpt-4o"}}},
	}, 65536)
	server := New(config.Config{ListenAddr: ":11434"}, router, &http.Client{Timeout: 5 * time.Second}, stats.New())

	req := httptest.NewRequest(http.MethodPost, "/v1/models", nil)
	w := httptest.NewRecorder()
	server.handleOpenAIModels(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", w.Code)
	}
}

// TestOpenAIEmbeddings verifies the /v1/embeddings endpoint routes correctly
// and returns embeddings from the upstream.
func TestOpenAIEmbeddings(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer embed-key" {
			t.Errorf("Authorization = %q", r.Header.Get("Authorization"))
		}

		var req types.OpenAIEmbedRequest
		json.NewDecoder(r.Body).Decode(&req)
		if req.Model != "text-embedding-upstream" {
			t.Errorf("upstream model = %q, want %q", req.Model, "text-embedding-upstream")
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(types.OpenAIEmbedResponse{
			Object: "list",
			Data: []types.OpenAIEmbedData{
				{Object: "embedding", Embedding: []float64{0.1, 0.2, 0.3}, Index: 0},
			},
			Model: "text-embedding-upstream",
			Usage: &types.OpenAIUsage{PromptTokens: 5, CompletionTokens: 0, TotalTokens: 5},
		})
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: upstream.URL, APIKey: "embed-key",
			Models: []config.ModelMapping{
				{Upstream: "text-embedding-upstream", Local: "text-embedding-3-small"},
			},
		},
	}, 8192)

	server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	body := `{"model":"text-embedding-3-small","input":"Hello world"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleOpenAIEmbeddings(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}

	var resp types.OpenAIEmbedResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(resp.Data) != 1 {
		t.Fatalf("len(data) = %d, want 1", len(resp.Data))
	}
	if len(resp.Data[0].Embedding) != 3 {
		t.Errorf("len(embedding) = %d, want 3", len(resp.Data[0].Embedding))
	}
}

// TestOpenAIChat_UpstreamError verifies upstream errors are propagated
// correctly with proper status codes.
func TestOpenAIChat_UpstreamError(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		json.NewEncoder(w).Encode(map[string]string{"error": "rate limited"})
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "gpt-4o"}}},
	}, 65536)
	server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		defer close(done)
		server.handleOpenAIChat(w, req)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("handler hung")
	}

	// 429 maps to 503 from the upstream (503 handling in retry loop)
	if w.Code != http.StatusServiceUnavailable && w.Code != http.StatusTooManyRequests {
		t.Errorf("status = %d, want 503 or 429 (body=%s)", w.Code, w.Body.String())
	}
}

// TestOpenAIChat_StreamWithUsage verifies stream options with usage included.
func TestOpenAIChat_StreamWithUsage(t *testing.T) {
	sseData := strings.Join([]string{
		`data: {"id":"1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"delta":{"content":" there"}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"finish_reason":"stop","delta":{}}]}`,
		``,
		`data: {"id":"1","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(sseData))
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "gpt-4o"}}},
	}, 65536)
	server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}],"stream":true,"stream_options":{"include_usage":true}}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		defer close(done)
		server.handleOpenAIChat(w, req)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("handler hung")
	}

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	// Verify we got SSE data with content
	bodyStr := w.Body.String()
	if !strings.Contains(bodyStr, "Hi") {
		t.Errorf("response should contain content 'Hi', got: %s", bodyStr)
	}
}
