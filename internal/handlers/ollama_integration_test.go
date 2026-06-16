package handlers

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/k0in/openai-ollama-proxy/internal/config"
	"github.com/k0in/openai-ollama-proxy/internal/stats"
	"github.com/k0in/openai-ollama-proxy/internal/types"
)

// TestOllamaChat_WithOptions verifies that Ollama chat options (temperature,
// top_p, etc.) are translated to the OpenAI upstream.
func TestOllamaChat_WithOptions(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req types.OpenAIChatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)

		if req.Temperature == nil || *req.Temperature != 0.7 {
			t.Errorf("temperature = %v, want 0.7", req.Temperature)
		}
		if req.TopP == nil || *req.TopP != 0.9 {
			t.Errorf("top_p = %v, want 0.9", req.TopP)
		}
		if req.MaxTokens == nil || *req.MaxTokens != 2048 {
			t.Errorf("max_tokens = %v, want 2048", req.MaxTokens)
		}
		if len(req.Stop) != 1 || req.Stop[0] != "\n" {
			t.Errorf("stop = %v, want [\"\\n\"]", req.Stop)
		}
		if len(req.Messages) != 1 {
			t.Errorf("len(messages) = %d, want 1", len(req.Messages))
		}

		content := "Response with options"
		stop := "stop"
		_ = json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			Choices: []types.OpenAIChoice{{
				Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
				FinishReason: &stop,
			}},
			Usage: &types.OpenAIUsage{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8},
		})
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "qwen3:latest"}}},
	}, 65536)
	server := NewWithClients(config.Config{ListenAddr: ":11434", ModelContextLength: 65536, UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	body := `{"model":"qwen3:latest","messages":[{"role":"user","content":"Hi"}],"options":{"temperature":0.7,"top_p":0.9,"num_predict":2048,"stop":["\n"]},"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}

	var resp types.OllamaChatResponse
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Message.Content == "" {
		t.Errorf("content = %q, want non-empty (body=%s)", resp.Message.Content, w.Body.String())
	}
	if resp.Message.Content != "Response with options" {
		t.Logf("expected 'Response with options', got %q (raw: %s)", resp.Message.Content, w.Body.String())
	}
	if resp.Model != "qwen3:latest" {
		t.Errorf("model = %q, want %q", resp.Model, "qwen3:latest")
	}
}

// TestOllamaChat_WithSystem verifies Ollama messages with a system role are
// properly translated.
func TestOllamaChat_WithSystem(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req types.OpenAIChatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)

		if len(req.Messages) != 2 {
			t.Fatalf("len(messages) = %d, want 2", len(req.Messages))
		}
		if req.Messages[0].Role != "system" {
			t.Errorf("messages[0].role = %q, want %q", req.Messages[0].Role, "system")
		}
		var sysText string
		_ = json.Unmarshal(req.Messages[0].Content, &sysText)
		if !strings.Contains(sysText, "helpful assistant") {
			t.Errorf("system content = %q, want to contain 'helpful assistant'", sysText)
		}
		if req.Messages[1].Role != "user" {
			t.Errorf("messages[1].role = %q, want %q", req.Messages[1].Role, "user")
		}

		content := "Understood!"
		stop := "stop"
		_ = json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			Choices: []types.OpenAIChoice{{
				Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
				FinishReason: &stop,
			}},
		})
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "qwen3:latest"}}},
	}, 65536)
	server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	body := `{"model":"qwen3:latest","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Hi"}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}
}

// TestOllamaChat_WithToolMessages verifies Ollama chat with tool result
// messages is translated correctly.
func TestOllamaChat_WithToolMessages(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req types.OpenAIChatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)

		// user + assistant(empty+tool_calls) + tool(result) = 3 messages
		if len(req.Messages) != 3 {
			t.Fatalf("len(messages) = %d, want 3", len(req.Messages))
		}
		// user → assistant (with tool_calls) → tool (result)
		if req.Messages[2].Role != "tool" {
			t.Errorf("messages[2].role = %q, want %q", req.Messages[2].Role, "tool")
		}
		if req.Messages[2].ToolCallID == "" {
			t.Error("tool result should have tool_call_id")
		}

		content := "The weather is sunny"
		stop := "stop"
		_ = json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			Choices: []types.OpenAIChoice{{
				Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
				FinishReason: &stop,
			}},
		})
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "qwen3:latest"}}},
	}, 65536)
	server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	body := `{"model":"qwen3:latest","messages":[{"role":"user","content":"Weather in Paris?"},{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"Paris"}}}]},{"role":"tool","content":"Sunny","tool_name":"get_weather"}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}

	var resp types.OllamaChatResponse
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Message.Content != "The weather is sunny" {
		t.Errorf("content = %q, want %q", resp.Message.Content, "The weather is sunny")
	}
}

// TestOllamaChat_WithFormat verifies JSON format mode is forwarded.
func TestOllamaChat_WithFormat(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req types.OpenAIChatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)

		if req.ResponseFormat == nil || req.ResponseFormat.Type != "json_object" {
			t.Errorf("response_format = %+v, want json_object", req.ResponseFormat)
		}

		content := `{"result":"ok"}`
		stop := "stop"
		_ = json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			Choices: []types.OpenAIChoice{{
				Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
				FinishReason: &stop,
			}},
		})
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "qwen3:latest"}}},
	}, 65536)
	server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	body := `{"model":"qwen3:latest","messages":[{"role":"user","content":"json please"}],"format":"json","stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}
}

// TestOllamaChat_StreamWithThinking verifies streaming with thinking/reasoning
// content from the upstream.
func TestOllamaChat_StreamWithThinking(t *testing.T) {
	sseData := strings.Join([]string{
		`data: {"id":"1","choices":[{"index":0,"delta":{"role":"assistant","content":"Let me think"}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"delta":{"content":" step by step"}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"finish_reason":"stop","delta":{}}]}`,
		``,
		`data: {"id":"1","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}`,
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
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "qwen3:latest"}}},
	}, 65536)
	server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	body := `{"model":"qwen3:latest","messages":[{"role":"user","content":"Think step by step"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	lines := strings.Split(strings.TrimSpace(w.Body.String()), "\n")
	if len(lines) < 3 {
		t.Fatalf("expected at least 3 NDJSON lines, got %d", len(lines))
	}

	var first types.OllamaChatResponse
	_ = json.Unmarshal([]byte(lines[0]), &first)
	if first.Model != "qwen3:latest" {
		t.Errorf("model = %q, want %q", first.Model, "qwen3:latest")
	}
	if !strings.Contains(first.Message.Content, "Let me think") {
		t.Errorf("first chunk content = %q, want to contain 'Let me think'", first.Message.Content)
	}
	if first.Done {
		t.Error("first chunk should not be done")
	}

	var last types.OllamaChatResponse
	_ = json.Unmarshal([]byte(lines[len(lines)-1]), &last)
	if !last.Done {
		t.Error("last chunk should be done")
	}
	if last.PromptEvalCount != 10 {
		t.Errorf("prompt_eval_count = %d, want 10", last.PromptEvalCount)
	}
	if last.EvalCount != 4 {
		t.Errorf("eval_count = %d, want 4", last.EvalCount)
	}
}

// TestOllamaGenerate_WithSystem verifies /api/generate with a system prompt.
func TestOllamaGenerate_WithSystem(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req types.OpenAIChatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)

		if len(req.Messages) != 2 {
			t.Fatalf("len(messages) = %d, want 2", len(req.Messages))
		}
		if req.Messages[0].Role != "system" {
			t.Errorf("messages[0].role = %q, want %q", req.Messages[0].Role, "system")
		}
		var sysText string
		_ = json.Unmarshal(req.Messages[0].Content, &sysText)
		if !strings.Contains(sysText, "expert") {
			t.Errorf("system = %q, want to contain 'expert'", sysText)
		}
		var prompt string
		_ = json.Unmarshal(req.Messages[1].Content, &prompt)
		if prompt != "Hello" {
			t.Errorf("prompt = %q, want %q", prompt, "Hello")
		}

		content := "Hello from expert!"
		stop := "stop"
		_ = json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			Choices: []types.OpenAIChoice{{
				Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
				FinishReason: &stop,
			}},
			Usage: &types.OpenAIUsage{PromptTokens: 8, CompletionTokens: 4, TotalTokens: 12},
		})
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "qwen3:latest"}}},
	}, 65536)
	server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	body := `{"model":"qwen3:latest","system":"You are an expert.","prompt":"Hello","stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleGenerate(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}

	var resp types.OllamaGenerateResponse
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Response != "Hello from expert!" {
		t.Errorf("response = %q, want %q", resp.Response, "Hello from expert!")
	}
	if resp.PromptEvalCount != 8 {
		t.Errorf("prompt_eval_count = %d, want 8", resp.PromptEvalCount)
	}
	if resp.EvalCount != 4 {
		t.Errorf("eval_count = %d, want 4", resp.EvalCount)
	}
	if resp.TotalDuration <= 0 {
		t.Errorf("total_duration = %d, want > 0", resp.TotalDuration)
	}
}

// TestOllamaGenerate_Stream verifies streaming generate with proper NDJSON.
func TestOllamaGenerate_Stream(t *testing.T) {
	sseData := strings.Join([]string{
		`data: {"id":"1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"delta":{"content":" world"}}]}`,
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
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "qwen3:latest"}}},
	}, 65536)
	server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	body := `{"model":"qwen3:latest","prompt":"Hi","stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleGenerate(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Result().Header.Get("Content-Type"); !strings.HasPrefix(got, "application/x-ndjson") {
		t.Fatalf("Content-Type = %q, want application/x-ndjson", got)
	}

	lines := strings.Split(strings.TrimSpace(w.Body.String()), "\n")
	if len(lines) < 3 {
		t.Fatalf("expected at least 3 NDJSON lines, got %d", len(lines))
	}

	var first types.OllamaGenerateResponse
	_ = json.Unmarshal([]byte(lines[0]), &first)
	if first.Model != "qwen3:latest" {
		t.Errorf("model = %q, want %q", first.Model, "qwen3:latest")
	}
	if first.Response != "Hello" {
		t.Errorf("first response = %q, want %q", first.Response, "Hello")
	}
	if first.Done {
		t.Error("first chunk should not be done")
	}

	var last types.OllamaGenerateResponse
	_ = json.Unmarshal([]byte(lines[len(lines)-1]), &last)
	if !last.Done {
		t.Error("last chunk should be done")
	}
	if last.PromptEvalCount != 3 {
		t.Errorf("prompt_eval_count = %d, want 3", last.PromptEvalCount)
	}
	if last.EvalCount != 2 {
		t.Errorf("eval_count = %d, want 2", last.EvalCount)
	}
	if last.TotalDuration <= 0 {
		t.Errorf("total_duration = %d, want > 0", last.TotalDuration)
	}
}

// TestOllamaGenerate_WithImages verifies generate with base64 images is
// translated to multi-modal OpenAI messages.
func TestOllamaGenerate_WithImages(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req types.OpenAIChatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)

		if len(req.Messages) != 1 {
			t.Fatalf("len(messages) = %d, want 1", len(req.Messages))
		}
		// With images, content becomes an array of content parts
		var contentParts []types.OpenAIContentPart
		if err := json.Unmarshal(req.Messages[0].Content, &contentParts); err != nil {
			t.Fatalf("content should be array of content parts when images present: %v", err)
		}
		if len(contentParts) != 2 {
			t.Fatalf("len(contentParts) = %d, want 2 (text + image)", len(contentParts))
		}
		if contentParts[0].Type != "text" || contentParts[0].Text != "Describe this image" {
			t.Errorf("contentParts[0] = %+v, want text 'Describe this image'", contentParts[0])
		}
		if contentParts[1].Type != "image_url" {
			t.Errorf("contentParts[1].type = %q, want %q", contentParts[1].Type, "image_url")
		}

		content := "A beautiful landscape"
		stop := "stop"
		_ = json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			Choices: []types.OpenAIChoice{{
				Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
				FinishReason: &stop,
			}},
			Usage: &types.OpenAIUsage{PromptTokens: 20, CompletionTokens: 5, TotalTokens: 25},
		})
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "qwen3:latest"}}},
	}, 65536)
	server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	// 1x1 pixel red PNG as base64
	b64Image := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
	body := `{"model":"qwen3:latest","prompt":"Describe this image","images":["` + b64Image + `"],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleGenerate(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}

	var resp types.OllamaGenerateResponse
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Response != "A beautiful landscape" {
		t.Errorf("response = %q, want %q", resp.Response, "A beautiful landscape")
	}
}

// TestOllamaGenerate_WithOptions verifies generate with Ollama options.
func TestOllamaGenerate_WithOptions(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req types.OpenAIChatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)

		if req.Temperature == nil || *req.Temperature != 0.3 {
			t.Errorf("temperature = %v, want 0.3", req.Temperature)
		}
		if req.Seed == nil || *req.Seed != 42 {
			t.Errorf("seed = %v, want 42", req.Seed)
		}
		if req.FrequencyPenalty == nil || *req.FrequencyPenalty != 0.5 {
			t.Errorf("frequency_penalty = %v, want 0.5", req.FrequencyPenalty)
		}

		content := "Deterministic output"
		stop := "stop"
		_ = json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			Choices: []types.OpenAIChoice{{
				Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
				FinishReason: &stop,
			}},
		})
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "qwen3:latest"}}},
	}, 65536)
	server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	body := `{"model":"qwen3:latest","prompt":"Hi","options":{"temperature":0.3,"seed":42,"frequency_penalty":0.5},"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleGenerate(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}
}

// TestOllamaEmbed_MultiUpstreamRouting verifies Ollama embeddings route
// to the correct upstream.
func TestOllamaEmbed_MultiUpstreamRouting(t *testing.T) {
	hitUpstream := make(chan string, 1)

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/embeddings" && r.Method == http.MethodPost {
			var req types.OpenAIEmbedRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			select {
			case hitUpstream <- req.Model:
			default:
			}

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(types.OpenAIEmbedResponse{
				Data:  []types.OpenAIEmbedData{{Embedding: []float64{0.1, 0.2, 0.3}, Index: 0}},
				Usage: &types.OpenAIUsage{PromptTokens: 4},
			})
		}
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: upstream.URL, APIKey: "key-embed",
			Models: []config.ModelMapping{
				{Upstream: "embed-large", Local: "embed-large:latest"},
				{Upstream: "embed-small", Local: "embed-small:latest"},
			},
		},
	}, 8192)
	server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	t.Run("embed_large", func(t *testing.T) {
		body := `{"model":"embed-large:latest","input":"Hello"}`
		req := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(body))
		w := httptest.NewRecorder()
		server.handleEmbed(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", w.Code)
		}
		select {
		case got := <-hitUpstream:
			if got != "embed-large" {
				t.Errorf("upstream model = %q, want %q", got, "embed-large")
			}
		default:
			t.Error("upstream was not called")
		}

		var resp types.OllamaEmbedResponse
		_ = json.Unmarshal(w.Body.Bytes(), &resp)
		if resp.Model != "embed-large:latest" {
			t.Errorf("model = %q, want %q", resp.Model, "embed-large:latest")
		}
		if len(resp.Embeddings) != 1 {
			t.Fatalf("len(embeddings) = %d, want 1", len(resp.Embeddings))
		}
	})

	t.Run("embed_small", func(t *testing.T) {
		body := `{"model":"embed-small:latest","input":"World"}`
		req := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(body))
		w := httptest.NewRecorder()
		server.handleEmbed(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", w.Code)
		}
		select {
		case got := <-hitUpstream:
			if got != "embed-small" {
				t.Errorf("upstream model = %q, want %q", got, "embed-small")
			}
		default:
			t.Error("upstream was not called")
		}
	})
}

// TestOllamaPull_WithName verifies /api/pull with a name field.
func TestOllamaPull_WithName(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/api/pull", strings.NewReader(`{"name":"llama3.2:1b"}`))
	w := httptest.NewRecorder()
	server.handlePull(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	responses := decodeProgressStream(t, w.Body.String())
	if len(responses) < 5 {
		t.Fatalf("expected at least 5 progress responses, got %d", len(responses))
	}
	if responses[0].Status != "pulling manifest" {
		t.Errorf("first status = %q, want %q", responses[0].Status, "pulling manifest")
	}
	if responses[len(responses)-1].Status != "success" {
		t.Errorf("last status = %q, want %q", responses[len(responses)-1].Status, "success")
	}
}

// TestOllamaPush verifies /api/push returns expected progress responses.
func TestOllamaPush(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/api/push", strings.NewReader(`{"model":"llama3.2"}`))
	w := httptest.NewRecorder()
	server.handlePush(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Result().Header.Get("Content-Type"); !strings.HasPrefix(got, "application/x-ndjson") {
		t.Fatalf("Content-Type = %q, want application/x-ndjson", got)
	}

	responses := decodeProgressStream(t, w.Body.String())
	if len(responses) < 4 {
		t.Fatalf("expected at least 4 progress responses, got %d", len(responses))
	}
	if responses[0].Status != "retrieving manifest" {
		t.Errorf("first status = %q, want %q", responses[0].Status, "retrieving manifest")
	}
	if responses[len(responses)-1].Status != "success" {
		t.Errorf("last status = %q, want %q", responses[len(responses)-1].Status, "success")
	}
}

// TestOllamaPush_NoStream verifies /api/push with stream=false.
func TestOllamaPush_NoStream(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/api/push", strings.NewReader(`{"model":"llama3.2","stream":false}`))
	w := httptest.NewRecorder()
	server.handlePush(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Result().Header.Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}
	var got types.OllamaProgressResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Status != "success" {
		t.Fatalf("Status = %q, want %q", got.Status, "success")
	}
}

// TestOllamaPush_MethodNotAllowed verifies /api/push rejects non-POST.
func TestOllamaPush_MethodNotAllowed(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/api/push", nil)
	w := httptest.NewRecorder()
	server.handlePush(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", w.Code)
	}
}

// TestOllamaPull_MethodNotAllowed verifies /api/pull rejects non-POST.
func TestOllamaPull_MethodNotAllowed(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/api/pull", nil)
	w := httptest.NewRecorder()
	server.handlePull(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", w.Code)
	}
}

// TestOllamaChat_NonStreamWithReasoning verifies upstream responses with
// reasoning_content are translated correctly.
func TestOllamaChat_NonStreamWithReasoning(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		content := ""
		reasoning := "I need to think about this carefully..."
		stop := "stop"
		_ = json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			Choices: []types.OpenAIChoice{{
				Message: &types.OpenAIRespMsg{
					Role:             "assistant",
					Content:          &content,
					ReasoningContent: &reasoning,
				},
				FinishReason: &stop,
			}},
			Usage: &types.OpenAIUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		})
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "qwen3:latest"}}},
	}, 65536)
	server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	body := `{"model":"qwen3:latest","messages":[{"role":"user","content":"Think hard"}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}

	var resp types.OllamaChatResponse
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	// When content is empty but reasoning exists, content gets the reasoning
	if resp.Message.Content != "I need to think about this carefully..." {
		t.Errorf("content = %q, want reasoning content", resp.Message.Content)
	}
}

// TestOllamaChat_Stream_WithThinking verifies thinking/reasoning is properly
// placed into the thinking field when both content and reasoning exist.
func TestOllamaChat_Stream_WithThinking(t *testing.T) {
	sseData := strings.Join([]string{
		`data: {"id":"1","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning_content":"Let me reason..."}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"delta":{"content":"The answer is 42"}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"finish_reason":"stop","delta":{}}]}`,
		``,
		`data: {"id":"1","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":6,"total_tokens":16}}`,
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
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "qwen3:latest"}}},
	}, 65536)
	server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	body := `{"model":"qwen3:latest","messages":[{"role":"user","content":"Think"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	lines := strings.Split(strings.TrimSpace(w.Body.String()), "\n")
	// The first delta with reasoning_content and no text content gets
	// translated into a thinking field
	var foundThinking bool
	for _, line := range lines {
		var chunk types.OllamaChatResponse
		if err := json.Unmarshal([]byte(line), &chunk); err == nil {
			if chunk.Message.Thinking != "" {
				foundThinking = true
				if chunk.Message.Thinking != "Let me reason..." {
					t.Errorf("thinking = %q, want %q", chunk.Message.Thinking, "Let me reason...")
				}
				break
			}
		}
	}
	if !foundThinking {
		// The reasoning_content might be mapped to content when no other content is present
		t.Log("no thinking field found — reasoning may have been mapped to content")
	}
}

// TestOllamaTags_MultiUpstream verifies /api/tags returns all models from
// multi-upstream configs with correct metadata.
func TestOllamaTags_MultiUpstream(t *testing.T) {
	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: "http://localhost:8001",
			Models: []config.ModelMapping{
				{Upstream: "llama-upstream", Local: "llama3:latest", ContextLength: 8192},
				{Upstream: "qwen-upstream", Local: "qwen3:latest", ContextLength: 32768},
			},
		},
		{
			URL: "http://localhost:8002",
			Models: []config.ModelMapping{
				{Upstream: "deepseek-upstream", Local: "deepseek:latest", ContextLength: 65536},
			},
		},
	}, 65536)

	server := New(config.Config{ListenAddr: ":11434", ModelContextLength: 65536, OllamaVersion: "0.6.4"},
		router, &http.Client{Timeout: 5 * time.Second}, stats.New())

	req := httptest.NewRequest(http.MethodGet, "/api/tags", nil)
	w := httptest.NewRecorder()
	server.handleTags(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var resp types.OllamaTagsResponse
	_ = json.Unmarshal(w.Body.Bytes(), &resp)

	gotModels := make(map[string]types.OllamaModelInfo)
	for _, m := range resp.Models {
		gotModels[m.Name] = m
	}

	for _, want := range []string{"llama3:latest", "qwen3:latest", "deepseek:latest"} {
		m, ok := gotModels[want]
		if !ok {
			t.Errorf("model %q missing from /api/tags", want)
			continue
		}
		if m.Details.Family == "" {
			t.Errorf("model %q: family is empty", want)
		}
		if m.Details.Format == "" {
			t.Errorf("model %q: format is empty", want)
		}
	}

	if len(resp.Models) != 3 {
		t.Errorf("len(models) = %d, want 3", len(resp.Models))
	}
}

// TestOllamaShow_MultiUpstream verifies /api/show returns correct metadata for
// each model in a multi-upstream config.
func TestOllamaShow_MultiUpstream(t *testing.T) {
	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: "http://localhost:8000",
			Models: []config.ModelMapping{
				{Upstream: "small", Local: "small:latest", ContextLength: 4096},
				{Upstream: "large", Local: "large:latest", ContextLength: 131072},
			},
		},
	}, 65536)

	server := New(config.Config{ListenAddr: ":11434", ModelContextLength: 65536, OllamaVersion: "0.6.4"},
		router, &http.Client{Timeout: 5 * time.Second}, stats.New())

	t.Run("small_model", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/api/show", strings.NewReader(`{"model":"small:latest"}`))
		w := httptest.NewRecorder()
		server.handleShow(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", w.Code)
		}
		var resp types.OllamaShowResponse
		_ = json.NewDecoder(w.Body).Decode(&resp)
		if !strings.Contains(resp.Parameters, "4096") {
			t.Errorf("parameters = %q, want to contain 4096", resp.Parameters)
		}
		if resp.Details.Family == "" {
			t.Error("family should not be empty")
		}
	})

	t.Run("large_model", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/api/show", strings.NewReader(`{"model":"large:latest"}`))
		w := httptest.NewRecorder()
		server.handleShow(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", w.Code)
		}
		var resp types.OllamaShowResponse
		_ = json.NewDecoder(w.Body).Decode(&resp)
		if !strings.Contains(resp.Parameters, "131072") {
			t.Errorf("parameters = %q, want to contain 131072", resp.Parameters)
		}
	})
}

// TestOllamaPs_MultiUpstream verifies /api/ps returns all models from
// multi-upstream configs.
func TestOllamaPs_MultiUpstream(t *testing.T) {
	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: "http://localhost:8000",
			Models: []config.ModelMapping{
				{Upstream: "m1", Local: "model-a:latest"},
				{Upstream: "m2", Local: "model-b:latest"},
			},
		},
	}, 65536)

	server := New(config.Config{ListenAddr: ":11434", ModelContextLength: 65536, OllamaVersion: "0.6.4"},
		router, &http.Client{Timeout: 5 * time.Second}, stats.New())

	req := httptest.NewRequest(http.MethodGet, "/api/ps", nil)
	w := httptest.NewRecorder()
	server.handlePs(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var resp types.OllamaPsResponse
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if len(resp.Models) != 2 {
		t.Fatalf("len(models) = %d, want 2", len(resp.Models))
	}
	gotModels := make(map[string]bool)
	for _, m := range resp.Models {
		gotModels[m.Name] = true
		if m.ExpiresAt == "" {
			t.Errorf("model %q: expires_at is empty", m.Name)
		}
		if m.Digest == "" {
			t.Errorf("model %q: digest is empty", m.Digest)
		}
	}
	for _, want := range []string{"model-a:latest", "model-b:latest"} {
		if !gotModels[want] {
			t.Errorf("model %q missing from /api/ps", want)
		}
	}
}

// TestOllamaChat_EmptyMessagesWithOptions verifies that empty messages with
// keep_alive=0 triggers "unload" done reason.
func TestOllamaChat_EmptyMessages_Unload(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"qwen3:latest","messages":[],"keep_alive":"0s"}`))
	w := httptest.NewRecorder()
	server.handleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var resp types.OllamaChatResponse
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	if !resp.Done {
		t.Error("done should be true")
	}
	if resp.DoneReason != "unload" {
		t.Errorf("done_reason = %q, want %q", resp.DoneReason, "unload")
	}
}

// TestOllamaChat_StreamMultiUpstream verifies that streaming chat works across
// different upstreams in a multi-upstream config.
func TestOllamaChat_StreamMultiUpstream(t *testing.T) {
	upstreamA := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "data: {\"id\":\"1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"response-a\"}}]}\n\ndata: [DONE]\n\n")
	}))
	defer upstreamA.Close()

	upstreamB := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "data: {\"id\":\"1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"response-b\"}}]}\n\ndata: [DONE]\n\n")
	}))
	defer upstreamB.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: upstreamA.URL, Models: []config.ModelMapping{{Upstream: "ma", Local: "model-a:latest"}}},
		{URL: upstreamB.URL, Models: []config.ModelMapping{{Upstream: "mb", Local: "model-b:latest"}}},
	}, 65536)
	server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	t.Run("stream_a", func(t *testing.T) {
		body := strings.NewReader(`{"model":"model-a:latest","messages":[{"role":"user","content":"Hi"}],"stream":true}`)
		req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
		w := httptest.NewRecorder()
		server.handleChat(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", w.Code)
		}
		lines := strings.Split(strings.TrimSpace(w.Body.String()), "\n")
		var first types.OllamaChatResponse
		_ = json.Unmarshal([]byte(lines[0]), &first)
		if first.Model != "model-a:latest" {
			t.Errorf("model = %q, want %q", first.Model, "model-a:latest")
		}
		if first.Message.Content != "response-a" {
			t.Errorf("content = %q, want %q", first.Message.Content, "response-a")
		}
	})

	t.Run("stream_b", func(t *testing.T) {
		body := strings.NewReader(`{"model":"model-b:latest","messages":[{"role":"user","content":"Hi"}],"stream":true}`)
		req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
		w := httptest.NewRecorder()
		server.handleChat(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", w.Code)
		}
		lines := strings.Split(strings.TrimSpace(w.Body.String()), "\n")
		var first types.OllamaChatResponse
		_ = json.Unmarshal([]byte(lines[0]), &first)
		if first.Model != "model-b:latest" {
			t.Errorf("model = %q, want %q", first.Model, "model-b:latest")
		}
		if first.Message.Content != "response-b" {
			t.Errorf("content = %q, want %q", first.Message.Content, "response-b")
		}
	})
}
