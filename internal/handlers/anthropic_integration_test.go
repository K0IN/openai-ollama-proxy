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

// TestAnthropicMessages_NonStream verifies that an Anthropic /v1/messages
// request is translated to OpenAI, routed to the correct upstream, and the
// response is translated back to the Anthropic format.
func TestAnthropicMessages_NonStream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer anthropic-key" {
			t.Errorf("Authorization = %q", r.Header.Get("Authorization"))
		}

		var req types.OpenAIChatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Model != "claude-upstream" {
			t.Errorf("upstream model = %q, want %q", req.Model, "claude-upstream")
		}
		if len(req.Messages) != 1 {
			t.Errorf("len(messages) = %d, want 1", len(req.Messages))
		}

		content := "Hello from Claude!"
		stop := "stop"
		_ = json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			ID:      "chatcmpl-123",
			Object:  "chat.completion",
			Model:   "claude-upstream",
			Created: 1700000000,
			Choices: []types.OpenAIChoice{{
				Index:        0,
				Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
				FinishReason: &stop,
			}},
			Usage: &types.OpenAIUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		})
	}))
	defer upstream.Close()

	router, err := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL:    upstream.URL,
			APIKey: "anthropic-key",
			Models: []config.ModelMapping{
				{Upstream: "claude-upstream", Local: "claude-sonnet-4-20250514", ContextLength: 200000},
			},
		},
	}, 200000)
	if err != nil {
		t.Fatalf("BuildRoutingTable: %v", err)
	}

	cfg := config.Config{
		ListenAddr:            ":11434",
		ModelContextLength:    200000,
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	server := NewWithClients(cfg, router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, nil)

	body := `{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 1024,
		"messages": [
			{"role": "user", "content": "Hello, Claude!"}
		]
	}`

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleAnthropicMessages(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}

	var resp types.AnthropicMessageResponse
	_ = json.Unmarshal(w.Body.Bytes(), &resp)

	if resp.ID == "" {
		t.Error("response ID should not be empty")
	}
	if resp.Type != "message" {
		t.Errorf("type = %q, want %q", resp.Type, "message")
	}
	if resp.Role != "assistant" {
		t.Errorf("role = %q, want %q", resp.Role, "assistant")
	}
	if resp.Model != "claude-sonnet-4-20250514" {
		t.Errorf("model = %q, want %q", resp.Model, "claude-sonnet-4-20250514")
	}
	if len(resp.Content) == 0 {
		t.Fatal("content should not be empty")
	}
	if resp.Content[0].Type != "text" {
		t.Errorf("content[0].type = %q, want %q", resp.Content[0].Type, "text")
	}
	if resp.Content[0].Text != "Hello from Claude!" {
		t.Errorf("content[0].text = %q, want %q", resp.Content[0].Text, "Hello from Claude!")
	}
	if resp.Usage == nil {
		t.Fatal("usage should not be nil")
	}
	if resp.Usage.InputTokens != 10 {
		t.Errorf("input_tokens = %d, want 10", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 5 {
		t.Errorf("output_tokens = %d, want 5", resp.Usage.OutputTokens)
	}
}

// TestAnthropicMessages_Stream verifies streaming Anthropic messages are
// properly translated from OpenAI SSE to Anthropic SSE events.
func TestAnthropicMessages_Stream(t *testing.T) {
	sseData := strings.Join([]string{
		`data: {"id":"1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"delta":{"content":" world"}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"finish_reason":"stop","delta":{}}]}`,
		``,
		`data: {"id":"1","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
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

	router, err := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: upstream.URL,
			Models: []config.ModelMapping{
				{Upstream: "claude-upstream", Local: "claude-sonnet-4-20250514"},
			},
		},
	}, 65536)
	if err != nil {
		t.Fatalf("BuildRoutingTable: %v", err)
	}

	cfg := config.Config{
		ListenAddr:            ":11434",
		ModelContextLength:    65536,
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	server := NewWithClients(cfg, router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, nil)

	body := `{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 1024,
		"stream": true,
		"messages": [
			{"role": "user", "content": "Hi"}
		]
	}`

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleAnthropicMessages(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}

	contentType := w.Result().Header.Get("Content-Type")
	if !strings.HasPrefix(contentType, "text/event-stream") {
		t.Fatalf("Content-Type = %q, want text/event-stream", contentType)
	}

	// Parse SSE events from the response
	lines := strings.Split(w.Body.String(), "\n")
	var events []struct {
		Event string
		Data  string
	}
	for i := 0; i < len(lines); i++ {
		line := lines[i]
		if strings.HasPrefix(line, "event: ") {
			e := strings.TrimPrefix(line, "event: ")
			if i+1 < len(lines) && strings.HasPrefix(lines[i+1], "data: ") {
				d := strings.TrimPrefix(lines[i+1], "data: ")
				events = append(events, struct {
					Event string
					Data  string
				}{Event: e, Data: d})
				i++ // skip the data line
			}
		}
	}

	if len(events) < 4 {
		t.Fatalf("expected at least 4 SSE events, got %d", len(events))
	}

	// Event 0: message_start
	if events[0].Event != "message_start" {
		t.Errorf("event[0] = %q, want %q", events[0].Event, "message_start")
	}
	var msgStart types.AnthropicMessageStartEvent
	if err := json.Unmarshal([]byte(events[0].Data), &msgStart); err != nil {
		t.Fatalf("unmarshal message_start: %v", err)
	}
	if msgStart.Message.Model != "claude-sonnet-4-20250514" {
		t.Errorf("message.model = %q, want %q", msgStart.Message.Model, "claude-sonnet-4-20250514")
	}

	// Event 1: content_block_start
	if events[1].Event != "content_block_start" {
		t.Errorf("event[1] = %q, want %q", events[1].Event, "content_block_start")
	}
	var cbStart types.AnthropicContentBlockStartEvent
	if err := json.Unmarshal([]byte(events[1].Data), &cbStart); err != nil {
		t.Fatalf("unmarshal content_block_start: %v", err)
	}
	if cbStart.ContentBlock.Type != "text" {
		t.Errorf("content_block.type = %q, want %q", cbStart.ContentBlock.Type, "text")
	}

	// Event 2: content_block_delta with first text
	if events[2].Event != "content_block_delta" {
		t.Errorf("event[2] = %q, want %q", events[2].Event, "content_block_delta")
	}
	var delta types.AnthropicContentBlockDeltaEvent
	if err := json.Unmarshal([]byte(events[2].Data), &delta); err != nil {
		t.Fatalf("unmarshal content_block_delta: %v", err)
	}
	if delta.Delta.Type != "text_delta" {
		t.Errorf("delta.type = %q, want %q", delta.Delta.Type, "text_delta")
	}
	if delta.Delta.Text != "Hello" {
		t.Errorf("delta.text = %q, want %q", delta.Delta.Text, "Hello")
	}

	// Should end with message_stop
	lastEvent := events[len(events)-1]
	if lastEvent.Event != "message_stop" {
		t.Errorf("last event = %q, want %q", lastEvent.Event, "message_stop")
	}
}

// TestAnthropicMessages_WithSystem tests Anthropic requests with system
// message (both string and block formats).
func TestAnthropicMessages_WithSystem(t *testing.T) {
	t.Run("system_string", func(t *testing.T) {
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req types.OpenAIChatRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			if len(req.Messages) != 2 {
				t.Fatalf("len(messages) = %d, want 2", len(req.Messages))
			}
			if req.Messages[0].Role != "system" {
				t.Errorf("message[0].role = %q, want %q", req.Messages[0].Role, "system")
			}
			var sysText string
			_ = json.Unmarshal(req.Messages[0].Content, &sysText)
			if sysText != "You are a helpful assistant." {
				t.Errorf("system = %q, want %q", sysText, "You are a helpful assistant.")
			}

			content := "Got it!"
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
			{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "claude-3"}}},
		}, 65536)
		server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
			router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

		body := `{"model":"claude-3","max_tokens":100,"system":"You are a helpful assistant.","messages":[{"role":"user","content":"Hi"}]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
		w := httptest.NewRecorder()
		server.handleAnthropicMessages(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
		}
	})

	t.Run("system_block", func(t *testing.T) {
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var req types.OpenAIChatRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			if len(req.Messages) != 2 {
				t.Fatalf("len(messages) = %d, want 2", len(req.Messages))
			}
			var sysText string
			_ = json.Unmarshal(req.Messages[0].Content, &sysText)
			if sysText != "Be concise." {
				t.Errorf("system = %q, want %q", sysText, "Be concise.")
			}
			content := "OK!"
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
			{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "claude-3"}}},
		}, 65536)
		server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
			router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

		body := `{"model":"claude-3","max_tokens":100,"system":[{"type":"text","text":"Be concise."}],"messages":[{"role":"user","content":"Hi"}]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
		w := httptest.NewRecorder()
		server.handleAnthropicMessages(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
		}
	})
}

// TestAnthropicMessages_WithTools verifies Anthropic tool definitions and
// tool_use responses are translated correctly.
func TestAnthropicMessages_WithTools(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req types.OpenAIChatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Tools == nil {
			t.Error("tools should not be nil")
		}

		content := ""
		toolID := "toolu_123"
		stop := "tool_calls"
		_ = json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			Choices: []types.OpenAIChoice{{
				Index: 0,
				Message: &types.OpenAIRespMsg{
					Role:    "assistant",
					Content: &content,
					ToolCalls: []types.OpenAIToolCall{{
						ID:   toolID,
						Type: "function",
						Function: types.OpenAIToolCallFunction{
							Name:      "get_weather",
							Arguments: `{"city":"Tokyo"}`,
						},
					}},
				},
				FinishReason: &stop,
			}},
			Usage: &types.OpenAIUsage{PromptTokens: 20, CompletionTokens: 10, TotalTokens: 30},
		})
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "claude-4"}}},
	}, 65536)
	server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	body := `{
		"model": "claude-4",
		"max_tokens": 500,
		"tools": [{"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object"}}],
		"messages": [{"role": "user", "content": "Weather in Tokyo?"}]
	}`

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleAnthropicMessages(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}

	var resp types.AnthropicMessageResponse
	_ = json.Unmarshal(w.Body.Bytes(), &resp)

	// When content is empty and tool calls exist, only the tool_use block is emitted
	if len(resp.Content) != 1 {
		t.Fatalf("len(content) = %d, want 1 (tool_use only)", len(resp.Content))
	}
	if resp.Content[0].Type != "tool_use" {
		t.Errorf("content[0].type = %q, want %q", resp.Content[0].Type, "tool_use")
	}
	if resp.Content[0].Name != "get_weather" {
		t.Errorf("tool_use.name = %q, want %q", resp.Content[0].Name, "get_weather")
	}
	if resp.StopReason != "tool_use" {
		t.Errorf("stop_reason = %q, want %q", resp.StopReason, "tool_use")
	}
}

// TestAnthropicMessages_ErrorHandling verifies that upstream errors and
// bad requests produce proper Anthropic error responses.
func TestAnthropicMessages_ErrorHandling(t *testing.T) {
	t.Run("method_not_allowed", func(t *testing.T) {
		server := newTestServer()
		req := httptest.NewRequest(http.MethodGet, "/v1/messages", nil)
		w := httptest.NewRecorder()
		server.handleAnthropicMessages(w, req)

		// The handler returns plain 405 for wrong methods
		if w.Code != http.StatusMethodNotAllowed {
			t.Fatalf("status = %d, want 405", w.Code)
		}
	})

	t.Run("invalid_json", func(t *testing.T) {
		router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
			{URL: "http://localhost:8000", Models: []config.ModelMapping{{Upstream: "m", Local: "claude-3"}}},
		}, 65536)
		server := New(config.Config{ListenAddr: ":11434"}, router, &http.Client{Timeout: 5 * time.Second}, stats.New())

		req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{invalid json}`))
		w := httptest.NewRecorder()
		server.handleAnthropicMessages(w, req)

		if w.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400", w.Code)
		}
		var errResp types.AnthropicErrorResponse
		_ = json.NewDecoder(w.Body).Decode(&errResp)
		if errResp.Error.Type != "invalid_request_error" {
			t.Errorf("error.type = %q, want %q", errResp.Error.Type, "invalid_request_error")
		}
	})

	t.Run("upstream_error", func(t *testing.T) {
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(map[string]string{"error": "internal error"})
		}))
		defer upstream.Close()

		router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
			{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "claude-3"}}},
		}, 65536)
		server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
			router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, nil)

		body := `{"model":"claude-3","max_tokens":100,"messages":[{"role":"user","content":"Hi"}]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
		w := httptest.NewRecorder()
		server.handleAnthropicMessages(w, req)

		if w.Code != http.StatusBadGateway {
			t.Fatalf("status = %d, want 502 (body=%s)", w.Code, w.Body.String())
		}
		var errResp types.AnthropicErrorResponse
		_ = json.NewDecoder(w.Body).Decode(&errResp)
		if errResp.Error.Type != "upstream_error" {
			t.Errorf("error.type = %q, want %q", errResp.Error.Type, "upstream_error")
		}
	})
}

// TestAnthropicMessages_MultiUpstreamRouting verifies Anthropic requests
// are routed to the correct upstream based on the model name.
func TestAnthropicMessages_MultiUpstreamRouting(t *testing.T) {
	upstreamA := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req types.OpenAIChatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Model != "sonnet-upstream" {
			t.Errorf("upstream A got model=%q, want %q", req.Model, "sonnet-upstream")
		}
		content := "From Sonnet"
		stop := "stop"
		_ = json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			Choices: []types.OpenAIChoice{{
				Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
				FinishReason: &stop,
			}},
		})
	}))
	defer upstreamA.Close()

	upstreamB := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req types.OpenAIChatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Model != "haiku-upstream" {
			t.Errorf("upstream B got model=%q, want %q", req.Model, "haiku-upstream")
		}
		content := "From Haiku"
		stop := "stop"
		_ = json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			Choices: []types.OpenAIChoice{{
				Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
				FinishReason: &stop,
			}},
		})
	}))
	defer upstreamB.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: upstreamA.URL, APIKey: "key-a",
			Models: []config.ModelMapping{{Upstream: "sonnet-upstream", Local: "claude-sonnet-4"}},
		},
		{
			URL: upstreamB.URL, APIKey: "key-b",
			Models: []config.ModelMapping{{Upstream: "haiku-upstream", Local: "claude-haiku-3"}},
		},
	}, 65536)

	cfg := config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond}
	server := NewWithClients(cfg, router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	t.Run("route_to_sonnet", func(t *testing.T) {
		body := `{"model":"claude-sonnet-4","max_tokens":100,"messages":[{"role":"user","content":"Hi"}]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
		w := httptest.NewRecorder()
		server.handleAnthropicMessages(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", w.Code)
		}
		var resp types.AnthropicMessageResponse
		_ = json.Unmarshal(w.Body.Bytes(), &resp)
		if len(resp.Content) == 0 || resp.Content[0].Text != "From Sonnet" {
			t.Errorf("got content %+v, want 'From Sonnet'", resp.Content)
		}
	})

	t.Run("route_to_haiku", func(t *testing.T) {
		body := `{"model":"claude-haiku-3","max_tokens":100,"messages":[{"role":"user","content":"Hi"}]}`
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
		w := httptest.NewRecorder()
		server.handleAnthropicMessages(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", w.Code)
		}
		var resp types.AnthropicMessageResponse
		_ = json.Unmarshal(w.Body.Bytes(), &resp)
		if len(resp.Content) == 0 || resp.Content[0].Text != "From Haiku" {
			t.Errorf("got content %+v, want 'From Haiku'", resp.Content)
		}
	})
}
