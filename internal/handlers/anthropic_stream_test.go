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

type sseEvent struct {
	Event string
	Data  string
}

func parseSSEEvents(raw string) []sseEvent {
	lines := strings.Split(raw, "\n")
	var events []sseEvent
	for i := 0; i < len(lines); i++ {
		if strings.HasPrefix(lines[i], "event: ") {
			e := strings.TrimPrefix(lines[i], "event: ")
			if i+1 < len(lines) && strings.HasPrefix(lines[i+1], "data: ") {
				events = append(events, sseEvent{Event: e, Data: strings.TrimPrefix(lines[i+1], "data: ")})
				i++
			}
		}
	}
	return events
}

func newAnthropicStreamServer(t *testing.T, sseData string) *Server {
	t.Helper()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(sseData))
	}))
	t.Cleanup(upstream.Close)

	router, err := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "claude-4"}}},
	}, 65536)
	if err != nil {
		t.Fatalf("BuildRoutingTable: %v", err)
	}
	return NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())
}

// TestAnthropicStream_ToolCallAccumulation verifies that tool-call argument
// fragments streamed across multiple OpenAI deltas are forwarded as
// input_json_delta events that reassemble into the full arguments.
func TestAnthropicStream_ToolCallAccumulation(t *testing.T) {
	sseData := strings.Join([]string{
		`data: {"id":"1","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Tokyo\"}"}}]}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"finish_reason":"tool_calls","delta":{}}]}`,
		``,
		`data: {"id":"1","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	server := newAnthropicStreamServer(t, sseData)

	body := `{"model":"claude-4","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"Weather in Tokyo?"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleAnthropicMessages(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}

	events := parseSSEEvents(w.Body.String())

	var toolStart *types.AnthropicContentBlockStartEvent
	var partialJSON strings.Builder
	stopReason := ""
	for _, e := range events {
		switch e.Event {
		case "content_block_start":
			var cb types.AnthropicContentBlockStartEvent
			if err := json.Unmarshal([]byte(e.Data), &cb); err != nil {
				t.Fatalf("unmarshal content_block_start: %v", err)
			}
			if cb.ContentBlock.Type == "tool_use" {
				toolStart = &cb
			}
		case "content_block_delta":
			var probe struct {
				Delta struct {
					Type        string `json:"type"`
					PartialJSON string `json:"partial_json"`
				} `json:"delta"`
			}
			if err := json.Unmarshal([]byte(e.Data), &probe); err != nil {
				t.Fatalf("unmarshal content_block_delta: %v", err)
			}
			if probe.Delta.Type == "input_json_delta" {
				partialJSON.WriteString(probe.Delta.PartialJSON)
			}
		case "message_delta":
			var md types.AnthropicMessageDeltaEvent
			if err := json.Unmarshal([]byte(e.Data), &md); err != nil {
				t.Fatalf("unmarshal message_delta: %v", err)
			}
			stopReason = md.Delta.StopReason
		}
	}

	if toolStart == nil {
		t.Fatal("no tool_use content_block_start emitted; events=", events)
		return
	}
	if toolStart.ContentBlock.Name != "get_weather" {
		t.Errorf("tool name = %q, want get_weather", toolStart.ContentBlock.Name)
	}
	if toolStart.ContentBlock.ID != "call_1" {
		t.Errorf("tool id = %q, want call_1", toolStart.ContentBlock.ID)
	}
	if got := partialJSON.String(); got != `{"city":"Tokyo"}` {
		t.Errorf("accumulated arguments = %q, want %q", got, `{"city":"Tokyo"}`)
	}
	if stopReason != "tool_use" {
		t.Errorf("stop_reason = %q, want tool_use", stopReason)
	}

	// Stream must end with message_stop.
	if len(events) == 0 || events[len(events)-1].Event != "message_stop" {
		t.Errorf("last event = %v, want message_stop", events[len(events)-1])
	}
}

// TestAnthropicStream_Reasoning verifies reasoning deltas are surfaced as a
// thinking content block in the Anthropic stream.
func TestAnthropicStream_Reasoning(t *testing.T) {
	sseData := strings.Join([]string{
		`data: {"id":"1","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Let me think"}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"delta":{"content":"Answer"}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"finish_reason":"stop","delta":{}}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	server := newAnthropicStreamServer(t, sseData)

	body := `{"model":"claude-4","max_tokens":100,"stream":true,"messages":[{"role":"user","content":"Hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleAnthropicMessages(w, req)

	events := parseSSEEvents(w.Body.String())

	var thinking, text strings.Builder
	for _, e := range events {
		if e.Event != "content_block_delta" {
			continue
		}
		var probe struct {
			Delta struct {
				Type     string `json:"type"`
				Text     string `json:"text"`
				Thinking string `json:"thinking"`
			} `json:"delta"`
		}
		if err := json.Unmarshal([]byte(e.Data), &probe); err != nil {
			t.Fatalf("unmarshal content_block_delta: %v", err)
		}
		switch probe.Delta.Type {
		case "thinking_delta":
			thinking.WriteString(probe.Delta.Thinking)
		case "text_delta":
			text.WriteString(probe.Delta.Text)
		}
	}

	if thinking.String() != "Let me think" {
		t.Errorf("thinking = %q, want %q", thinking.String(), "Let me think")
	}
	if text.String() != "Answer" {
		t.Errorf("text = %q, want %q", text.String(), "Answer")
	}
}
