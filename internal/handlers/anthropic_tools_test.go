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

// TestConvertAnthropicTools verifies the Anthropic tool schema is rewritten
// into the OpenAI function-tool schema the upstream expects.
func TestConvertAnthropicTools(t *testing.T) {
	in := json.RawMessage(`[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}]`)

	out, err := convertAnthropicTools(in)
	if err != nil {
		t.Fatalf("convertAnthropicTools: %v", err)
	}

	var tools []struct {
		Type     string `json:"type"`
		Function struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			Parameters  json.RawMessage `json:"parameters"`
		} `json:"function"`
	}
	if err := json.Unmarshal(out, &tools); err != nil {
		t.Fatalf("unmarshal converted tools: %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("len(tools) = %d, want 1", len(tools))
	}
	if tools[0].Type != "function" {
		t.Errorf("type = %q, want function", tools[0].Type)
	}
	if tools[0].Function.Name != "get_weather" {
		t.Errorf("name = %q, want get_weather", tools[0].Function.Name)
	}
	if tools[0].Function.Description != "Get weather" {
		t.Errorf("description = %q, want Get weather", tools[0].Function.Description)
	}
	if !strings.Contains(string(tools[0].Function.Parameters), `"city"`) {
		t.Errorf("parameters missing input_schema content: %s", tools[0].Function.Parameters)
	}
}

func TestConvertAnthropicTools_Empty(t *testing.T) {
	for _, in := range []json.RawMessage{nil, json.RawMessage(`null`), json.RawMessage(`[]`)} {
		out, err := convertAnthropicTools(in)
		if err != nil {
			t.Fatalf("convertAnthropicTools(%s): %v", in, err)
		}
		if out != nil {
			t.Errorf("convertAnthropicTools(%s) = %s, want nil", in, out)
		}
	}
}

func TestConvertAnthropicToolChoice(t *testing.T) {
	cases := map[string]string{
		`{"type":"auto"}`:                      `"auto"`,
		`{"type":"any"}`:                       `"required"`,
		`{"type":"none"}`:                      `"none"`,
		`{"type":"tool","name":"get_weather"}`: `{"function":{"name":"get_weather"},"type":"function"}`,
	}
	for in, want := range cases {
		got := convertAnthropicToolChoice(json.RawMessage(in))
		if string(got) != want {
			t.Errorf("convertAnthropicToolChoice(%s) = %s, want %s", in, got, want)
		}
	}

	if got := convertAnthropicToolChoice(nil); got != nil {
		t.Errorf("convertAnthropicToolChoice(nil) = %s, want nil", got)
	}
}

// TestAnthropicMessages_ToolSchemaForwarded ensures the upstream receives a
// valid OpenAI tool definition (not the raw Anthropic schema).
func TestAnthropicMessages_ToolSchemaForwarded(t *testing.T) {
	var gotTools json.RawMessage
	var gotToolChoice json.RawMessage
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req types.OpenAIChatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		gotTools = req.Tools
		gotToolChoice = req.ToolChoice

		content := "ok"
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
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "claude-4"}}},
	}, 65536)
	server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	body := `{
		"model":"claude-4","max_tokens":100,
		"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object"}}],
		"tool_choice":{"type":"tool","name":"get_weather"},
		"messages":[{"role":"user","content":"Weather?"}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleAnthropicMessages(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}

	var tools []struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(gotTools, &tools); err != nil {
		t.Fatalf("upstream tools not OpenAI schema: %v (%s)", err, gotTools)
	}
	if len(tools) != 1 || tools[0].Type != "function" || tools[0].Function.Name != "get_weather" {
		t.Fatalf("unexpected upstream tools: %s", gotTools)
	}
	if !strings.Contains(string(gotToolChoice), "get_weather") {
		t.Errorf("tool_choice not forwarded: %s", gotToolChoice)
	}
}
