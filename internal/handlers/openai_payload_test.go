package handlers

import (
	"encoding/json"
	"strings"
	"testing"
)

func Test_rewriteRequestModel(t *testing.T) {
	tests := []struct {
		name       string
		input      map[string]any
		wantModel  string
		wantErr    bool
	}{
		{
			name:      "adds model field",
			input:     map[string]any{"messages": []any{}},
			wantModel: "test-model",
		},
		{
			name:      "overwrites existing model",
			input:     map[string]any{"model": "old-model", "messages": []any{}},
			wantModel: "test-model",
		},
		{
			name:      "preserves other fields",
			input:     map[string]any{"temperature": 0.7, "stream": true},
			wantModel: "test-model",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := newTestServer()
			server.cfg.VLLMModel = "test-model"

			body, _ := json.Marshal(tt.input)
			result, err := server.rewriteRequestModel(body)

			if (err != nil) != tt.wantErr {
				t.Errorf("rewriteRequestModel error: %v", err)
			}

			var parsed map[string]any
			if err := json.Unmarshal(result, &parsed); err != nil {
				t.Fatalf("failed to parse result: %v", err)
			}

			if got := parsed["model"]; got != tt.wantModel {
				t.Errorf("model = %v, want %v", got, tt.wantModel)
			}
		})
	}
}

func Test_rewriteRequestModel_invalidJSON(t *testing.T) {
	server := newTestServer()
	_, err := server.rewriteRequestModel([]byte("not valid json"))
	if err == nil {
		t.Error("expected error for invalid JSON, got nil")
	}
}

func Test_rewriteRequestForChat(t *testing.T) {
	tests := []struct {
		name           string
		input          map[string]any
		wantModel      string
		wantChatKwargs bool
	}{
		{
			name:           "adds model and chat_template_kwargs",
			input:          map[string]any{"messages": []any{}},
			wantModel:      "test-model",
			wantChatKwargs: true,
		},
		{
			name:           "preserves existing chat_template_kwargs",
			input:          map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": true}},
			wantModel:      "test-model",
			wantChatKwargs: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := newTestServer()
			server.cfg.VLLMModel = "test-model"

			body, _ := json.Marshal(tt.input)
			result, changed, err := server.rewriteRequestForChat(body)

			if err != nil {
				t.Fatalf("rewriteRequestForChat error: %v", err)
			}

			if changed {
				t.Error("expected changed to be false")
			}

			var parsed map[string]any
			if err := json.Unmarshal(result, &parsed); err != nil {
				t.Fatalf("failed to parse result: %v", err)
			}

			if got := parsed["model"]; got != tt.wantModel {
				t.Errorf("model = %v, want %v", got, tt.wantModel)
			}

			if tt.wantChatKwargs {
				if _, ok := parsed["chat_template_kwargs"]; !ok {
					t.Error("expected chat_template_kwargs to be present")
				}
			}
		})
	}
}

func Test_rewriteRequestForChat_invalidJSON(t *testing.T) {
	server := newTestServer()
	_, _, err := server.rewriteRequestForChat([]byte("invalid"))
	if err == nil {
		t.Error("expected error for invalid JSON, got nil")
	}
}

func Test_requestDebugSummary(t *testing.T) {
	tests := []struct {
		name    string
		input   []byte
		wantSub string
	}{
		{
			name:    "full payload with all fields",
			input:   mustMarshal(t, map[string]any{"model": "test", "stream": true, "messages": []any{map[string]any{}, map[string]any{}}, "tools": []any{map[string]any{}}, "tool_choice": "auto", "stream_options": map[string]any{}}),
			wantSub: "model=\"test\"",
		},
		{
			name:    "minimal payload",
			input:   mustMarshal(t, map[string]any{"model": "minimal"}),
			wantSub: "messages=0",
		},
		{
			name:    "invalid JSON",
			input:   []byte("not json"),
			wantSub: "invalid-json=",
		},
		{
			name:    "counts messages correctly",
			input:   mustMarshal(t, map[string]any{"messages": []any{map[string]any{}, map[string]any{}, map[string]any{}}}),
			wantSub: "messages=3",
		},
		{
			name:    "counts tools correctly",
			input:   mustMarshal(t, map[string]any{"tools": []any{map[string]any{}, map[string]any{}}}),
			wantSub: "tools=2",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := requestDebugSummary(tt.input)
			if !containsSubstring(t, result, tt.wantSub) {
				t.Errorf("result %q does not contain %q", result, tt.wantSub)
			}
		})
	}
}

func Test_truncateForLog(t *testing.T) {
	tests := []struct {
		name   string
		input  string
		max    int
		wanted string
	}{
		{
			name:   "short string unchanged",
			input:  "short",
			max:    10,
			wanted: "short",
		},
		{
			name:   "exact length unchanged",
			input:  "exactly10",
			max:    9,
			wanted: "exactly10",
		},
		{
			name:   "long string truncated",
			input:  "this is a very long string that exceeds the max length",
			max:    10,
			wanted: "this is a  ...(truncated)",
		},
		{
			name:   "empty string",
			input:  "",
			max:    5,
			wanted: "",
		},
		{
			name:   "max is zero with long string",
			input:  "anything",
			max:    0,
			wanted: " ...(truncated)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := truncateForLog(tt.input, tt.max)
			if result != tt.wanted {
				t.Errorf("truncateForLog(%q, %d) = %q, want %q", tt.input, tt.max, result, tt.wanted)
			}
		})
	}
}

func Test_normalizeOpenAIMessageMap(t *testing.T) {
	tests := []struct {
		name   string
		input  map[string]any
		check  func(t *testing.T, result map[string]any)
	}{
		{
			name:  "uses content as-is",
			input: map[string]any{"content": "hello", "role": "user"},
			check: func(t *testing.T, result map[string]any) {
				if result["content"] != "hello" {
					t.Errorf("content = %v, want hello", result["content"])
				}
			},
		},
		{
			name:  "falls back to reasoning_content when content empty",
			input: map[string]any{"content": "", "reasoning_content": "reasoning here"},
			check: func(t *testing.T, result map[string]any) {
				if result["content"] != "reasoning here" {
					t.Errorf("content = %v, want 'reasoning here'", result["content"])
				}
			},
		},
		{
			name:  "falls back to reasoning when content and reasoning_content empty",
			input: map[string]any{"content": "", "reasoning": "reasoning"},
			check: func(t *testing.T, result map[string]any) {
				if result["content"] != "reasoning" {
					t.Errorf("content = %v, want 'reasoning'", result["content"])
				}
			},
		},
		{
			name:  "removes reasoning_content and reasoning fields",
			input: map[string]any{"content": "hello", "reasoning_content": "rc", "reasoning": "r"},
			check: func(t *testing.T, result map[string]any) {
				if _, ok := result["reasoning_content"]; ok {
					t.Error("reasoning_content should be deleted")
				}
				if _, ok := result["reasoning"]; ok {
					t.Error("reasoning should be deleted")
				}
			},
		},
		{
			name:  "removes empty tool_calls array",
			input: map[string]any{"content": "hello", "tool_calls": []any{}},
			check: func(t *testing.T, result map[string]any) {
				if _, ok := result["tool_calls"]; ok {
					t.Error("empty tool_calls should be deleted")
				}
			},
		},
		{
			name:  "keeps non-empty tool_calls",
			input: map[string]any{"content": "hello", "tool_calls": []any{map[string]any{"id": "1"}}},
			check: func(t *testing.T, result map[string]any) {
				if _, ok := result["tool_calls"]; !ok {
					t.Error("non-empty tool_calls should be kept")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			normalizeOpenAIMessageMap(tt.input)
			tt.check(t, tt.input)
		})
	}
}

func Test_normalizeOpenAIChoiceMap(t *testing.T) {
	tests := []struct {
		name   string
		input  map[string]any
		check  func(t *testing.T, result map[string]any)
	}{
		{
			name: "removes token_ids and stop_reason",
			input: map[string]any{"token_ids": []any{1, 2}, "stop_reason": "eos"},
			check: func(t *testing.T, result map[string]any) {
				if _, ok := result["token_ids"]; ok {
					t.Error("token_ids should be deleted")
				}
				if _, ok := result["stop_reason"]; ok {
					t.Error("stop_reason should be deleted")
				}
			},
		},
		{
			name:  "normalizes message field",
			input: map[string]any{"message": map[string]any{"content": "", "reasoning_content": "rc"}},
			check: func(t *testing.T, result map[string]any) {
				msg := result["message"].(map[string]any)
				if msg["content"] != "rc" {
					t.Errorf("message content not normalized")
				}
				if _, ok := msg["reasoning_content"]; ok {
					t.Error("reasoning_content should be deleted from message")
				}
			},
		},
		{
			name:  "normalizes delta field",
			input: map[string]any{"delta": map[string]any{"content": "", "reasoning": "r"}},
			check: func(t *testing.T, result map[string]any) {
				delta := result["delta"].(map[string]any)
				if delta["content"] != "r" {
					t.Errorf("delta content not normalized")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			normalizeOpenAIChoiceMap(tt.input)
			tt.check(t, tt.input)
		})
	}
}

func Test_normalizeOpenAIJSON(t *testing.T) {
	tests := []struct {
		name   string
		input  map[string]any
		check  func(t *testing.T, result map[string]any)
	}{
		{
			name:  "sets model to config name",
			input: map[string]any{"model": "original"},
			check: func(t *testing.T, result map[string]any) {
				if result["model"] != "configured-model" {
					t.Errorf("model = %v, want configured-model", result["model"])
				}
			},
		},
		{
			name:  "removes prompt_token_ids",
			input: map[string]any{"prompt_token_ids": []any{1, 2, 3}},
			check: func(t *testing.T, result map[string]any) {
				if _, ok := result["prompt_token_ids"]; ok {
					t.Error("prompt_token_ids should be deleted")
				}
			},
		},
		{
			name: "normalizes choices",
			input: map[string]any{
				"choices": []any{
					map[string]any{"token_ids": []any{1}, "message": map[string]any{"content": "", "reasoning_content": "rc"}},
				},
			},
			check: func(t *testing.T, result map[string]any) {
				choices := result["choices"].([]any)
				choice := choices[0].(map[string]any)
				if _, ok := choice["token_ids"]; ok {
					t.Error("token_ids should be deleted from choice")
				}
				msg := choice["message"].(map[string]any)
				if msg["content"] != "rc" {
					t.Error("message content should be normalized")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := newTestServer()
			server.cfg.ModelName = "configured-model"

			body, _ := json.Marshal(tt.input)
			result, err := server.normalizeOpenAIJSON(body)

			if err != nil {
				t.Fatalf("normalizeOpenAIJSON error: %v", err)
			}

			var parsed map[string]any
			if err := json.Unmarshal(result, &parsed); err != nil {
				t.Fatalf("failed to parse result: %v", err)
			}

			tt.check(t, parsed)
		})
	}
}

func Test_normalizeOpenAIJSON_invalidJSON(t *testing.T) {
	server := newTestServer()
	_, err := server.normalizeOpenAIJSON([]byte("not json"))
	if err == nil {
		t.Error("expected error for invalid JSON, got nil")
	}
}

func Test_normalizeOpenAIStreamLine(t *testing.T) {
	tests := []struct {
		name   string
		line   string
		check  func(t *testing.T, result string)
	}{
		{
			name: "non-data line unchanged",
			line: "some other prefix",
			check: func(t *testing.T, result string) {
				if result != "some other prefix" {
					t.Errorf("result = %q, want unchanged", result)
				}
			},
		},
		{
			name: "data: [DONE] unchanged",
			line: "data: [DONE]",
			check: func(t *testing.T, result string) {
				if result != "data: [DONE]" {
					t.Errorf("result = %q, want 'data: [DONE]'", result)
				}
			},
		},
		{
			name:  "data line with valid JSON gets normalized",
			line:  "data: {\"content\":\"hello\",\"reasoning_content\":\"rc\"}",
			check: func(t *testing.T, result string) {
				if !containsSubstring(t, result, "data: ") {
					t.Error("result should start with 'data: '")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := newTestServer()
			server.cfg.ModelName = "test-model"

			result := server.normalizeOpenAIStreamLine(tt.line)
			tt.check(t, result)
		})
	}
}

// Helper functions

func mustMarshal(t *testing.T, v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	return b
}

func containsSubstring(t *testing.T, s, substr string) bool {
	t.Helper()
	return strings.Contains(s, substr)
}
