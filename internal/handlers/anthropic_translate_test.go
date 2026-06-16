package handlers

import (
	"encoding/json"
	"testing"

	"github.com/k0in/openai-ollama-proxy/internal/types"
)

func anthropicFloatPtr(f float64) *float64 { return &f }
func anthropicIntPtr(i int) *int           { return &i }
func anthropicStrPtr(s string) *string     { return &s }

// TestTranslateAnthropicToOpenAI_Parameters verifies all sampling parameters
// and the system prompt are mapped onto the upstream OpenAI request.
func TestTranslateAnthropicToOpenAI_Parameters(t *testing.T) {
	req := types.AnthropicMessageRequest{
		Model:         "claude-4",
		MaxTokens:     321,
		Temperature:   anthropicFloatPtr(0.5),
		TopP:          anthropicFloatPtr(0.9),
		TopK:          anthropicIntPtr(40),
		StopSequences: []string{"STOP", "END"},
		System:        json.RawMessage(`"You are helpful."`),
		Messages: []types.AnthropicMessage{
			{Role: "user", Content: json.RawMessage(`"Hi"`)},
		},
	}

	got, err := translateAnthropicToOpenAI(req, req.MaxTokens)
	if err != nil {
		t.Fatalf("translateAnthropicToOpenAI: %v", err)
	}

	if got.MaxTokens == nil || *got.MaxTokens != 321 {
		t.Errorf("MaxTokens = %v, want 321", got.MaxTokens)
	}
	if got.Temperature == nil || *got.Temperature != 0.5 {
		t.Errorf("Temperature = %v, want 0.5", got.Temperature)
	}
	if got.TopP == nil || *got.TopP != 0.9 {
		t.Errorf("TopP = %v, want 0.9", got.TopP)
	}
	if got.TopK == nil || *got.TopK != 40 {
		t.Errorf("TopK = %v, want 40", got.TopK)
	}
	if len(got.Stop) != 2 || got.Stop[0] != "STOP" || got.Stop[1] != "END" {
		t.Errorf("Stop = %v, want [STOP END]", got.Stop)
	}
	if len(got.Messages) != 2 {
		t.Fatalf("len(messages) = %d, want 2 (system + user)", len(got.Messages))
	}
	if got.Messages[0].Role != "system" {
		t.Errorf("messages[0].role = %q, want system", got.Messages[0].Role)
	}
	var sysText string
	_ = json.Unmarshal(got.Messages[0].Content, &sysText)
	if sysText != "You are helpful." {
		t.Errorf("system text = %q, want %q", sysText, "You are helpful.")
	}
}

// TestTranslateAnthropicMsg_TextAndImage verifies mixed text + image content
// blocks are converted to OpenAI multimodal content parts.
func TestTranslateAnthropicMsg_TextAndImage(t *testing.T) {
	msg := types.AnthropicMessage{
		Role: "user",
		Content: json.RawMessage(`[
			{"type":"text","text":"What is this?"},
			{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}}
		]`),
	}

	got := translateAnthropicMsg(msg)
	if got.Role != "user" {
		t.Errorf("role = %q, want user", got.Role)
	}

	var parts []types.OpenAIContentPart
	if err := json.Unmarshal(got.Content, &parts); err != nil {
		t.Fatalf("content is not a multimodal array: %v (%s)", err, got.Content)
	}
	if len(parts) != 2 {
		t.Fatalf("len(parts) = %d, want 2", len(parts))
	}
	if parts[0].Type != "text" || parts[0].Text != "What is this?" {
		t.Errorf("parts[0] = %+v, want text 'What is this?'", parts[0])
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil {
		t.Fatalf("parts[1] = %+v, want image_url", parts[1])
	}
	if parts[1].ImageURL.URL != "data:image/png;base64,iVBORw0KGgo=" {
		t.Errorf("image url = %q, want data:image/png;base64,iVBORw0KGgo=", parts[1].ImageURL.URL)
	}
}

// TestTranslateAnthropicMsg_ToolUse verifies an assistant tool_use block
// becomes an OpenAI tool_call.
func TestTranslateAnthropicMsg_ToolUse(t *testing.T) {
	msg := types.AnthropicMessage{
		Role: "assistant",
		Content: json.RawMessage(`[
			{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Tokyo"}}
		]`),
	}

	got := translateAnthropicMsg(msg)
	if len(got.ToolCalls) != 1 {
		t.Fatalf("len(tool_calls) = %d, want 1", len(got.ToolCalls))
	}
	tc := got.ToolCalls[0]
	if tc.ID != "toolu_1" {
		t.Errorf("tool_call id = %q, want toolu_1", tc.ID)
	}
	if tc.Type != "function" {
		t.Errorf("tool_call type = %q, want function", tc.Type)
	}
	if tc.Function.Name != "get_weather" {
		t.Errorf("tool_call name = %q, want get_weather", tc.Function.Name)
	}
	if tc.Function.Arguments != `{"city":"Tokyo"}` {
		t.Errorf("tool_call arguments = %q, want {\"city\":\"Tokyo\"}", tc.Function.Arguments)
	}
}

// TestTranslateAnthropicMsg_ToolResult verifies a tool_result block becomes an
// OpenAI tool-role message carrying the matching tool_call_id.
func TestTranslateAnthropicMsg_ToolResult(t *testing.T) {
	t.Run("string_content", func(t *testing.T) {
		msg := types.AnthropicMessage{
			Role: "user",
			Content: json.RawMessage(`[
				{"type":"tool_result","tool_use_id":"toolu_1","content":"sunny, 22C"}
			]`),
		}
		got := translateAnthropicMsg(msg)
		if got.Role != "tool" {
			t.Errorf("role = %q, want tool", got.Role)
		}
		if got.ToolCallID != "toolu_1" {
			t.Errorf("tool_call_id = %q, want toolu_1", got.ToolCallID)
		}
		var text string
		_ = json.Unmarshal(got.Content, &text)
		if text != "sunny, 22C" {
			t.Errorf("content = %q, want 'sunny, 22C'", text)
		}
	})

	t.Run("block_content", func(t *testing.T) {
		msg := types.AnthropicMessage{
			Role: "user",
			Content: json.RawMessage(`[
				{"type":"tool_result","tool_use_id":"toolu_2","content":[{"type":"text","text":"rainy"}]}
			]`),
		}
		got := translateAnthropicMsg(msg)
		if got.ToolCallID != "toolu_2" {
			t.Errorf("tool_call_id = %q, want toolu_2", got.ToolCallID)
		}
		var text string
		_ = json.Unmarshal(got.Content, &text)
		if text != "rainy" {
			t.Errorf("content = %q, want 'rainy'", text)
		}
	})
}

// TestExtractAnthropicSystemText covers string, block-array, and empty system
// prompt shapes.
func TestExtractAnthropicSystemText(t *testing.T) {
	cases := []struct {
		name string
		in   json.RawMessage
		want string
	}{
		{"string", json.RawMessage(`"be brief"`), "be brief"},
		{"blocks", json.RawMessage(`[{"type":"text","text":"line1"},{"type":"text","text":"line2"}]`), "line1\nline2"},
		{"empty", json.RawMessage(``), ""},
		{"null", json.RawMessage(`null`), ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := extractAnthropicSystemText(c.in); got != c.want {
				t.Errorf("extractAnthropicSystemText(%s) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

// TestMapAnthropicStopReason verifies OpenAI finish reasons map to Anthropic
// stop reasons.
func TestMapAnthropicStopReason(t *testing.T) {
	cases := map[string]string{
		"stop":           "end_turn",
		"length":         "max_tokens",
		"tool_calls":     "tool_use",
		"content_filter": "content_filter",
		"unknown":        "end_turn",
	}
	for in, want := range cases {
		reason := in
		if got := mapAnthropicStopReason(&reason); got != want {
			t.Errorf("mapAnthropicStopReason(%q) = %q, want %q", in, got, want)
		}
	}
	if got := mapAnthropicStopReason(nil); got != "end_turn" {
		t.Errorf("mapAnthropicStopReason(nil) = %q, want end_turn", got)
	}
}

// TestConvertOpenAIToAnthropic_ContentBlocks verifies reasoning, text, and
// tool_use are emitted as ordered Anthropic content blocks.
func TestConvertOpenAIToAnthropic_ContentBlocks(t *testing.T) {
	stop := "tool_calls"
	resp := types.OpenAIChatResponse{
		Choices: []types.OpenAIChoice{{
			Message: &types.OpenAIRespMsg{
				Role:             "assistant",
				Content:          anthropicStrPtr("Here you go"),
				ReasoningContent: anthropicStrPtr("thinking..."),
				ToolCalls: []types.OpenAIToolCall{{
					ID:   "toolu_9",
					Type: "function",
					Function: types.OpenAIToolCallFunction{
						Name:      "get_weather",
						Arguments: `{"city":"Paris"}`,
					},
				}},
			},
			FinishReason: &stop,
		}},
		Usage: &types.OpenAIUsage{PromptTokens: 11, CompletionTokens: 7},
	}

	got := convertOpenAIToAnthropic(resp, "claude-4")

	if got.StopReason != "tool_use" {
		t.Errorf("stop_reason = %q, want tool_use", got.StopReason)
	}
	if got.Usage == nil || got.Usage.InputTokens != 11 || got.Usage.OutputTokens != 7 {
		t.Errorf("usage = %+v, want {11 7}", got.Usage)
	}
	if len(got.Content) != 3 {
		t.Fatalf("len(content) = %d, want 3 (thinking, text, tool_use)", len(got.Content))
	}
	if got.Content[0].Type != "thinking" || got.Content[0].Text != "thinking..." {
		t.Errorf("content[0] = %+v, want thinking block", got.Content[0])
	}
	if got.Content[1].Type != "text" || got.Content[1].Text != "Here you go" {
		t.Errorf("content[1] = %+v, want text block", got.Content[1])
	}
	if got.Content[2].Type != "tool_use" || got.Content[2].Name != "get_weather" {
		t.Errorf("content[2] = %+v, want tool_use block", got.Content[2])
	}
}

// TestConvertOpenAIToAnthropic_EmptyContent ensures an empty response still
// produces a valid (non-nil) content array with a single empty text block.
func TestConvertOpenAIToAnthropic_EmptyContent(t *testing.T) {
	stop := "stop"
	resp := types.OpenAIChatResponse{
		Choices: []types.OpenAIChoice{{
			Message:      &types.OpenAIRespMsg{Role: "assistant", Content: anthropicStrPtr("")},
			FinishReason: &stop,
		}},
	}

	got := convertOpenAIToAnthropic(resp, "claude-4")
	if got.Content == nil {
		t.Fatal("content should never be nil")
	}
	if len(got.Content) != 1 || got.Content[0].Type != "text" {
		t.Fatalf("content = %+v, want single empty text block", got.Content)
	}
}
