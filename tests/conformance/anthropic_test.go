package conformance

import (
	"context"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

func newAnthropicClient(h *harness) anthropic.Client {
	return anthropic.NewClient(
		option.WithBaseURL(h.ProxyURL),
		option.WithAPIKey("test-key"),
	)
}

// TestAnthropic_MessagesParams verifies system prompt and sampling parameters
// are translated onto the upstream OpenAI request.
func TestAnthropic_MessagesParams(t *testing.T) {
	h := newHarness(t)
	defer h.Close()

	client := newAnthropicClient(h)

	_, err := client.Messages.New(context.Background(), anthropic.MessageNewParams{
		Model:         localModel,
		MaxTokens:     222,
		Temperature:   anthropic.Float(0.4),
		TopP:          anthropic.Float(0.8),
		TopK:          anthropic.Int(35),
		StopSequences: []string{"STOP"},
		System: []anthropic.TextBlockParam{
			{Text: "You are helpful."},
		},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Hi there")),
		},
	})
	if err != nil {
		t.Fatalf("messages: %v", err)
	}

	got := h.LastChat(t)
	if got.Chat.Model != upstreamModel {
		t.Errorf("upstream model = %q, want %q", got.Chat.Model, upstreamModel)
	}
	if got.Chat.MaxTokens == nil || *got.Chat.MaxTokens != 222 {
		t.Errorf("max_tokens = %v, want 222", got.Chat.MaxTokens)
	}
	if got.Chat.Temperature == nil || *got.Chat.Temperature != 0.4 {
		t.Errorf("temperature = %v, want 0.4", got.Chat.Temperature)
	}
	if got.Chat.TopP == nil || *got.Chat.TopP != 0.8 {
		t.Errorf("top_p = %v, want 0.8", got.Chat.TopP)
	}
	if got.Chat.TopK == nil || *got.Chat.TopK != 35 {
		t.Errorf("top_k = %v, want 35", got.Chat.TopK)
	}
	if len(got.Chat.Stop) != 1 || got.Chat.Stop[0] != "STOP" {
		t.Errorf("stop = %v, want [STOP]", got.Chat.Stop)
	}

	// System prompt should become the first (system-role) message upstream.
	if len(got.Chat.Messages) < 1 || got.Chat.Messages[0].Role != "system" {
		t.Fatalf("messages[0] role = %v, want system message; messages=%+v", got.Chat.Messages, got.Chat.Messages)
	}
	if !strings.Contains(string(got.Chat.Messages[0].Content), "You are helpful.") {
		t.Errorf("system content = %s, want to contain 'You are helpful.'", got.Chat.Messages[0].Content)
	}
}

// TestAnthropic_ToolsTranslated verifies the Anthropic tool schema is rewritten
// into the OpenAI function-tool schema (the critical earlier bug fix).
func TestAnthropic_ToolsTranslated(t *testing.T) {
	h := newHarness(t)
	defer h.Close()

	client := newAnthropicClient(h)

	_, err := client.Messages.New(context.Background(), anthropic.MessageNewParams{
		Model:     localModel,
		MaxTokens: 256,
		Tools: []anthropic.ToolUnionParam{
			anthropic.ToolUnionParamOfTool(anthropic.ToolInputSchemaParam{
				Properties: map[string]any{
					"city": map[string]any{"type": "string"},
				},
				Required: []string{"city"},
			}, "get_weather"),
		},
		ToolChoice: anthropic.ToolChoiceParamOfTool("get_weather"),
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("weather in Tokyo?")),
		},
	})
	if err != nil {
		t.Fatalf("messages: %v", err)
	}

	got := h.LastChat(t)
	// Tools must be OpenAI function schema: {type:"function", function:{name,...}}.
	toolsStr := string(got.Chat.Tools)
	if !strings.Contains(toolsStr, `"type":"function"`) {
		t.Errorf("tools not in OpenAI function schema: %s", toolsStr)
	}
	if !strings.Contains(toolsStr, "get_weather") {
		t.Errorf("tool name not forwarded: %s", toolsStr)
	}
	// tool_choice must map to OpenAI's named function choice.
	if !strings.Contains(string(got.Chat.ToolChoice), "get_weather") {
		t.Errorf("tool_choice not forwarded: %s", got.Chat.ToolChoice)
	}
}

// TestAnthropic_ImageBlock verifies an image content block is translated into a
// multimodal image_url part upstream.
func TestAnthropic_ImageBlock(t *testing.T) {
	h := newHarness(t)
	defer h.Close()

	client := newAnthropicClient(h)

	_, err := client.Messages.New(context.Background(), anthropic.MessageNewParams{
		Model:     localModel,
		MaxTokens: 128,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(
				anthropic.NewTextBlock("describe this"),
				anthropic.NewImageBlockBase64("image/png", "iVBORw0KGgo="),
			),
		},
	})
	if err != nil {
		t.Fatalf("messages: %v", err)
	}

	got := h.LastChat(t)
	var found bool
	for _, m := range got.Chat.Messages {
		if strings.Contains(string(m.Content), "image_url") && strings.Contains(string(m.Content), "iVBORw0KGgo=") {
			found = true
		}
	}
	if !found {
		t.Errorf("image block not forwarded as image_url: %+v", got.Chat.Messages)
	}
}

// TestAnthropic_ToolResultRoundTrip verifies an assistant tool_use followed by
// a user tool_result is forwarded as an OpenAI tool-role message carrying the
// matching tool_call_id.
func TestAnthropic_ToolResultRoundTrip(t *testing.T) {
	h := newHarness(t)
	defer h.Close()

	client := newAnthropicClient(h)

	_, err := client.Messages.New(context.Background(), anthropic.MessageNewParams{
		Model:     localModel,
		MaxTokens: 128,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("weather?")),
			anthropic.NewAssistantMessage(
				anthropic.NewToolUseBlock("toolu_1", map[string]any{"city": "Tokyo"}, "get_weather"),
			),
			anthropic.NewUserMessage(
				anthropic.NewToolResultBlock("toolu_1", "sunny, 22C", false),
			),
		},
	})
	if err != nil {
		t.Fatalf("messages: %v", err)
	}

	got := h.LastChat(t)
	var foundToolMsg bool
	for _, m := range got.Chat.Messages {
		if m.Role == "tool" && m.ToolCallID == "toolu_1" {
			foundToolMsg = true
		}
	}
	if !foundToolMsg {
		t.Errorf("no tool-role message with tool_call_id=toolu_1 forwarded: %+v", got.Chat.Messages)
	}
}

// TestAnthropic_MessagesStream verifies the streaming response is valid enough
// for the SDK's accumulator to rebuild the final message, and that the upstream
// request was marked as a stream.
func TestAnthropic_MessagesStream(t *testing.T) {
	h := newHarness(t)
	defer h.Close()

	client := newAnthropicClient(h)

	stream := client.Messages.NewStreaming(context.Background(), anthropic.MessageNewParams{
		Model:     localModel,
		MaxTokens: 128,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("stream please")),
		},
	})

	var message anthropic.Message
	for stream.Next() {
		event := stream.Current()
		if err := message.Accumulate(event); err != nil {
			t.Fatalf("accumulate: %v", err)
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}

	var text strings.Builder
	for _, block := range message.Content {
		if block.Type == "text" {
			text.WriteString(block.Text)
		}
	}
	if text.String() != "Hello world" {
		t.Errorf("accumulated text = %q, want %q", text.String(), "Hello world")
	}

	got := h.LastChat(t)
	if !got.Chat.Stream {
		t.Error("upstream request should have stream=true")
	}
}
