package conformance

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
)

func newOpenAIClient(h *harness) openai.Client {
	return openai.NewClient(
		option.WithBaseURL(h.ProxyURL+"/v1"),
		option.WithAPIKey("test-key"),
	)
}

// TestOpenAI_ChatNonStream verifies sampling params, stop, and the model
// rewrite are forwarded upstream.
func TestOpenAI_ChatNonStream(t *testing.T) {
	h := newHarness(t)
	defer h.Close()

	client := newOpenAIClient(h)

	_, err := client.Chat.Completions.New(context.Background(), openai.ChatCompletionNewParams{
		Model: localModel,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Say hi"),
		},
		Temperature: openai.Float(0.55),
		TopP:        openai.Float(0.9),
		MaxTokens:   openai.Int(123),
		Stop: openai.ChatCompletionNewParamsStopUnion{
			OfStringArray: []string{"END"},
		},
	})
	if err != nil {
		t.Fatalf("chat completion: %v", err)
	}

	got := h.LastChat(t)
	if got.Chat.Model != upstreamModel {
		t.Errorf("upstream model = %q, want %q", got.Chat.Model, upstreamModel)
	}
	if got.Chat.Temperature == nil || *got.Chat.Temperature != 0.55 {
		t.Errorf("temperature = %v, want 0.55", got.Chat.Temperature)
	}
	if got.Chat.TopP == nil || *got.Chat.TopP != 0.9 {
		t.Errorf("top_p = %v, want 0.9", got.Chat.TopP)
	}
	if got.Chat.MaxTokens == nil || *got.Chat.MaxTokens != 123 {
		t.Errorf("max_tokens = %v, want 123", got.Chat.MaxTokens)
	}
	if len(got.Chat.Stop) != 1 || got.Chat.Stop[0] != "END" {
		t.Errorf("stop = %v, want [END]", got.Chat.Stop)
	}
}

// TestOpenAI_ChatTools verifies a function tool definition reaches the upstream
// intact.
func TestOpenAI_ChatTools(t *testing.T) {
	h := newHarness(t)
	defer h.Close()

	client := newOpenAIClient(h)

	_, err := client.Chat.Completions.New(context.Background(), openai.ChatCompletionNewParams{
		Model:    localModel,
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("weather?")},
		Tools: []openai.ChatCompletionToolUnionParam{
			openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
				Name:        "get_weather",
				Description: openai.String("Get the weather"),
				Parameters: shared.FunctionParameters{
					"type": "object",
					"properties": map[string]any{
						"city": map[string]any{"type": "string"},
					},
				},
			}),
		},
	})
	if err != nil {
		t.Fatalf("chat completion: %v", err)
	}

	got := h.LastChat(t)
	if len(got.Chat.Tools) == 0 || !strings.Contains(string(got.Chat.Tools), "get_weather") {
		t.Errorf("tools not forwarded: %s", got.Chat.Tools)
	}
}

// TestOpenAI_ChatVision verifies an image_url content part is forwarded as a
// multimodal message.
func TestOpenAI_ChatVision(t *testing.T) {
	h := newHarness(t)
	defer h.Close()

	client := newOpenAIClient(h)

	_, err := client.Chat.Completions.New(context.Background(), openai.ChatCompletionNewParams{
		Model: localModel,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
				openai.TextContentPart("What is in this image?"),
				openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
					URL: "data:image/png;base64,iVBORw0KGgo=",
				}),
			}),
		},
	})
	if err != nil {
		t.Fatalf("chat completion: %v", err)
	}

	got := h.LastChat(t)
	if len(got.Chat.Messages) != 1 {
		t.Fatalf("len(messages) = %d, want 1", len(got.Chat.Messages))
	}
	if !strings.Contains(string(got.Chat.Messages[0].Content), "image_url") {
		t.Errorf("image_url not forwarded: %s", got.Chat.Messages[0].Content)
	}
	if !strings.Contains(string(got.Chat.Messages[0].Content), "iVBORw0KGgo=") {
		t.Errorf("image data not forwarded: %s", got.Chat.Messages[0].Content)
	}
}

// TestOpenAI_ChatStream verifies a streaming request is marked as a stream
// upstream and the SDK reassembles the content.
func TestOpenAI_ChatStream(t *testing.T) {
	h := newHarness(t)
	defer h.Close()

	client := newOpenAIClient(h)

	stream := client.Chat.Completions.NewStreaming(context.Background(), openai.ChatCompletionNewParams{
		Model:    localModel,
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("Stream please")},
	})

	var content strings.Builder
	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) > 0 {
			content.WriteString(chunk.Choices[0].Delta.Content)
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}

	if content.String() != "Hello world" {
		t.Errorf("streamed content = %q, want %q", content.String(), "Hello world")
	}

	got := h.LastChat(t)
	if !got.Chat.Stream {
		t.Error("upstream request should have stream=true")
	}
}

// TestOpenAI_Embeddings verifies the embeddings request reaches the upstream
// with the rewritten model.
func TestOpenAI_Embeddings(t *testing.T) {
	h := newHarness(t)
	defer h.Close()

	client := newOpenAIClient(h)

	_, err := client.Embeddings.New(context.Background(), openai.EmbeddingNewParams{
		Model: localModel,
		Input: openai.EmbeddingNewParamsInputUnion{
			OfString: openai.String("hello world"),
		},
	})
	if err != nil {
		t.Fatalf("embeddings: %v", err)
	}

	got := h.LastEmbed(t)
	if got.Embed["model"] != upstreamModel {
		t.Errorf("upstream embed model = %v, want %q", got.Embed["model"], upstreamModel)
	}
}

// TestOpenAI_ModelsList verifies the proxy advertises the local model alias.
func TestOpenAI_ModelsList(t *testing.T) {
	h := newHarness(t)
	defer h.Close()

	client := newOpenAIClient(h)

	page, err := client.Models.List(context.Background())
	if err != nil {
		t.Fatalf("models list: %v", err)
	}

	var ids []string
	for _, m := range page.Data {
		ids = append(ids, m.ID)
	}
	found := false
	for _, id := range ids {
		if id == localModel {
			found = true
		}
	}
	if !found {
		t.Errorf("model list = %v, want to contain %q", ids, localModel)
	}
}

// ensure encoding/json stays referenced even if assertions change.
var _ = json.Marshal
