package conformance

import (
	"context"
	"encoding/base64"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func newOllamaClient(t *testing.T, h *harness) *api.Client {
	t.Helper()
	u, err := url.Parse(h.ProxyURL)
	if err != nil {
		t.Fatalf("parse proxy URL: %v", err)
	}
	return api.NewClient(u, &http.Client{Timeout: 30 * time.Second})
}

// TestOllama_ChatOptions verifies that Ollama options are translated to the
// corresponding OpenAI sampling parameters upstream.
func TestOllama_ChatOptions(t *testing.T) {
	h := newHarness(t)
	defer h.Close()

	client := newOllamaClient(t, h)

	stream := false
	err := client.Chat(context.Background(), &api.ChatRequest{
		Model:    localModel,
		Messages: []api.Message{{Role: "user", Content: "Hello"}},
		Stream:   &stream,
		Options: map[string]any{
			"temperature":       0.33,
			"top_p":             0.85,
			"top_k":             35,
			"seed":              42,
			"num_predict":       64,
			"stop":              []string{"END"},
			"frequency_penalty": 0.2,
			"presence_penalty":  0.3,
			"repeat_penalty":    1.2,
			"min_p":             0.05,
		},
	}, func(resp api.ChatResponse) error { return nil })
	if err != nil {
		t.Fatalf("chat: %v", err)
	}

	got := h.LastChat(t)
	if got.Chat.Model != upstreamModel {
		t.Errorf("upstream model = %q, want %q", got.Chat.Model, upstreamModel)
	}
	if got.Chat.Temperature == nil || *got.Chat.Temperature != 0.33 {
		t.Errorf("temperature = %v, want 0.33", got.Chat.Temperature)
	}
	if got.Chat.TopP == nil || *got.Chat.TopP != 0.85 {
		t.Errorf("top_p = %v, want 0.85", got.Chat.TopP)
	}
	if got.Chat.TopK == nil || *got.Chat.TopK != 35 {
		t.Errorf("top_k = %v, want 35", got.Chat.TopK)
	}
	if got.Chat.Seed == nil || *got.Chat.Seed != 42 {
		t.Errorf("seed = %v, want 42", got.Chat.Seed)
	}
	if got.Chat.MaxTokens == nil || *got.Chat.MaxTokens != 64 {
		t.Errorf("max_tokens = %v, want 64 (from num_predict)", got.Chat.MaxTokens)
	}
	var foundStop bool
	for _, s := range got.Chat.Stop {
		if s == "END" {
			foundStop = true
		}
	}
	if !foundStop {
		t.Errorf("stop = %v, want to contain END", got.Chat.Stop)
	}
	if got.Chat.FrequencyPenalty == nil || *got.Chat.FrequencyPenalty != 0.2 {
		t.Errorf("frequency_penalty = %v, want 0.2", got.Chat.FrequencyPenalty)
	}
	if got.Chat.PresencePenalty == nil || *got.Chat.PresencePenalty != 0.3 {
		t.Errorf("presence_penalty = %v, want 0.3", got.Chat.PresencePenalty)
	}
	if got.Chat.RepetitionPenalty == nil || *got.Chat.RepetitionPenalty != 1.2 {
		t.Errorf("repetition_penalty = %v, want 1.2 (from repeat_penalty)", got.Chat.RepetitionPenalty)
	}
	if got.Chat.MinP == nil || *got.Chat.MinP != 0.05 {
		t.Errorf("min_p = %v, want 0.05", got.Chat.MinP)
	}
}

// TestOllama_ChatImages verifies that an Ollama message with images is
// translated to a multimodal content array upstream.
func TestOllama_ChatImages(t *testing.T) {
	h := newHarness(t)
	defer h.Close()

	client := newOllamaClient(t, h)

	rawImage, err := base64.StdEncoding.DecodeString("iVBORw0KGgo=")
	if err != nil {
		t.Fatalf("decode test image: %v", err)
	}

	stream := false
	err = client.Chat(context.Background(), &api.ChatRequest{
		Model:  localModel,
		Stream: &stream,
		Messages: []api.Message{
			{
				Role:    "user",
				Content: "What is this?",
				Images:  []api.ImageData{api.ImageData(rawImage)},
			},
		},
	}, func(resp api.ChatResponse) error { return nil })
	if err != nil {
		t.Fatalf("chat: %v", err)
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

// TestOllama_ChatTools verifies that Ollama tools are forwarded upstream.
func TestOllama_ChatTools(t *testing.T) {
	h := newHarness(t)
	defer h.Close()

	client := newOllamaClient(t, h)

	stream := false
	err := client.Chat(context.Background(), &api.ChatRequest{
		Model:  localModel,
		Stream: &stream,
		Messages: []api.Message{
			{Role: "user", Content: "weather?"},
		},
		Tools: api.Tools{
			{
				Type: "function",
				Function: api.ToolFunction{
					Name:        "get_weather",
					Description: "Get weather",
					Parameters: api.ToolFunctionParameters{
						Type: "object",
						Properties: func() *api.ToolPropertiesMap {
							p := api.NewToolPropertiesMap()
							return p
						}(),
						Required: []string{},
					},
				},
			},
		},
	}, func(resp api.ChatResponse) error { return nil })
	if err != nil {
		t.Fatalf("chat: %v", err)
	}

	got := h.LastChat(t)
	toolsStr := string(got.Chat.Tools)
	if !strings.Contains(toolsStr, "get_weather") {
		t.Errorf("tools not forwarded: %s", toolsStr)
	}
	if !strings.Contains(toolsStr, `"type":"function"`) {
		t.Errorf("tools should be in OpenAI function schema: %s", toolsStr)
	}
}

// TestOllama_ChatStream verifies streaming chat content is reassembled and the
// upstream request has stream=true.
func TestOllama_ChatStream(t *testing.T) {
	h := newHarness(t)
	defer h.Close()

	client := newOllamaClient(t, h)

	var content strings.Builder
	err := client.Chat(context.Background(), &api.ChatRequest{
		Model:    localModel,
		Messages: []api.Message{{Role: "user", Content: "Stream please"}},
		// Stream defaults to true in the Ollama SDK.
	}, func(resp api.ChatResponse) error {
		content.WriteString(resp.Message.Content)
		return nil
	})
	if err != nil {
		t.Fatalf("chat stream: %v", err)
	}

	if content.String() != "Hello world" {
		t.Errorf("streamed content = %q, want %q", content.String(), "Hello world")
	}

	got := h.LastChat(t)
	if !got.Chat.Stream {
		t.Error("upstream request should have stream=true")
	}
}

// TestOllama_List verifies the proxy advertises the local model alias via the
// Ollama list endpoint.
func TestOllama_List(t *testing.T) {
	h := newHarness(t)
	defer h.Close()

	client := newOllamaClient(t, h)

	list, err := client.List(context.Background())
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if list == nil {
		t.Fatal("list response is nil")
	}

	var ids []string
	for _, m := range list.Models {
		ids = append(ids, m.Name)
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
