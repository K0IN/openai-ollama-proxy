package main

import (
	"encoding/json"
	"testing"
)

func init() {
	cfg = Config{
		VLLMModel:          "test-model",
		ModelName:          "qwen3:latest",
		ModelContextLength: 65536,
		OllamaVersion:      "0.7.0",
	}
}

func boolPtr(b bool) *bool { return &b }

func TestOllamaChatToOpenAI_Basic(t *testing.T) {
	req := OllamaChatRequest{
		Model: "qwen3:latest",
		Messages: []OllamaMessage{
			{Role: "user", Content: "Hello"},
		},
		Stream: boolPtr(false),
	}

	got, err := ollamaChatToOpenAI(req)
	if err != nil {
		t.Fatal(err)
	}
	if got.Stream {
		t.Error("Stream should be false")
	}
	if len(got.Messages) != 1 {
		t.Fatalf("len(Messages) = %d, want 1", len(got.Messages))
	}
	if got.Messages[0].Role != "user" {
		t.Errorf("Messages[0].Role = %q, want %q", got.Messages[0].Role, "user")
	}
	var content string
	json.Unmarshal(got.Messages[0].Content, &content)
	if content != "Hello" {
		t.Errorf("Messages[0].Content = %q, want %q", content, "Hello")
	}
}

func TestOllamaChatToOpenAI_StreamDefault(t *testing.T) {
	req := OllamaChatRequest{
		Model:    "qwen3:latest",
		Messages: []OllamaMessage{{Role: "user", Content: "Hi"}},
	}

	got, err := ollamaChatToOpenAI(req)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Stream {
		t.Error("Stream should default to true")
	}
}

func TestOllamaChatToOpenAI_FormatJSON(t *testing.T) {
	req := OllamaChatRequest{
		Model:    "qwen3:latest",
		Messages: []OllamaMessage{{Role: "user", Content: "json"}},
		Stream:   boolPtr(false),
		Format:   json.RawMessage(`"json"`),
	}

	got, err := ollamaChatToOpenAI(req)
	if err != nil {
		t.Fatal(err)
	}
	if got.ResponseFormat == nil {
		t.Fatal("ResponseFormat should not be nil")
	}
	if got.ResponseFormat.Type != "json_object" {
		t.Errorf("ResponseFormat.Type = %q, want %q", got.ResponseFormat.Type, "json_object")
	}
}

func TestOllamaChatToOpenAI_FormatSchema(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`)
	req := OllamaChatRequest{
		Model:    "qwen3:latest",
		Messages: []OllamaMessage{{Role: "user", Content: "schema"}},
		Stream:   boolPtr(false),
		Format:   schema,
	}

	got, err := ollamaChatToOpenAI(req)
	if err != nil {
		t.Fatal(err)
	}
	if got.ResponseFormat == nil {
		t.Fatal("ResponseFormat should not be nil")
	}
	if got.ResponseFormat.Type != "json_schema" {
		t.Errorf("ResponseFormat.Type = %q, want %q", got.ResponseFormat.Type, "json_schema")
	}
}

func TestConvertMessagesToOpenAI_Images(t *testing.T) {
	msgs := []OllamaMessage{
		{
			Role:    "user",
			Content: "What is this?",
			Images:  []string{"iVBORw0KGgo="}, // PNG magic bytes (base64 of 0x89 0x50...)
		},
	}

	got, err := convertMessagesToOpenAI(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("len = %d, want 1", len(got))
	}

	var parts []OpenAIContentPart
	if err := json.Unmarshal(got[0].Content, &parts); err != nil {
		t.Fatal(err)
	}
	if len(parts) != 2 {
		t.Fatalf("len(parts) = %d, want 2", len(parts))
	}
	if parts[0].Type != "text" || parts[0].Text != "What is this?" {
		t.Errorf("parts[0] = %+v", parts[0])
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil {
		t.Errorf("parts[1] = %+v", parts[1])
	}
}

func TestConvertMessagesToOpenAI_ToolCalls(t *testing.T) {
	msgs := []OllamaMessage{
		{Role: "user", Content: "What is the weather?"},
		{
			Role:    "assistant",
			Content: "",
			ToolCalls: []OllamaToolCall{
				{Function: OllamaToolCallFunction{
					Name:      "get_weather",
					Arguments: json.RawMessage(`{"city":"NYC"}`),
				}},
			},
		},
		{
			Role:     "tool",
			Content:  `{"temp": 72}`,
			ToolName: "get_weather",
		},
	}

	got, err := convertMessagesToOpenAI(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("len = %d, want 3", len(got))
	}

	// Assistant message should have tool calls with generated IDs
	if len(got[1].ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(got[1].ToolCalls))
	}
	tc := got[1].ToolCalls[0]
	if tc.Function.Name != "get_weather" {
		t.Errorf("Name = %q, want %q", tc.Function.Name, "get_weather")
	}
	if tc.Type != "function" {
		t.Errorf("Type = %q, want %q", tc.Type, "function")
	}
	if tc.ID == "" {
		t.Error("ID should not be empty")
	}

	// Tool response should reference the same ID
	if got[2].ToolCallID != tc.ID {
		t.Errorf("ToolCallID = %q, want %q", got[2].ToolCallID, tc.ID)
	}
}

func TestOpenAIChatToOllama(t *testing.T) {
	stop := "stop"
	content := "Hello, world!"
	resp := OpenAIChatResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1700000000,
		Model:   "test-model",
		Choices: []OpenAIChoice{
			{
				Index:        0,
				Message:      &OpenAIRespMsg{Role: "assistant", Content: &content},
				FinishReason: &stop,
			},
		},
		Usage: &OpenAIUsage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}

	got := openAIChatToOllama(resp, "qwen3:latest")
	if got.Model != "qwen3:latest" {
		t.Errorf("Model = %q, want %q", got.Model, "qwen3:latest")
	}
	if !got.Done {
		t.Error("Done should be true")
	}
	if got.DoneReason != "stop" {
		t.Errorf("DoneReason = %q, want %q", got.DoneReason, "stop")
	}
	if got.Message.Role != "assistant" {
		t.Errorf("Message.Role = %q, want %q", got.Message.Role, "assistant")
	}
	if got.Message.Content != "Hello, world!" {
		t.Errorf("Message.Content = %q, want %q", got.Message.Content, "Hello, world!")
	}
	if got.PromptEvalCount != 10 {
		t.Errorf("PromptEvalCount = %d, want 10", got.PromptEvalCount)
	}
	if got.EvalCount != 5 {
		t.Errorf("EvalCount = %d, want 5", got.EvalCount)
	}
}

func TestOpenAIChatToOllama_WithReasoning(t *testing.T) {
	content := "42"
	reasoning := "Let me think about this..."
	resp := OpenAIChatResponse{
		Choices: []OpenAIChoice{
			{
				Message: &OpenAIRespMsg{
					Role:      "assistant",
					Content:   &content,
					Reasoning: &reasoning,
				},
			},
		},
	}

	got := openAIChatToOllama(resp, "qwen3:latest")
	if got.Message.Thinking != "Let me think about this..." {
		t.Errorf("Thinking = %q, want %q", got.Message.Thinking, "Let me think about this...")
	}
}

func TestOpenAIChatToOllama_WithReasoningContent(t *testing.T) {
	content := "42"
	reasoning := "Detailed chain of thought"
	resp := OpenAIChatResponse{
		Choices: []OpenAIChoice{
			{
				Message: &OpenAIRespMsg{
					Role:             "assistant",
					Content:          &content,
					ReasoningContent: &reasoning,
				},
			},
		},
	}

	got := openAIChatToOllama(resp, "qwen3:latest")
	if got.Message.Thinking != reasoning {
		t.Errorf("Thinking = %q, want %q", got.Message.Thinking, reasoning)
	}
}

func TestOpenAIChatToOllama_ToolCalls(t *testing.T) {
	resp := OpenAIChatResponse{
		Choices: []OpenAIChoice{
			{
				Message: &OpenAIRespMsg{
					Role: "assistant",
					ToolCalls: []OpenAIToolCall{
						{
							ID:   "call_123",
							Type: "function",
							Function: OpenAIToolCallFunction{
								Name:      "get_weather",
								Arguments: `{"city":"NYC"}`,
							},
						},
					},
				},
			},
		},
	}

	got := openAIChatToOllama(resp, "qwen3:latest")
	if len(got.Message.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(got.Message.ToolCalls))
	}
	tc := got.Message.ToolCalls[0]
	if tc.Function.Name != "get_weather" {
		t.Errorf("Name = %q, want %q", tc.Function.Name, "get_weather")
	}
	if string(tc.Function.Arguments) != `{"city":"NYC"}` {
		t.Errorf("Arguments = %s, want %s", tc.Function.Arguments, `{"city":"NYC"}`)
	}
}

func TestOpenAIStreamChunkToOllama(t *testing.T) {
	content := "Hi"
	chunk := OpenAIChatResponse{
		Choices: []OpenAIChoice{
			{Delta: &OpenAIRespMsg{Role: "assistant", Content: &content}},
		},
	}

	got := openAIStreamChunkToOllama(chunk, "qwen3:latest")
	if got.Done {
		t.Error("Done should be false for content chunk")
	}
	if got.Message.Role != "assistant" {
		t.Errorf("Role = %q, want %q", got.Message.Role, "assistant")
	}
	if got.Message.Content != "Hi" {
		t.Errorf("Content = %q, want %q", got.Message.Content, "Hi")
	}
}

func TestOpenAIStreamChunkToOllama_Done(t *testing.T) {
	stop := "stop"
	chunk := OpenAIChatResponse{
		Choices: []OpenAIChoice{
			{FinishReason: &stop},
		},
	}

	got := openAIStreamChunkToOllama(chunk, "qwen3:latest")
	if !got.Done {
		t.Error("Done should be true")
	}
	if got.Message.Role != "assistant" {
		t.Errorf("Role = %q, want %q", got.Message.Role, "assistant")
	}
	if got.DoneReason != "stop" {
		t.Errorf("DoneReason = %q, want %q", got.DoneReason, "stop")
	}
}

func TestOpenAIStreamChunkToOllama_Usage(t *testing.T) {
	chunk := OpenAIChatResponse{
		Usage: &OpenAIUsage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30},
	}

	got := openAIStreamChunkToOllama(chunk, "qwen3:latest")
	if !got.Done {
		t.Error("Done should be true when usage is present")
	}
	if got.Message.Role != "assistant" {
		t.Errorf("Role = %q, want %q", got.Message.Role, "assistant")
	}
	if got.PromptEvalCount != 10 {
		t.Errorf("PromptEvalCount = %d, want 10", got.PromptEvalCount)
	}
	if got.EvalCount != 20 {
		t.Errorf("EvalCount = %d, want 20", got.EvalCount)
	}
}

func TestMapFinishReason(t *testing.T) {
	tests := []struct {
		input, want string
	}{
		{"stop", "stop"},
		{"tool_calls", "stop"},
	}
	for _, tt := range tests {
		got := mapFinishReason(tt.input)
		if got != tt.want {
			t.Errorf("mapFinishReason(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestOpenAIEmbedToOllama(t *testing.T) {
	resp := OpenAIEmbedResponse{
		Data: []OpenAIEmbedData{
			{Embedding: []float64{0.1, 0.2, 0.3}, Index: 0},
			{Embedding: []float64{0.4, 0.5, 0.6}, Index: 1},
		},
		Usage: &OpenAIUsage{PromptTokens: 5},
	}

	got := openAIEmbedToOllama(resp, "embed-model")
	if got.Model != "embed-model" {
		t.Errorf("Model = %q, want %q", got.Model, "embed-model")
	}
	if len(got.Embeddings) != 2 {
		t.Fatalf("len(Embeddings) = %d, want 2", len(got.Embeddings))
	}
	if got.Embeddings[0][0] != 0.1 {
		t.Errorf("Embeddings[0][0] = %v, want 0.1", got.Embeddings[0][0])
	}
	if got.PromptEvalCount != 5 {
		t.Errorf("PromptEvalCount = %d, want 5", got.PromptEvalCount)
	}
}
