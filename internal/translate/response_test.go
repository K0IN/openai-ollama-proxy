package translate

import (
	"testing"

	"github.com/k0in/openai-ollama-proxy/internal/types"
)

func TestOpenAIChatToOllama_Basic(t *testing.T) {
	resp := types.OpenAIChatResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "qwen3:latest",
		Choices: []types.OpenAIChoice{
			{
				Index: 0,
				Message: &types.OpenAIRespMsg{
					Role:    "assistant",
					Content: ptr("Hello!"),
				},
				FinishReason: ptr("stop"),
			},
		},
	}

	got := OpenAIChatToOllama(resp, "qwen3:latest")
	if got.Model != "qwen3:latest" {
		t.Errorf("Model = %q, want %q", got.Model, "qwen3:latest")
	}
	if got.Message.Content != "Hello!" {
		t.Errorf("Content = %q, want %q", got.Message.Content, "Hello!")
	}
	if !got.Done {
		t.Error("Done should be true")
	}
}

func TestOpenAIChatToOllama_WithToolCalls(t *testing.T) {
	resp := types.OpenAIChatResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "qwen3:latest",
		Choices: []types.OpenAIChoice{
			{
				Index: 0,
				Message: &types.OpenAIRespMsg{
					Role: "assistant",
					ToolCalls: []types.OpenAIToolCall{
						{
							ID:   "call_1",
							Type: "function",
							Function: types.OpenAIToolCallFunction{
								Name:      "get_weather",
								Arguments: `{"city":"NYC"}`,
							},
						},
					},
				},
				FinishReason: ptr("tool_calls"),
			},
		},
	}

	got := OpenAIChatToOllama(resp, "qwen3:latest")
	if len(got.Message.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(got.Message.ToolCalls))
	}
	if got.Message.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("ToolCall name = %q, want %q", got.Message.ToolCalls[0].Function.Name, "get_weather")
	}
}

func TestOpenAIChatToOllama_EmptyContent(t *testing.T) {
	resp := types.OpenAIChatResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "qwen3:latest",
		Choices: []types.OpenAIChoice{
			{
				Index: 0,
				Message: &types.OpenAIRespMsg{
					Role:    "assistant",
					Content: ptr(""),
				},
				FinishReason: ptr("stop"),
			},
		},
	}

	got := OpenAIChatToOllama(resp, "qwen3:latest")
	if got.Message.Content != "" {
		t.Errorf("Content = %q, want empty", got.Message.Content)
	}
}

func TestOpenAIStreamChunkToOllama_TextChunk(t *testing.T) {
	resp := types.OpenAIChatResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion.chunk",
		Created: 1234567890,
		Model:   "qwen3:latest",
		Choices: []types.OpenAIChoice{
			{
				Index: 0,
				Delta: &types.OpenAIRespMsg{
					Role:    "assistant",
					Content: ptr("Hello"),
				},
			},
		},
	}

	got := OpenAIStreamChunkToOllama(resp, "qwen3:latest")
	if got.Model != "qwen3:latest" {
		t.Errorf("Model = %q, want %q", got.Model, "qwen3:latest")
	}
	if got.Message.Content != "Hello" {
		t.Errorf("Content = %q, want %q", got.Message.Content, "Hello")
	}
}

func TestOpenAIStreamChunkToOllama_DoneChunk(t *testing.T) {
	resp := types.OpenAIChatResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion.chunk",
		Created: 1234567890,
		Model:   "qwen3:latest",
		Choices: []types.OpenAIChoice{
			{
				Index:        0,
				FinishReason: ptr("stop"),
			},
		},
	}

	got := OpenAIStreamChunkToOllama(resp, "qwen3:latest")
	if !got.Done {
		t.Error("Done should be true")
	}
}

func TestOpenAIChatToOllamaGenerate_Basic(t *testing.T) {
	resp := types.OpenAIChatResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "qwen3:latest",
		Choices: []types.OpenAIChoice{
			{
				Index: 0,
				Message: &types.OpenAIRespMsg{
					Role:    "assistant",
					Content: ptr("Generated text"),
				},
				FinishReason: ptr("stop"),
			},
		},
	}

	got := OpenAIChatToOllamaGenerate(resp, "qwen3:latest")
	if got.Response != "Generated text" {
		t.Errorf("Response = %q, want %q", got.Response, "Generated text")
	}
	if !got.Done {
		t.Error("Done should be true")
	}
}

func TestOpenAIStreamChunkToOllamaGenerate_TextChunk(t *testing.T) {
	resp := types.OpenAIChatResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion.chunk",
		Created: 1234567890,
		Model:   "qwen3:latest",
		Choices: []types.OpenAIChoice{
			{
				Index: 0,
				Delta: &types.OpenAIRespMsg{
					Content: ptr("Hello"),
				},
			},
		},
	}

	got := OpenAIStreamChunkToOllamaGenerate(resp, "qwen3:latest")
	if got.Response != "Hello" {
		t.Errorf("Response = %q, want %q", got.Response, "Hello")
	}
}

func TestOpenAIStreamChunkToOllamaGenerate_DoneChunk(t *testing.T) {
	resp := types.OpenAIChatResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion.chunk",
		Created: 1234567890,
		Model:   "qwen3:latest",
		Choices: []types.OpenAIChoice{
			{
				Index:        0,
				FinishReason: ptr("stop"),
			},
		},
	}

	got := OpenAIStreamChunkToOllamaGenerate(resp, "qwen3:latest")
	if !got.Done {
		t.Error("Done should be true")
	}
}

func TestOpenAIEmbedToOllama_Basic(t *testing.T) {
	resp := types.OpenAIEmbedResponse{
		Object: "list",
		Data: []types.OpenAIEmbedData{
			{
				Object:    `""`,
				Index:     0,
				Embedding: []float64{0.1, 0.2, 0.3},
			},
		},
		Model: "qwen3:latest",
	}

	got := OpenAIEmbedToOllama(resp, "qwen3:latest")
	if len(got.Embeddings) != 1 {
		t.Fatalf("len(Embeddings) = %d, want 1", len(got.Embeddings))
	}
	if len(got.Embeddings[0]) != 3 {
		t.Fatalf("len(Embeddings[0]) = %d, want 3", len(got.Embeddings[0]))
	}
	if got.Embeddings[0][0] != 0.1 {
		t.Errorf("Embeddings[0][0] = %v, want 0.1", got.Embeddings[0][0])
	}
}

func TestOpenAIEmbedToOllama_MultipleEmbeddings(t *testing.T) {
	resp := types.OpenAIEmbedResponse{
		Object: "list",
		Data: []types.OpenAIEmbedData{
			{Index: 0, Embedding: []float64{0.1}},
			{Index: 1, Embedding: []float64{0.2}},
			{Index: 2, Embedding: []float64{0.3}},
		},
		Model: "qwen3:latest",
	}

	got := OpenAIEmbedToOllama(resp, "qwen3:latest")
	if len(got.Embeddings) != 3 {
		t.Fatalf("len(Embeddings) = %d, want 3", len(got.Embeddings))
	}
}

func TestMapFinishReason_Stop(t *testing.T) {
	got := mapFinishReason("stop")
	if got != "stop" {
		t.Errorf("mapFinishReason = %q, want %q", got, "stop")
	}
}

func TestMapFinishReason_ToolCalls(t *testing.T) {
	got := mapFinishReason("tool_calls")
	if got != "stop" {
		t.Errorf("mapFinishReason = %q, want %q", got, "stop")
	}
}

func TestMapFinishReason_Length(t *testing.T) {
	got := mapFinishReason("length")
	if got != "length" {
		t.Errorf("mapFinishReason = %q, want %q", got, "length")
	}
}

func TestMapFinishReason_Default(t *testing.T) {
	got := mapFinishReason("unknown")
	if got != "stop" {
		t.Errorf("mapFinishReason = %q, want %q", got, "stop")
	}
}

func TestOpenAIRespMsgToOllama_TextMessage(t *testing.T) {
	msg := &types.OpenAIRespMsg{
		Role:    "assistant",
		Content: ptr("Hello"),
	}

	got := OpenAIRespMsgToOllama(msg)
	if got.Role != "assistant" {
		t.Errorf("Role = %q, want %q", got.Role, "assistant")
	}
	if got.Content != "Hello" {
		t.Errorf("Content = %q, want %q", got.Content, "Hello")
	}
}

func TestOpenAIRespMsgToOllama_ToolCallMessage(t *testing.T) {
	msg := &types.OpenAIRespMsg{
		Role: "assistant",
		ToolCalls: []types.OpenAIToolCall{
			{
				ID:   "call_1",
				Type: "function",
				Function: types.OpenAIToolCallFunction{
					Name:      "get_weather",
					Arguments: `{"city":"NYC"}`,
				},
			},
		},
	}

	got := OpenAIRespMsgToOllama(msg)
	if len(got.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(got.ToolCalls))
	}
	if got.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("ToolCall name = %q, want %q", got.ToolCalls[0].Function.Name, "get_weather")
	}
}

func TestOpenAIRespMsgToOllama_ReasoningContent(t *testing.T) {
	msg := &types.OpenAIRespMsg{
		Role:             "assistant",
		Content:          ptr("Hello"),
		ReasoningContent: ptr("thinking process"),
	}

	got := OpenAIRespMsgToOllama(msg)
	if got.Content != "Hello" {
		t.Errorf("Content = %q, want %q", got.Content, "Hello")
	}
	if got.Thinking != "thinking process" {
		t.Errorf("Thinking = %q, want %q", got.Thinking, "thinking process")
	}
}

func TestOpenAIRespMsgToOllama_ReasoningOnly(t *testing.T) {
	msg := &types.OpenAIRespMsg{
		Role:             "assistant",
		ReasoningContent: ptr("thinking process"),
	}

	got := OpenAIRespMsgToOllama(msg)
	if got.Content != "" {
		t.Errorf("Content = %q, want empty (reasoning belongs in Thinking, not Content)", got.Content)
	}
	if got.Thinking != "thinking process" {
		t.Errorf("Thinking = %q, want %q", got.Thinking, "thinking process")
	}
}

func TestOpenAIStreamChunkToOllama_RoleOnly(t *testing.T) {
	resp := types.OpenAIChatResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion.chunk",
		Created: 1234567890,
		Model:   "qwen3:latest",
		Choices: []types.OpenAIChoice{
			{
				Index: 0,
				Delta: &types.OpenAIRespMsg{
					Role: "assistant",
				},
			},
		},
	}

	got := OpenAIStreamChunkToOllama(resp, "qwen3:latest")
	if got.Message.Role != "assistant" {
		t.Errorf("Role = %q, want %q", got.Message.Role, "assistant")
	}
}

func TestOpenAIStreamChunkToOllama_ToolCallChunk(t *testing.T) {
	resp := types.OpenAIChatResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion.chunk",
		Created: 1234567890,
		Model:   "qwen3:latest",
		Choices: []types.OpenAIChoice{
			{
				Index: 0,
				Delta: &types.OpenAIRespMsg{
					ToolCalls: []types.OpenAIToolCall{
						{
							ID:   "call_1",
							Type: "function",
							Function: types.OpenAIToolCallFunction{
								Name:      "get_weather",
								Arguments: `{"city":"NYC"}`,
							},
						},
					},
				},
			},
		},
	}

	got := OpenAIStreamChunkToOllama(resp, "qwen3:latest")
	if len(got.Message.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(got.Message.ToolCalls))
	}
}

func TestOpenAIChatToOllama_WithUsage(t *testing.T) {
	resp := types.OpenAIChatResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "qwen3:latest",
		Choices: []types.OpenAIChoice{
			{
				Index: 0,
				Message: &types.OpenAIRespMsg{
					Role:    "assistant",
					Content: ptr("Hello"),
				},
				FinishReason: ptr("stop"),
			},
		},
		Usage: &types.OpenAIUsage{
			PromptTokens:     10,
			CompletionTokens: 5,
		},
	}

	got := OpenAIChatToOllama(resp, "qwen3:latest")
	if got.PromptEvalCount != 10 {
		t.Errorf("PromptEvalCount = %d, want 10", got.PromptEvalCount)
	}
	if got.EvalCount != 5 {
		t.Errorf("EvalCount = %d, want 5", got.EvalCount)
	}
}

func TestOpenAIStreamChunkToOllama_WithUsage(t *testing.T) {
	resp := types.OpenAIChatResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion.chunk",
		Created: 1234567890,
		Model:   "qwen3:latest",
		Choices: []types.OpenAIChoice{
			{
				Index: 0,
				Delta: &types.OpenAIRespMsg{
					Content: ptr("Hello"),
				},
			},
		},
		Usage: &types.OpenAIUsage{
			PromptTokens:     10,
			CompletionTokens: 5,
		},
	}

	got := OpenAIStreamChunkToOllama(resp, "qwen3:latest")
	if !got.Done {
		t.Error("Done should be true when usage is present")
	}
	if got.PromptEvalCount != 10 {
		t.Errorf("PromptEvalCount = %d, want 10", got.PromptEvalCount)
	}
}

func TestOpenAIRespMsgToOllama_Reasoning(t *testing.T) {
	msg := &types.OpenAIRespMsg{
		Role:      "assistant",
		Content:   ptr("Hello"),
		Reasoning: ptr("thinking"),
	}

	got := OpenAIRespMsgToOllama(msg)
	if got.Content != "Hello" {
		t.Errorf("Content = %q, want %q", got.Content, "Hello")
	}
	if got.Thinking != "thinking" {
		t.Errorf("Thinking = %q, want %q", got.Thinking, "thinking")
	}
}

func ptr(s string) *string {
	return &s
}
