package translate

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/k0in/openai-ollama-proxy/internal/types"
)

func boolPtr(b bool) *bool { return &b }

func TestOllamaChatToOpenAI_Basic(t *testing.T) {
	req := types.OllamaChatRequest{
		Model: "qwen3:latest",
		Messages: []types.OllamaMessage{
			{Role: "user", Content: "Hello"},
		},
		Stream: boolPtr(false),
	}

	got, err := OllamaChatToOpenAI(req)
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
	if err := json.Unmarshal(got.Messages[0].Content, &content); err != nil {
		t.Fatal(err)
	}
	if content != "Hello" {
		t.Errorf("Messages[0].Content = %q, want %q", content, "Hello")
	}
}

func TestOllamaChatToOpenAI_StreamDefault(t *testing.T) {
	req := types.OllamaChatRequest{
		Model:    "qwen3:latest",
		Messages: []types.OllamaMessage{{Role: "user", Content: "Hi"}},
	}

	got, err := OllamaChatToOpenAI(req)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Stream {
		t.Error("Stream should default to true")
	}
}

func TestOllamaChatToOpenAI_MapsToolsOptionsAndThinking(t *testing.T) {
	temperature := 0.7
	topP := 0.8
	minP := 0.1
	topK := 40
	seed := 123
	numPredict := 64
	frequencyPenalty := 0.2
	presencePenalty := 0.3
	repeatPenalty := 1.1
	think := true
	tools := json.RawMessage(`[{"type":"function","function":{"name":"get_weather"}}]`)

	req := types.OllamaChatRequest{
		Model:    "qwen3:latest",
		Messages: []types.OllamaMessage{{Role: "user", Content: "Hi"}},
		Tools:    tools,
		Think:    &think,
		Options: types.OllamaOptions{
			Temperature:      &temperature,
			TopP:             &topP,
			MinP:             &minP,
			TopK:             &topK,
			Seed:             &seed,
			NumPredict:       &numPredict,
			Stop:             []string{"END", "STOP"},
			FrequencyPenalty: &frequencyPenalty,
			PresencePenalty:  &presencePenalty,
			RepeatPenalty:    &repeatPenalty,
		},
	}

	got, err := OllamaChatToOpenAI(req)
	if err != nil {
		t.Fatal(err)
	}
	if string(got.Tools) != string(tools) {
		t.Fatalf("Tools = %s, want %s", got.Tools, tools)
	}
	if got.StreamOptions == nil || !got.StreamOptions.IncludeUsage {
		t.Fatal("StreamOptions.IncludeUsage should be enabled for streaming requests")
	}
	if enabled, ok := got.ChatTemplateKwargs["enable_thinking"].(bool); !ok || !enabled {
		t.Fatalf("enable_thinking = %#v, want true", got.ChatTemplateKwargs["enable_thinking"])
	}
	if got.Temperature == nil || *got.Temperature != temperature {
		t.Fatalf("Temperature = %#v, want %v", got.Temperature, temperature)
	}
	if got.TopP == nil || *got.TopP != topP {
		t.Fatalf("TopP = %#v, want %v", got.TopP, topP)
	}
	if got.MinP == nil || *got.MinP != minP {
		t.Fatalf("MinP = %#v, want %v", got.MinP, minP)
	}
	if got.TopK == nil || *got.TopK != topK {
		t.Fatalf("TopK = %#v, want %v", got.TopK, topK)
	}
	if got.Seed == nil || *got.Seed != seed {
		t.Fatalf("Seed = %#v, want %v", got.Seed, seed)
	}
	if got.MaxTokens == nil || *got.MaxTokens != numPredict {
		t.Fatalf("MaxTokens = %#v, want %v", got.MaxTokens, numPredict)
	}
	if len(got.Stop) != 2 || got.Stop[0] != "END" || got.Stop[1] != "STOP" {
		t.Fatalf("Stop = %#v, want [END STOP]", got.Stop)
	}
	if got.FrequencyPenalty == nil || *got.FrequencyPenalty != frequencyPenalty {
		t.Fatalf("FrequencyPenalty = %#v, want %v", got.FrequencyPenalty, frequencyPenalty)
	}
	if got.PresencePenalty == nil || *got.PresencePenalty != presencePenalty {
		t.Fatalf("PresencePenalty = %#v, want %v", got.PresencePenalty, presencePenalty)
	}
	if got.RepetitionPenalty == nil || *got.RepetitionPenalty != repeatPenalty {
		t.Fatalf("RepetitionPenalty = %#v, want %v", got.RepetitionPenalty, repeatPenalty)
	}
}

func TestOllamaChatToOpenAI_FormatJSON(t *testing.T) {
	req := types.OllamaChatRequest{
		Model:    "qwen3:latest",
		Messages: []types.OllamaMessage{{Role: "user", Content: "json"}},
		Stream:   boolPtr(false),
		Format:   json.RawMessage(`"json"`),
	}

	got, err := OllamaChatToOpenAI(req)
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
	req := types.OllamaChatRequest{
		Model:    "qwen3:latest",
		Messages: []types.OllamaMessage{{Role: "user", Content: "schema"}},
		Stream:   boolPtr(false),
		Format:   schema,
	}

	got, err := OllamaChatToOpenAI(req)
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
	msgs := []types.OllamaMessage{
		{
			Role:    "user",
			Content: "What is this?",
			Images:  []string{"iVBORw0KGgo="},
		},
	}

	got, err := ConvertMessagesToOpenAI(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("len = %d, want 1", len(got))
	}

	var parts []types.OpenAIContentPart
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
	msgs := []types.OllamaMessage{
		{Role: "user", Content: "What is the weather?"},
		{
			Role:    "assistant",
			Content: "",
			ToolCalls: []types.OllamaToolCall{
				{Function: types.OllamaToolCallFunction{
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

	got, err := ConvertMessagesToOpenAI(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("len = %d, want 3", len(got))
	}
	if len(got[1].ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(got[1].ToolCalls))
	}

	toolCall := got[1].ToolCalls[0]
	if toolCall.Function.Name != "get_weather" {
		t.Errorf("Name = %q, want %q", toolCall.Function.Name, "get_weather")
	}
	if toolCall.Type != "function" {
		t.Errorf("Type = %q, want %q", toolCall.Type, "function")
	}
	if toolCall.ID == "" {
		t.Error("ID should not be empty")
	}
	if got[2].ToolCallID != toolCall.ID {
		t.Errorf("ToolCallID = %q, want %q", got[2].ToolCallID, toolCall.ID)
	}
}

func TestOllamaGenerateToOpenAI_MapsSystemImagesFormatAndThinking(t *testing.T) {
	stream := false
	think := true
	req := types.OllamaGenerateRequest{
		Model:  "qwen3:latest",
		Prompt: "What is in this picture?",
		System: "You are helpful.",
		Images: []string{"iVBORw0KGgo="},
		Stream: &stream,
		Format: json.RawMessage(`"json"`),
		Think:  &think,
	}

	got, err := OllamaGenerateToOpenAI(req)
	if err != nil {
		t.Fatal(err)
	}
	if got.Stream {
		t.Fatal("Stream should be false")
	}
	if got.ResponseFormat == nil || got.ResponseFormat.Type != "json_object" {
		t.Fatalf("ResponseFormat = %#v, want json_object", got.ResponseFormat)
	}
	if enabled, ok := got.ChatTemplateKwargs["enable_thinking"].(bool); !ok || !enabled {
		t.Fatalf("enable_thinking = %#v, want true", got.ChatTemplateKwargs["enable_thinking"])
	}
	if len(got.Messages) != 2 {
		t.Fatalf("len(Messages) = %d, want 2", len(got.Messages))
	}
	if got.Messages[0].Role != "system" {
		t.Fatalf("Messages[0].Role = %q, want %q", got.Messages[0].Role, "system")
	}

	var system string
	if err := json.Unmarshal(got.Messages[0].Content, &system); err != nil {
		t.Fatal(err)
	}
	if system != "You are helpful." {
		t.Fatalf("system = %q, want %q", system, "You are helpful.")
	}

	if got.Messages[1].Role != "user" {
		t.Fatalf("Messages[1].Role = %q, want %q", got.Messages[1].Role, "user")
	}
	var parts []types.OpenAIContentPart
	if err := json.Unmarshal(got.Messages[1].Content, &parts); err != nil {
		t.Fatal(err)
	}
	if len(parts) != 2 {
		t.Fatalf("len(parts) = %d, want 2", len(parts))
	}
	if parts[0].Type != "text" || parts[0].Text != "What is in this picture?" {
		t.Fatalf("parts[0] = %+v, want text part", parts[0])
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil {
		t.Fatalf("parts[1] = %+v, want image_url part", parts[1])
	}
	if !strings.HasPrefix(parts[1].ImageURL.URL, "data:image/png;base64,") {
		t.Fatalf("image URL = %q, want PNG data URL", parts[1].ImageURL.URL)
	}
}

func TestOpenAIChatToOllama(t *testing.T) {
	stop := "stop"
	content := "Hello, world!"
	resp := types.OpenAIChatResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1700000000,
		Model:   "test-model",
		Choices: []types.OpenAIChoice{
			{
				Index:        0,
				Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
				FinishReason: &stop,
			},
		},
		Usage: &types.OpenAIUsage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}

	got := OpenAIChatToOllama(resp, "qwen3:latest")
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
	resp := types.OpenAIChatResponse{
		Choices: []types.OpenAIChoice{
			{
				Message: &types.OpenAIRespMsg{
					Role:      "assistant",
					Content:   &content,
					Reasoning: &reasoning,
				},
			},
		},
	}

	got := OpenAIChatToOllama(resp, "qwen3:latest")
	if got.Message.Thinking != "Let me think about this..." {
		t.Errorf("Thinking = %q, want %q", got.Message.Thinking, "Let me think about this...")
	}
}

func TestOpenAIChatToOllama_WithReasoningContent(t *testing.T) {
	content := "42"
	reasoning := "Detailed chain of thought"
	resp := types.OpenAIChatResponse{
		Choices: []types.OpenAIChoice{
			{
				Message: &types.OpenAIRespMsg{
					Role:             "assistant",
					Content:          &content,
					ReasoningContent: &reasoning,
				},
			},
		},
	}

	got := OpenAIChatToOllama(resp, "qwen3:latest")
	if got.Message.Thinking != reasoning {
		t.Errorf("Thinking = %q, want %q", got.Message.Thinking, reasoning)
	}
}

func TestOpenAIChatToOllama_ToolCalls(t *testing.T) {
	resp := types.OpenAIChatResponse{
		Choices: []types.OpenAIChoice{
			{
				Message: &types.OpenAIRespMsg{
					Role: "assistant",
					ToolCalls: []types.OpenAIToolCall{
						{
							ID:   "call_123",
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

	got := OpenAIChatToOllama(resp, "qwen3:latest")
	if len(got.Message.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(got.Message.ToolCalls))
	}
	toolCall := got.Message.ToolCalls[0]
	if toolCall.Function.Name != "get_weather" {
		t.Errorf("Name = %q, want %q", toolCall.Function.Name, "get_weather")
	}
	if string(toolCall.Function.Arguments) != `{"city":"NYC"}` {
		t.Errorf("Arguments = %s, want %s", toolCall.Function.Arguments, `{"city":"NYC"}`)
	}
}

func TestOpenAIStreamChunkToOllama(t *testing.T) {
	content := "Hi"
	chunk := types.OpenAIChatResponse{
		Choices: []types.OpenAIChoice{
			{Delta: &types.OpenAIRespMsg{Role: "assistant", Content: &content}},
		},
	}

	got := OpenAIStreamChunkToOllama(chunk, "qwen3:latest")
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
	chunk := types.OpenAIChatResponse{
		Choices: []types.OpenAIChoice{
			{FinishReason: &stop},
		},
	}

	got := OpenAIStreamChunkToOllama(chunk, "qwen3:latest")
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
	chunk := types.OpenAIChatResponse{
		Usage: &types.OpenAIUsage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30},
	}

	got := OpenAIStreamChunkToOllama(chunk, "qwen3:latest")
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
		input string
		want  string
	}{
		{input: "stop", want: "stop"},
		{input: "tool_calls", want: "stop"},
	}
	for _, testCase := range tests {
		got := mapFinishReason(testCase.input)
		if got != testCase.want {
			t.Errorf("mapFinishReason(%q) = %q, want %q", testCase.input, got, testCase.want)
		}
	}
}

func TestOpenAIEmbedToOllama(t *testing.T) {
	resp := types.OpenAIEmbedResponse{
		Data: []types.OpenAIEmbedData{
			{Embedding: []float64{0.1, 0.2, 0.3}, Index: 0},
			{Embedding: []float64{0.4, 0.5, 0.6}, Index: 1},
		},
		Usage: &types.OpenAIUsage{PromptTokens: 5},
	}

	got := OpenAIEmbedToOllama(resp, "embed-model")
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
