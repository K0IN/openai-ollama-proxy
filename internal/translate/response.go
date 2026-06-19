package translate

import (
	"encoding/json"
	"time"

	"github.com/k0in/openai-ollama-proxy/internal/types"
)

func OpenAIChatToOllamaGenerate(resp types.OpenAIChatResponse, model string) types.OllamaGenerateResponse {
	chatResp := OpenAIChatToOllama(resp, model)
	return types.OllamaGenerateResponse{
		Model:              chatResp.Model,
		CreatedAt:          chatResp.CreatedAt,
		Response:           chatResp.Message.Content,
		Thinking:           chatResp.Message.Thinking,
		Done:               chatResp.Done,
		DoneReason:         chatResp.DoneReason,
		TotalDuration:      chatResp.TotalDuration,
		LoadDuration:       chatResp.LoadDuration,
		PromptEvalCount:    chatResp.PromptEvalCount,
		PromptEvalDuration: chatResp.PromptEvalDuration,
		EvalCount:          chatResp.EvalCount,
		EvalDuration:       chatResp.EvalDuration,
		ToolCalls:          chatResp.Message.ToolCalls,
	}
}

func OpenAIStreamChunkToOllamaGenerate(resp types.OpenAIChatResponse, model string) types.OllamaGenerateResponse {
	chatResp := OpenAIStreamChunkToOllama(resp, model)
	return types.OllamaGenerateResponse{
		Model:              chatResp.Model,
		CreatedAt:          chatResp.CreatedAt,
		Response:           chatResp.Message.Content,
		Thinking:           chatResp.Message.Thinking,
		Done:               chatResp.Done,
		DoneReason:         chatResp.DoneReason,
		TotalDuration:      chatResp.TotalDuration,
		LoadDuration:       chatResp.LoadDuration,
		PromptEvalCount:    chatResp.PromptEvalCount,
		PromptEvalDuration: chatResp.PromptEvalDuration,
		EvalCount:          chatResp.EvalCount,
		EvalDuration:       chatResp.EvalDuration,
		ToolCalls:          chatResp.Message.ToolCalls,
	}
}

func OpenAIChatToOllama(resp types.OpenAIChatResponse, model string) types.OllamaChatResponse {
	now := time.Now().UTC().Format(time.RFC3339Nano)
	out := types.OllamaChatResponse{
		Model:     model,
		CreatedAt: now,
		Done:      true,
		Message:   types.OllamaMessage{Role: "assistant"},
	}

	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		if choice.Message != nil {
			out.Message = OpenAIRespMsgToOllama(choice.Message)
		}
		if choice.FinishReason != nil {
			out.DoneReason = mapFinishReason(*choice.FinishReason)
		}
	}

	applyUsage(&out, resp.Usage)

	return out
}

func OpenAIRespMsgToOllama(message *types.OpenAIRespMsg) types.OllamaMessage {
	out := types.OllamaMessage{Role: message.Role}
	if message.Content != nil {
		out.Content = *message.Content
	}

	reasoning := ""
	if message.ReasoningContent != nil {
		reasoning = *message.ReasoningContent
	} else if message.Reasoning != nil {
		reasoning = *message.Reasoning
	}
	if reasoning != "" {
		out.Thinking = reasoning
	}

	if len(message.ToolCalls) > 0 {
		out.ToolCalls = make([]types.OllamaToolCall, len(message.ToolCalls))
		for i, toolCall := range message.ToolCalls {
			out.ToolCalls[i] = types.OllamaToolCall{
				Function: types.OllamaToolCallFunction{
					Name:      toolCall.Function.Name,
					Arguments: json.RawMessage(toolCall.Function.Arguments),
				},
			}
		}
	}

	return out
}

func OpenAIStreamChunkToOllama(resp types.OpenAIChatResponse, model string) types.OllamaChatResponse {
	now := time.Now().UTC().Format(time.RFC3339Nano)
	out := types.OllamaChatResponse{
		Model:     model,
		CreatedAt: now,
		Done:      false,
		Message:   types.OllamaMessage{Role: "assistant"},
	}

	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		if choice.Delta != nil {
			message := OpenAIRespMsgToOllama(choice.Delta)
			if message.Role == "" {
				message.Role = "assistant"
			}
			out.Message = message
		}
		if choice.FinishReason != nil {
			out.Done = true
			out.DoneReason = mapFinishReason(*choice.FinishReason)
		}
	}

	if resp.Usage != nil {
		out.Done = true
		applyUsage(&out, resp.Usage)
	}

	return out
}

func OpenAIEmbedToOllama(resp types.OpenAIEmbedResponse, model string) types.OllamaEmbedResponse {
	embeddings := make([][]float64, len(resp.Data))
	for i, datum := range resp.Data {
		embeddings[i] = datum.Embedding
	}

	out := types.OllamaEmbedResponse{
		Model:      model,
		Embeddings: embeddings,
	}
	if resp.Usage != nil {
		applyEmbedUsage(&out, resp.Usage)
	}

	return out
}

func applyEmbedUsage(out *types.OllamaEmbedResponse, usage *types.OpenAIUsage) {
	if usage == nil {
		return
	}

	out.PromptEvalCount = usage.PromptTokens
}

func applyUsage(out *types.OllamaChatResponse, usage *types.OpenAIUsage) {
	if usage == nil {
		return
	}

	out.PromptEvalCount = usage.PromptTokens
	out.EvalCount = usage.CompletionTokens
}

func mapFinishReason(reason string) string {
	switch reason {
	case "stop":
		return "stop"
	case "length":
		return "length"
	case "tool_calls":
		return "stop"
	default:
		return "stop"
	}
}
