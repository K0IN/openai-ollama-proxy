package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// resolveModel returns the vLLM model name, ignoring whatever the client sent.
func resolveModel(_ string) string {
	return cfg.VLLMModel
}

// ollamaChatToOpenAI converts an Ollama /api/chat request to an OpenAI chat completion request.
func ollamaChatToOpenAI(req OllamaChatRequest) (OpenAIChatRequest, error) {
	stream := true
	if req.Stream != nil {
		stream = *req.Stream
	}

	out := OpenAIChatRequest{
		Model:              resolveModel(req.Model),
		Stream:             stream,
		Tools:              req.Tools,
		ChatTemplateKwargs: map[string]any{"enable_thinking": false},
	}

	if stream {
		out.StreamOptions = &OpenAIStreamOptions{IncludeUsage: true}
	}

	// Options → top-level params
	o := req.Options
	out.Temperature = o.Temperature
	out.TopP = o.TopP
	out.MinP = o.MinP
	out.Seed = o.Seed
	out.Stop = o.Stop
	out.FrequencyPenalty = o.FrequencyPenalty
	out.PresencePenalty = o.PresencePenalty
	out.TopK = o.TopK
	out.RepetitionPenalty = o.RepeatPenalty
	if o.NumPredict != nil {
		v := *o.NumPredict
		out.MaxTokens = &v
	}

	// Default to non-thinking responses so normal chat requests yield content.
	if req.Think != nil {
		out.ChatTemplateKwargs["enable_thinking"] = *req.Think
	}

	// Format → response_format
	if len(req.Format) > 0 {
		var rawStr string
		if err := json.Unmarshal(req.Format, &rawStr); err == nil {
			if rawStr == "json" {
				out.ResponseFormat = &OpenAIResponseFormat{Type: "json_object"}
			}
		} else {
			// Assume JSON schema object
			out.ResponseFormat = &OpenAIResponseFormat{
				Type:       "json_schema",
				JSONSchema: req.Format,
			}
		}
	}

	// Messages
	msgs, err := convertMessagesToOpenAI(req.Messages)
	if err != nil {
		return out, err
	}
	out.Messages = msgs

	return out, nil
}

// convertMessagesToOpenAI translates Ollama messages to OpenAI format.
func convertMessagesToOpenAI(msgs []OllamaMessage) ([]OpenAIMessage, error) {
	out := make([]OpenAIMessage, 0, len(msgs))
	// Track tool call IDs by function name so tool responses can reference them.
	toolCallIDs := map[string]string{}

	for _, m := range msgs {
		om := OpenAIMessage{Role: m.Role}

		// Handle tool calls from assistant
		if len(m.ToolCalls) > 0 {
			om.ToolCalls = make([]OpenAIToolCall, len(m.ToolCalls))
			for i, tc := range m.ToolCalls {
				id := fmt.Sprintf("call_%d", time.Now().UnixNano()+int64(i))
				toolCallIDs[tc.Function.Name] = id
				args := string(tc.Function.Arguments)
				om.ToolCalls[i] = OpenAIToolCall{
					ID:   id,
					Type: "function",
					Function: OpenAIToolCallFunction{
						Name:      tc.Function.Name,
						Arguments: args,
					},
				}
			}
		}

		// Handle tool response
		if m.Role == "tool" && m.ToolName != "" {
			if id, ok := toolCallIDs[m.ToolName]; ok {
				om.ToolCallID = id
			} else {
				om.ToolCallID = fmt.Sprintf("call_%s", m.ToolName)
			}
		}

		// Content: may include images
		if len(m.Images) > 0 {
			parts := make([]OpenAIContentPart, 0, len(m.Images)+1)
			if m.Content != "" {
				parts = append(parts, OpenAIContentPart{Type: "text", Text: m.Content})
			}
			for _, img := range m.Images {
				// Ollama sends raw base64; assume JPEG if no prefix
				dataURL := img
				if !strings.HasPrefix(img, "data:") {
					// Try to detect format from base64 header bytes
					mime := detectImageMIME(img)
					dataURL = fmt.Sprintf("data:%s;base64,%s", mime, img)
				}
				parts = append(parts, OpenAIContentPart{
					Type:     "image_url",
					ImageURL: &OpenAIImageURL{URL: dataURL},
				})
			}
			b, err := json.Marshal(parts)
			if err != nil {
				return nil, err
			}
			om.Content = b
		} else {
			b, err := json.Marshal(m.Content)
			if err != nil {
				return nil, err
			}
			om.Content = b
		}

		out = append(out, om)
	}
	return out, nil
}

// detectImageMIME peeks at the first bytes of a base64-encoded image to guess MIME type.
func detectImageMIME(b64 string) string {
	data, err := base64.StdEncoding.DecodeString(b64[:min(len(b64), 16)])
	if err != nil || len(data) < 4 {
		return "image/jpeg"
	}
	switch {
	case data[0] == 0x89 && data[1] == 0x50: // PNG
		return "image/png"
	case data[0] == 0x47 && data[1] == 0x49: // GIF
		return "image/gif"
	case data[0] == 0x52 && data[1] == 0x49: // WEBP (RIFF)
		return "image/webp"
	default:
		return "image/jpeg"
	}
}

func ollamaGenerateToOpenAI(req OllamaGenerateRequest) (OpenAIChatRequest, error) {
	chatReq := OllamaChatRequest{
		Model:  req.Model,
		Stream: req.Stream,
		Format: req.Format,
		Think:  req.Think,
		Options: OllamaOptions{
			Temperature:      req.Options.Temperature,
			TopP:             req.Options.TopP,
			MinP:             req.Options.MinP,
			TopK:             req.Options.TopK,
			Seed:             req.Options.Seed,
			NumPredict:       req.Options.NumPredict,
			Stop:             req.Options.Stop,
			FrequencyPenalty: req.Options.FrequencyPenalty,
			PresencePenalty:  req.Options.PresencePenalty,
			RepeatPenalty:    req.Options.RepeatPenalty,
		},
	}

	if req.System != "" {
		chatReq.Messages = append(chatReq.Messages, OllamaMessage{Role: "system", Content: req.System})
	}

	chatReq.Messages = append(chatReq.Messages, OllamaMessage{
		Role:    "user",
		Content: req.Prompt,
		Images:  req.Images,
	})

	return ollamaChatToOpenAI(chatReq)
}

func openAIChatToOllamaGenerate(resp OpenAIChatResponse, model string) OllamaGenerateResponse {
	chatResp := openAIChatToOllama(resp, model)
	return OllamaGenerateResponse{
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

func openAIStreamChunkToOllamaGenerate(resp OpenAIChatResponse, model string) OllamaGenerateResponse {
	chatResp := openAIStreamChunkToOllama(resp, model)
	return OllamaGenerateResponse{
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

// openAIChatToOllama converts a non-streaming OpenAI response to an Ollama chat response.
func openAIChatToOllama(resp OpenAIChatResponse, model string) OllamaChatResponse {
	now := time.Now().UTC().Format(time.RFC3339Nano)
	out := OllamaChatResponse{
		Model:     model,
		CreatedAt: now,
		Done:      true,
		Message:   OllamaMessage{Role: "assistant"},
	}

	if len(resp.Choices) > 0 {
		c := resp.Choices[0]
		if c.Message != nil {
			out.Message = openAIRespMsgToOllama(c.Message)
		}
		if c.FinishReason != nil {
			out.DoneReason = mapFinishReason(*c.FinishReason)
		}
	}

	if resp.Usage != nil {
		out.PromptEvalCount = resp.Usage.PromptTokens
		out.EvalCount = resp.Usage.CompletionTokens
		// Fake durations (nanoseconds) based on token counts
		out.PromptEvalDuration = int64(resp.Usage.PromptTokens) * 1_000_000 // ~1ms per token
		out.EvalDuration = int64(resp.Usage.CompletionTokens) * 10_000_000  // ~10ms per token
		out.TotalDuration = out.PromptEvalDuration + out.EvalDuration
		out.LoadDuration = 50_000_000 // 50ms fake
	}

	return out
}

// openAIRespMsgToOllama converts an OpenAI response message to Ollama format.
// If the upstream returns reasoning_content but no regular content (which can
// happen when vLLM ignores enable_thinking=false), fall back to using the
// reasoning text as content so callers that only read the content field (e.g.
// VS Code Copilot) still see the response.
func openAIRespMsgToOllama(m *OpenAIRespMsg) OllamaMessage {
	om := OllamaMessage{Role: m.Role}
	if m.Content != nil {
		om.Content = *m.Content
	}
	reasoning := ""
	if m.ReasoningContent != nil {
		reasoning = *m.ReasoningContent
	} else if m.Reasoning != nil {
		reasoning = *m.Reasoning
	}
	if reasoning != "" {
		if om.Content == "" {
			// No regular content — use reasoning as content so it isn't lost.
			om.Content = reasoning
		} else {
			om.Thinking = reasoning
		}
	}
	if len(m.ToolCalls) > 0 {
		om.ToolCalls = make([]OllamaToolCall, len(m.ToolCalls))
		for i, tc := range m.ToolCalls {
			om.ToolCalls[i] = OllamaToolCall{
				Function: OllamaToolCallFunction{
					Name:      tc.Function.Name,
					Arguments: json.RawMessage(tc.Function.Arguments),
				},
			}
		}
	}
	return om
}

// openAIStreamChunkToOllama converts a single SSE chunk to an Ollama streaming response.
func openAIStreamChunkToOllama(resp OpenAIChatResponse, model string) OllamaChatResponse {
	now := time.Now().UTC().Format(time.RFC3339Nano)
	out := OllamaChatResponse{
		Model:     model,
		CreatedAt: now,
		Done:      false,
		Message:   OllamaMessage{Role: "assistant"},
	}

	if len(resp.Choices) > 0 {
		c := resp.Choices[0]
		if c.Delta != nil {
			msg := openAIRespMsgToOllama(c.Delta)
			if msg.Role == "" {
				msg.Role = "assistant"
			}
			out.Message = msg
		}
		if c.FinishReason != nil {
			out.Done = true
			out.DoneReason = mapFinishReason(*c.FinishReason)
		}
	}

	// Usage comes in the final chunk when stream_options.include_usage=true
	if resp.Usage != nil {
		out.Done = true
		out.PromptEvalCount = resp.Usage.PromptTokens
		out.EvalCount = resp.Usage.CompletionTokens
		out.PromptEvalDuration = int64(resp.Usage.PromptTokens) * 1_000_000
		out.EvalDuration = int64(resp.Usage.CompletionTokens) * 10_000_000
		out.TotalDuration = out.PromptEvalDuration + out.EvalDuration
		out.LoadDuration = 50_000_000
	}

	return out
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

// openAIEmbedToOllama converts an OpenAI embedding response to Ollama /api/embed format.
func openAIEmbedToOllama(resp OpenAIEmbedResponse, model string) OllamaEmbedResponse {
	embeddings := make([][]float64, len(resp.Data))
	for i, d := range resp.Data {
		embeddings[i] = d.Embedding
	}
	out := OllamaEmbedResponse{
		Model:      model,
		Embeddings: embeddings,
	}
	if resp.Usage != nil {
		out.PromptEvalCount = resp.Usage.PromptTokens
	}
	return out
}
