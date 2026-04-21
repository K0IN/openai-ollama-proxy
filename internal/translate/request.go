package translate

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/k0in/openai-ollama-proxy/internal/types"
)

func OllamaChatToOpenAI(req types.OllamaChatRequest) (types.OpenAIChatRequest, error) {
	stream := true
	if req.Stream != nil {
		stream = *req.Stream
	}

	out := types.OpenAIChatRequest{
		Model:              req.Model,
		Stream:             stream,
		Tools:              req.Tools,
		ChatTemplateKwargs: map[string]any{"enable_thinking": false},
	}

	if stream {
		out.StreamOptions = &types.OpenAIStreamOptions{IncludeUsage: true}
	}

	applyOptions(&out, req.Options)
	applyThinkingPreference(&out, req.Think)
	applyResponseFormat(&out, req.Format)

	msgs, err := ConvertMessagesToOpenAI(req.Messages)
	if err != nil {
		return out, err
	}
	out.Messages = msgs

	return out, nil
}

func ConvertMessagesToOpenAI(msgs []types.OllamaMessage) ([]types.OpenAIMessage, error) {
	out := make([]types.OpenAIMessage, 0, len(msgs))
	toolCallIDs := map[string]string{}

	for _, msg := range msgs {
		openAIMessage := types.OpenAIMessage{Role: msg.Role}

		if len(msg.ToolCalls) > 0 {
			openAIMessage.ToolCalls = make([]types.OpenAIToolCall, len(msg.ToolCalls))
			for i, toolCall := range msg.ToolCalls {
				id := fmt.Sprintf("call_%d", time.Now().UnixNano()+int64(i))
				toolCallIDs[toolCall.Function.Name] = id
				openAIMessage.ToolCalls[i] = types.OpenAIToolCall{
					ID:   id,
					Type: "function",
					Function: types.OpenAIToolCallFunction{
						Name:      toolCall.Function.Name,
						Arguments: string(toolCall.Function.Arguments),
					},
				}
			}
		}

		if msg.Role == "tool" && msg.ToolName != "" {
			if id, ok := toolCallIDs[msg.ToolName]; ok {
				openAIMessage.ToolCallID = id
			} else {
				openAIMessage.ToolCallID = fmt.Sprintf("call_%s", msg.ToolName)
			}
		}

		content, err := marshalMessageContent(msg)
		if err != nil {
			return nil, err
		}
		openAIMessage.Content = content

		out = append(out, openAIMessage)
	}

	return out, nil
}

func DetectImageMIME(b64 string) string {
	data, err := base64.StdEncoding.DecodeString(b64[:min(len(b64), 16)])
	if err != nil || len(data) < 4 {
		return "image/jpeg"
	}

	switch {
	case data[0] == 0x89 && data[1] == 0x50:
		return "image/png"
	case data[0] == 0x47 && data[1] == 0x49:
		return "image/gif"
	case data[0] == 0x52 && data[1] == 0x49:
		return "image/webp"
	default:
		return "image/jpeg"
	}
}

func OllamaGenerateToOpenAI(req types.OllamaGenerateRequest) (types.OpenAIChatRequest, error) {
	chatReq := types.OllamaChatRequest{
		Model:  req.Model,
		Stream: req.Stream,
		Format: req.Format,
		Think:  req.Think,
		Options: types.OllamaOptions{
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
		chatReq.Messages = append(chatReq.Messages, types.OllamaMessage{Role: "system", Content: req.System})
	}

	chatReq.Messages = append(chatReq.Messages, types.OllamaMessage{
		Role:    "user",
		Content: req.Prompt,
		Images:  req.Images,
	})

	return OllamaChatToOpenAI(chatReq)
}

func applyOptions(out *types.OpenAIChatRequest, options types.OllamaOptions) {
	out.Temperature = options.Temperature
	out.TopP = options.TopP
	out.MinP = options.MinP
	out.Seed = options.Seed
	out.Stop = options.Stop
	out.FrequencyPenalty = options.FrequencyPenalty
	out.PresencePenalty = options.PresencePenalty
	out.TopK = options.TopK
	out.RepetitionPenalty = options.RepeatPenalty
	if options.NumPredict != nil {
		value := *options.NumPredict
		out.MaxTokens = &value
	}
}

func applyThinkingPreference(out *types.OpenAIChatRequest, think *bool) {
	if think != nil {
		out.ChatTemplateKwargs["enable_thinking"] = *think
	}
}

func applyResponseFormat(out *types.OpenAIChatRequest, format json.RawMessage) {
	if len(format) == 0 {
		return
	}

	var rawStr string
	if err := json.Unmarshal(format, &rawStr); err == nil {
		if rawStr == "json" {
			out.ResponseFormat = &types.OpenAIResponseFormat{Type: "json_object"}
		}
		return
	}

	out.ResponseFormat = &types.OpenAIResponseFormat{
		Type:       "json_schema",
		JSONSchema: format,
	}
}

func marshalMessageContent(msg types.OllamaMessage) (json.RawMessage, error) {
	if len(msg.Images) == 0 {
		return json.Marshal(msg.Content)
	}

	parts := make([]types.OpenAIContentPart, 0, len(msg.Images)+1)
	if msg.Content != "" {
		parts = append(parts, types.OpenAIContentPart{Type: "text", Text: msg.Content})
	}
	for _, image := range msg.Images {
		dataURL := image
		if !strings.HasPrefix(image, "data:") {
			dataURL = fmt.Sprintf("data:%s;base64,%s", DetectImageMIME(image), image)
		}
		parts = append(parts, types.OpenAIContentPart{
			Type:     "image_url",
			ImageURL: &types.OpenAIImageURL{URL: dataURL},
		})
	}

	return json.Marshal(parts)
}
