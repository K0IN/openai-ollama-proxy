package translate

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/k0in/openai-ollama-proxy/internal/types"
)

func isEmptyJSON(raw json.RawMessage) bool {
	trimmed := strings.TrimSpace(string(raw))
	return trimmed == "[]" || trimmed == "{}"
}

func OllamaChatToOpenAI(req types.OllamaChatRequest) (types.OpenAIChatRequest, error) {
	stream := true
	if req.Stream != nil {
		stream = *req.Stream
	}

	out := types.OpenAIChatRequest{
		Model:  req.Model,
		Stream: stream,
	}

	// Only set Tools if it's not empty to avoid upstream validation errors
	// that reject empty tools arrays.
	if len(req.Tools) > 0 && !isEmptyJSON(req.Tools) {
		out.Tools = req.Tools
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
	// Decode up to 512 bytes' worth of header data. http.DetectContentType
	// uses at most the first 512 bytes; base64 inflates by 4/3, so 700 chars
	// always covers the full sniff window even with padding/whitespace.
	b64Header := b64
	if len(b64Header) > 700 {
		b64Header = b64Header[:700]
	}
	// Trim trailing partial base64 quartet so DecodeString does not fail on
	// otherwise-valid prefixes.
	b64Header = b64Header[:len(b64Header)-(len(b64Header)%4)]

	data, err := base64.StdEncoding.DecodeString(b64Header)
	if err != nil || len(data) < 4 {
		return "image/jpeg"
	}

	// Containers that http.DetectContentType either misclassifies or cannot
	// distinguish without inspecting box types.
	if mime := detectISOBMFFImageMIME(data); mime != "" {
		return mime
	}
	if mime := detectRIFFImageMIME(data); mime != "" {
		return mime
	}

	if sniffed := http.DetectContentType(data); strings.HasPrefix(sniffed, "image/") {
		return sniffed
	}

	return "image/jpeg"
}

func detectISOBMFFImageMIME(data []byte) string {
	if len(data) < 12 {
		return ""
	}
	if string(data[4:8]) != "ftyp" {
		return ""
	}

	major := string(data[8:12])
	switch major {
	case "avif", "avis":
		return "image/avif"
	case "heic", "heix", "hevc", "hevx":
		return "image/heic"
	case "mif1", "msf1", "heim", "heis":
		return "image/heif"
	}

	// Walk compatible brands (4 bytes each) starting at offset 16.
	for i := 16; i+4 <= len(data) && i < 64; i += 4 {
		switch string(data[i : i+4]) {
		case "avif", "avis":
			return "image/avif"
		case "heic", "heix", "hevc", "hevx":
			return "image/heic"
		case "mif1", "msf1":
			return "image/heif"
		}
	}
	return ""
}

func detectRIFFImageMIME(data []byte) string {
	if len(data) < 12 {
		return ""
	}
	if string(data[0:4]) != "RIFF" {
		return ""
	}
	if string(data[8:12]) == "WEBP" {
		return "image/webp"
	}
	return ""
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
		if out.ChatTemplateKwargs == nil {
			out.ChatTemplateKwargs = map[string]any{}
		}
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
