package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"
)

func (server *Server) rewriteRequestModel(body []byte) ([]byte, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}
	// Resolve route based on the requested model.
	model, _ := payload["model"].(string)
	_, _, upstreamModel, _ := server.resolveRouteForModel(model)
	payload["model"] = upstreamModel
	return json.Marshal(payload)
}

func (server *Server) rewriteRequestForChat(body []byte) ([]byte, bool, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, false, err
	}
	// Resolve route based on the requested model.
	model, _ := payload["model"].(string)
	_, _, upstreamModel, _ := server.resolveRouteForModel(model)
	payload["model"] = upstreamModel

	normalized, err := json.Marshal(payload)
	if err != nil {
		return nil, false, err
	}

	return normalized, false, nil
}

func requestDebugSummary(body []byte) string {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return fmt.Sprintf("bytes=%d invalid-json=%v", len(body), err)
	}

	model, _ := payload["model"].(string)
	stream, _ := payload["stream"].(bool)
	messageCount := 0
	if messages, ok := payload["messages"].([]any); ok {
		messageCount = len(messages)
	}
	toolsCount := 0
	if tools, ok := payload["tools"].([]any); ok {
		toolsCount = len(tools)
	}
	toolChoice := ""
	if value, ok := payload["tool_choice"]; ok {
		toolChoice = fmt.Sprintf("%v", value)
	}
	hasStreamOptions := payload["stream_options"] != nil

	return fmt.Sprintf("model=%q stream=%t messages=%d tools=%d toolChoice=%q streamOptions=%t bytes=%d", model, stream, messageCount, toolsCount, toolChoice, hasStreamOptions, len(body))
}

func truncateForLog(value string, max int) string {
	if len(value) <= max {
		return value
	}
	return value[:max] + " ...(truncated)"
}

func normalizeOpenAIMessageMap(message map[string]any) {
	content, contentIsString := message["content"].(string)
	reasoningContent, _ := message["reasoning_content"].(string)
	reasoning, _ := message["reasoning"].(string)
	// Only overwrite empty-string content with reasoning when content is a
	// plain string.  Multi-modal content (images / audio / video) is an array
	// and must never be replaced by reasoning text.
	if contentIsString && content == "" {
		switch {
		case reasoningContent != "":
			message["content"] = reasoningContent
		case reasoning != "":
			message["content"] = reasoning
		}
	}

	if toolCalls, ok := message["tool_calls"].([]any); ok && len(toolCalls) == 0 {
		delete(message, "tool_calls")
	}

	delete(message, "reasoning_content")
	delete(message, "reasoning")
}

func normalizeOpenAIChoiceMap(choice map[string]any) {
	delete(choice, "token_ids")
	delete(choice, "stop_reason")

	if message, ok := choice["message"].(map[string]any); ok {
		normalizeOpenAIMessageMap(message)
	}
	if delta, ok := choice["delta"].(map[string]any); ok {
		normalizeOpenAIMessageMap(delta)
	}
}

func (server *Server) normalizeOpenAIJSON(payload []byte) ([]byte, error) {
	var value map[string]any
	if err := json.Unmarshal(payload, &value); err != nil {
		return nil, err
	}

	if _, ok := value["model"]; ok {
		if localModel := server.firstUpstreamModel(); localModel != "" {
			value["model"] = localModel
		}
	}

	delete(value, "prompt_token_ids")

	if choices, ok := value["choices"].([]any); ok {
		for _, rawChoice := range choices {
			choice, ok := rawChoice.(map[string]any)
			if !ok {
				continue
			}
			normalizeOpenAIChoiceMap(choice)
		}
	}

	return json.Marshal(value)
}

func (server *Server) normalizeOpenAIStreamLine(line string) string {
	if !strings.HasPrefix(line, "data: ") {
		return line
	}

	data := strings.TrimPrefix(line, "data: ")
	if data == "" || data == "[DONE]" {
		return line
	}

	normalized, err := server.normalizeOpenAIJSON([]byte(data))
	if err != nil {
		if server.cfg.Debug {
			log.Printf("openai chat normalize skipped invalid chunk: %v | line=%s", err, truncateForLog(line, 200))
		}
		return line
	}

	return "data: " + string(normalized)
}
