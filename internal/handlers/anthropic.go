package handlers

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	applogging "github.com/k0in/openai-ollama-proxy/internal/logging"
	"github.com/k0in/openai-ollama-proxy/internal/types"
)

func (server *Server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "failed to read body: "+err.Error())
		return
	}
	defer func() { _ = r.Body.Close() }()

	var anthropicReq types.AnthropicMessageRequest
	if err := json.Unmarshal(body, &anthropicReq); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "invalid json: "+err.Error())
		return
	}

	if server.cfg.Debug {
		log.Printf("anthropic messages request | model=%q stream=%t messages=%d system=%v",
			anthropicReq.Model, anthropicReq.Stream, len(anthropicReq.Messages), truncateForLog(string(anthropicReq.System), 100))
	}

	maxTokens := anthropicReq.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 8192
	}

	openAIReq, err := translateAnthropicToOpenAI(anthropicReq, maxTokens)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "translation error: "+err.Error())
		return
	}

	// Resolve route for the requested model.
	baseURL, apiKey, upstreamModel, _ := server.resolveRouteForModel(anthropicReq.Model)
	openAIReq.Model = upstreamModel

	openAIBody, err := json.Marshal(openAIReq)
	if err != nil {
		writeAnthropicError(w, http.StatusInternalServerError, "api_error", "marshal error: "+err.Error())
		return
	}

	if server.cfg.Debug {
		log.Printf(">>> UPSTREAM (anthropic->openai) POST %s/v1/chat/completions (%d bytes):\n  %s", baseURL, len(openAIBody), string(applogging.RedactJSONForLog(openAIBody)))
	}

	resp, err := server.doUpstreamChatWithRetryForRoute(r.Context(), openAIBody, baseURL, apiKey)
	if err != nil {
		log.Printf("anthropic upstream error: %v", err)
		writeAnthropicError(w, http.StatusServiceUnavailable, "api_error", "upstream not ready: "+err.Error())
		return
	}
	defer func() { _ = resp.Body.Close() }()

	if server.cfg.Debug {
		log.Printf("<<< UPSTREAM (anthropic) %d | content-type=%q", resp.StatusCode, resp.Header.Get("Content-Type"))
	}

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(resp.Body)
		log.Printf("anthropic upstream error %d: %s | sent: %s", resp.StatusCode, string(errBody), string(applogging.RedactJSONForLog(openAIBody)))
		writeAnthropicError(w, http.StatusBadGateway, "upstream_error", fmt.Sprintf("upstream error: %d", resp.StatusCode))
		return
	}

	responseModel := anthropicReq.Model
	if responseModel == "" {
		responseModel = server.firstUpstreamModel()
	}

	timings := newObservedTimings()
	timings.markResponseStart()

	if anthropicReq.Stream {
		server.handleAnthropicStream(w, resp.Body, responseModel, timings)
		return
	}

	server.handleAnthropicNonStream(w, resp.Body, responseModel, timings)
}

func (server *Server) handleAnthropicNonStream(w http.ResponseWriter, body io.Reader, model string, timings *observedTimings) {
	respBody, err := io.ReadAll(body)
	if err != nil {
		writeAnthropicError(w, http.StatusInternalServerError, "api_error", "read error: "+err.Error())
		return
	}
	timings.markResponseStart()

	var openAIResp types.OpenAIChatResponse
	if err := json.Unmarshal(respBody, &openAIResp); err != nil {
		writeAnthropicError(w, http.StatusInternalServerError, "api_error", "decode error: "+err.Error())
		return
	}
	timings.markComplete()

	if server.cfg.Debug {
		log.Printf("<<< UPSTREAM (anthropic) non-stream body (%d bytes): %s", len(respBody), string(respBody))
	}

	anthropicResp := convertOpenAIToAnthropic(openAIResp, model)

	if openAIResp.Usage != nil {
		server.stats.Record(model, openAIResp.Usage.PromptTokens, openAIResp.Usage.CompletionTokens, time.Duration(timings.evalDuration()))
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(anthropicResp)
}

func (server *Server) handleAnthropicStream(w http.ResponseWriter, body io.Reader, model string, timings *observedTimings) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeAnthropicError(w, http.StatusInternalServerError, "api_error", "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache, no-transform")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 256*1024), 10*1024*1024)

	messageID := fmt.Sprintf("msg_%s_%d", model, time.Now().UnixMilli())
	contentIndex := 0
	hasSentContentStart := false
	var accumulatedContent strings.Builder
	var promptTokens, completionTokens int

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var chunk types.OpenAIChatResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			log.Printf("anthropic stream decode error: %v, data: %s", err, data)
			continue
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]

		if !hasSentContentStart {
			msgStart := types.AnthropicMessageStartEvent{
				Type: "message_start",
				Message: types.AnthropicMessageResponse{
					ID:      messageID,
					Type:    "message",
					Role:    "assistant",
					Model:   model,
					Content: []types.AnthropicContentBlock{},
				},
			}
			writeAnthropicSSE(w, "message_start", msgStart)
			flusher.Flush()

			contentBlockStart := types.AnthropicContentBlockStartEvent{
				Type:  "content_block_start",
				Index: contentIndex,
				ContentBlock: types.AnthropicContentBlock{
					Type: "text",
					Text: "",
				},
			}
			writeAnthropicSSE(w, "content_block_start", contentBlockStart)
			flusher.Flush()

			hasSentContentStart = true
		}

		delta := choice.Delta
		if delta != nil && delta.Content != nil && *delta.Content != "" {
			timings.markFirstVisibleOutput()
			accumulatedContent.WriteString(*delta.Content)
			textDelta := types.AnthropicContentBlockDeltaEvent{
				Type:  "content_block_delta",
				Index: contentIndex,
				Delta: types.AnthropicTextDelta{
					Type: "text_delta",
					Text: *delta.Content,
				},
			}
			writeAnthropicSSE(w, "content_block_delta", textDelta)
			flusher.Flush()
		}

		if delta != nil && len(delta.ToolCalls) > 0 {
			timings.markFirstVisibleOutput()
			for _, tc := range delta.ToolCalls {
				if tc.Function.Name != "" {
					toolContentBlock := types.AnthropicContentBlockStartEvent{
						Type:  "content_block_start",
						Index: contentIndex + 1,
						ContentBlock: types.AnthropicContentBlock{
							Type:  "tool_use",
							ID:    tc.ID,
							Name:  tc.Function.Name,
							Input: json.RawMessage(tc.Function.Arguments),
						},
					}
					writeAnthropicSSE(w, "content_block_start", toolContentBlock)
					flusher.Flush()
					contentIndex++

					writeAnthropicSSE(w, "content_block_stop", types.AnthropicContentBlockStopEvent{
						Type: "content_block_stop", Index: contentIndex,
					})
					flusher.Flush()
				}
			}
		}

		if choice.FinishReason != nil || chunk.Usage != nil {
			timings.markComplete()

			if chunk.Usage != nil {
				promptTokens = chunk.Usage.PromptTokens
				completionTokens = chunk.Usage.CompletionTokens
			}

			stopReason := mapAnthropicStopReason(choice.FinishReason)

			writeAnthropicSSE(w, "content_block_stop", types.AnthropicContentBlockStopEvent{
				Type: "content_block_stop", Index: contentIndex,
			})
			flusher.Flush()

			msgDelta := types.AnthropicMessageDeltaEvent{
				Type: "message_delta",
				Delta: types.AnthropicMessageDelta{
					StopReason:   stopReason,
					StopSequence: nil,
				},
				Usage: &types.AnthropicUsage{
					InputTokens:  promptTokens,
					OutputTokens: completionTokens,
				},
			}
			writeAnthropicSSE(w, "message_delta", msgDelta)
			flusher.Flush()

			writeAnthropicSSE(w, "message_stop", types.AnthropicMessageStopEvent{
				Type: "message_stop",
			})
			flusher.Flush()

			server.stats.Record(model, promptTokens, completionTokens, time.Duration(timings.evalDuration()))

			return
		}
	}

	if err := scanner.Err(); err != nil {
		log.Printf("anthropic stream scanner error: %v", err)
	}

	if hasSentContentStart {
		timings.markComplete()
		writeAnthropicSSE(w, "content_block_stop", types.AnthropicContentBlockStopEvent{
			Type: "content_block_stop", Index: contentIndex,
		})
		flusher.Flush()

		msgDelta := types.AnthropicMessageDeltaEvent{
			Type: "message_delta",
			Delta: types.AnthropicMessageDelta{
				StopReason: "end_turn",
			},
		}
		writeAnthropicSSE(w, "message_delta", msgDelta)
		flusher.Flush()

		writeAnthropicSSE(w, "message_stop", types.AnthropicMessageStopEvent{
			Type: "message_stop",
		})
		flusher.Flush()
	}

	if completionTokens > 0 {
		server.stats.Record(model, promptTokens, completionTokens, time.Duration(timings.evalDuration()))
	}
}

func translateAnthropicToOpenAI(anthropicReq types.AnthropicMessageRequest, maxTokens int) (types.OpenAIChatRequest, error) {
	openAIReq := types.OpenAIChatRequest{
		Stream:    anthropicReq.Stream,
		MaxTokens: &maxTokens,
	}

	if anthropicReq.Temperature != nil {
		openAIReq.Temperature = anthropicReq.Temperature
	}
	if anthropicReq.TopP != nil {
		openAIReq.TopP = anthropicReq.TopP
	}
	if anthropicReq.TopK != nil {
		openAIReq.TopK = anthropicReq.TopK
	}
	if len(anthropicReq.StopSequences) > 0 {
		openAIReq.Stop = anthropicReq.StopSequences
	}
	if anthropicReq.Tools != nil {
		openAIReq.Tools = anthropicReq.Tools
	}

	var msgs []types.OpenAIMessage

	systemText := extractAnthropicSystemText(anthropicReq.System)
	if systemText != "" {
		systemContent, _ := json.Marshal(systemText)
		msgs = append(msgs, types.OpenAIMessage{
			Role:    "system",
			Content: systemContent,
		})
	}

	for _, msg := range anthropicReq.Messages {
		openAIMsg := translateAnthropicMsg(msg)
		msgs = append(msgs, openAIMsg)
	}

	openAIReq.Messages = msgs

	return openAIReq, nil
}

func translateAnthropicMsg(msg types.AnthropicMessage) types.OpenAIMessage {
	openAIMsg := types.OpenAIMessage{
		Role: msg.Role,
	}

	var contentStr string
	var contentBlocks []types.AnthropicContentBlock

	if err := json.Unmarshal(msg.Content, &contentStr); err == nil {
		contentJSON, _ := json.Marshal(contentStr)
		openAIMsg.Content = contentJSON
		return openAIMsg
	}

	if err := json.Unmarshal(msg.Content, &contentBlocks); err != nil {
		openAIMsg.Content = msg.Content
		return openAIMsg
	}

	var textParts []string
	var imageParts []string
	var toolCalls []types.OpenAIToolCall

	for _, block := range contentBlocks {
		switch block.Type {
		case "text":
			if block.Text != "" {
				textParts = append(textParts, block.Text)
			}
		case "image":
			if block.Source != nil {
				imageParts = append(imageParts, fmt.Sprintf("data:%s;base64,%s", block.Source.MediaType, block.Source.Data))
			}
		case "tool_use":
			toolCalls = append(toolCalls, types.OpenAIToolCall{
				ID:   block.ID,
				Type: "function",
				Function: types.OpenAIToolCallFunction{
					Name:      block.Name,
					Arguments: string(block.Input),
				},
			})
		case "tool_result":
			var resultContent string
			if err := json.Unmarshal(block.Content, &resultContent); err == nil {
				textParts = append(textParts, resultContent)
			} else {
				var resultBlocks []types.AnthropicContentBlock
				if err := json.Unmarshal(block.Content, &resultBlocks); err == nil {
					for _, rb := range resultBlocks {
						if rb.Type == "text" && rb.Text != "" {
							textParts = append(textParts, rb.Text)
						}
					}
				}
			}

			openAIMsg.Role = "tool"
			openAIMsg.ToolCallID = block.ToolUseID
		}
	}

	if len(toolCalls) > 0 {
		openAIMsg.ToolCalls = toolCalls
		if len(textParts) == 0 && len(imageParts) == 0 {
			contentJSON, _ := json.Marshal("")
			openAIMsg.Content = contentJSON
		}
	}

	if len(imageParts) > 0 {
		parts := make([]types.OpenAIContentPart, 0, len(textParts)+len(imageParts))
		for _, t := range textParts {
			parts = append(parts, types.OpenAIContentPart{Type: "text", Text: t})
		}
		for _, img := range imageParts {
			parts = append(parts, types.OpenAIContentPart{
				Type:     "image_url",
				ImageURL: &types.OpenAIImageURL{URL: img},
			})
		}
		contentJSON, _ := json.Marshal(parts)
		openAIMsg.Content = contentJSON
	} else {
		combined := strings.Join(textParts, "\n")
		contentJSON, _ := json.Marshal(combined)
		openAIMsg.Content = contentJSON
	}

	return openAIMsg
}

func convertOpenAIToAnthropic(openAIResp types.OpenAIChatResponse, model string) types.AnthropicMessageResponse {
	anthropicResp := types.AnthropicMessageResponse{
		ID:    fmt.Sprintf("msg_%s_%d", model, time.Now().UnixMilli()),
		Type:  "message",
		Role:  "assistant",
		Model: model,
	}

	if openAIResp.Usage != nil {
		anthropicResp.Usage = &types.AnthropicUsage{
			InputTokens:  openAIResp.Usage.PromptTokens,
			OutputTokens: openAIResp.Usage.CompletionTokens,
		}
	}

	if len(openAIResp.Choices) > 0 {
		choice := openAIResp.Choices[0]
		if choice.FinishReason != nil {
			anthropicResp.StopReason = mapAnthropicStopReason(choice.FinishReason)
		}

		if choice.Message != nil {
			contentBlocks := convertOpenAIMsgToAnthropicContent(choice.Message)
			anthropicResp.Content = contentBlocks
		}
	}

	if anthropicResp.Content == nil {
		anthropicResp.Content = []types.AnthropicContentBlock{}
	}

	return anthropicResp
}

func convertOpenAIMsgToAnthropicContent(msg *types.OpenAIRespMsg) []types.AnthropicContentBlock {
	var blocks []types.AnthropicContentBlock

	contentText := ""
	if msg.Content != nil {
		contentText = *msg.Content
	}

	reasoningText := ""
	if msg.ReasoningContent != nil {
		reasoningText = *msg.ReasoningContent
	} else if msg.Reasoning != nil {
		reasoningText = *msg.Reasoning
	}

	if reasoningText != "" {
		blocks = append(blocks, types.AnthropicContentBlock{
			Type: "thinking",
			Text: reasoningText,
		})
	}

	if contentText != "" {
		blocks = append(blocks, types.AnthropicContentBlock{
			Type: "text",
			Text: contentText,
		})
	}

	for _, tc := range msg.ToolCalls {
		blocks = append(blocks, types.AnthropicContentBlock{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: json.RawMessage(tc.Function.Arguments),
		})
	}

	if len(blocks) == 0 {
		blocks = append(blocks, types.AnthropicContentBlock{
			Type: "text",
			Text: "",
		})
	}

	return blocks
}

func mapAnthropicStopReason(finishReason *string) string {
	if finishReason == nil {
		return "end_turn"
	}
	switch *finishReason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	case "content_filter":
		return "content_filter"
	default:
		return "end_turn"
	}
}

func extractAnthropicSystemText(system json.RawMessage) string {
	if len(system) == 0 || string(system) == "null" {
		return ""
	}

	var text string
	if err := json.Unmarshal(system, &text); err == nil {
		return text
	}

	var blocks []types.AnthropicContentBlock
	if err := json.Unmarshal(system, &blocks); err == nil {
		var parts []string
		for _, block := range blocks {
			if block.Type == "text" && block.Text != "" {
				parts = append(parts, block.Text)
			}
		}
		return strings.Join(parts, "\n")
	}

	return ""
}

func writeAnthropicError(w http.ResponseWriter, statusCode int, errType, message string) {
	resp := types.AnthropicErrorResponse{
		Type: "error",
		Error: types.AnthropicError{
			Type:    errType,
			Message: message,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	_ = json.NewEncoder(w).Encode(resp)
}

func writeAnthropicSSE(w http.ResponseWriter, eventType string, data any) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		log.Printf("anthropic SSE marshal error: %v", err)
		return
	}

	_, _ = fmt.Fprintf(w, "event: %s\n", eventType)
	_, _ = fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
}
