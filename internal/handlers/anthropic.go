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

// handleAnthropicMessages handles the Anthropic Messages API endpoint (/v1/messages).
// It translates Anthropic requests to OpenAI chat completions, forwards them upstream,
// and translates the response back to Anthropic format.
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

	// Default max_tokens if not set
	maxTokens := anthropicReq.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 8192
	}

	// Translate Anthropic request to OpenAI chat request
	openAIReq, err := translateAnthropicToOpenAI(anthropicReq, maxTokens)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "translation error: "+err.Error())
		return
	}
	openAIReq.Model = server.cfg.UpstreamModel

	openAIBody, err := json.Marshal(openAIReq)
	if err != nil {
		writeAnthropicError(w, http.StatusInternalServerError, "api_error", "marshal error: "+err.Error())
		return
	}

	if server.cfg.Debug {
		log.Printf(">>> UPSTREAM (anthropic->openai) POST %s/v1/chat/completions (%d bytes):\n  %s", server.cfg.UpstreamBaseURL, len(openAIBody), string(applogging.RedactJSONForLog(openAIBody)))
	}

	resp, err := server.doUpstreamChatWithRetry(r.Context(), openAIBody)
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

	// Determine model name to use in response
	responseModel := anthropicReq.Model
	if responseModel == "" {
		responseModel = server.cfg.ModelName
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

	// Record token stats
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

	// Anthropic SSE streaming protocol:
	// event: message_start
	// data: {...}\n\n
	// event: content_block_start
	// data: {...}\n\n
	// event: content_block_delta
	// data: {...}\n\n
	// event: content_block_stop
	// data: {...}\n\n
	// event: message_delta
	// data: {...}\n\n
	// event: message_stop
	// data: {...}\n\n

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

		// Send message_start on first chunk
		if !hasSentContentStart {
			// Start with the message
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

			// Send content_block_start
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

		// Handle content/text delta
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

		// Handle tool calls from delta
		if delta != nil && len(delta.ToolCalls) > 0 {
			timings.markFirstVisibleOutput()
			for _, tc := range delta.ToolCalls {
				if tc.Function.Name != "" {
					// This is a tool_use content block start
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

					// Send content_block_stop for the tool block
					writeAnthropicSSE(w, "content_block_stop", types.AnthropicContentBlockStopEvent{
						Type: "content_block_stop", Index: contentIndex,
					})
					flusher.Flush()
				}
			}
		}

		// Handle finish (message_delta + message_stop)
		if choice.FinishReason != nil || chunk.Usage != nil {
			timings.markComplete()

			if chunk.Usage != nil {
				promptTokens = chunk.Usage.PromptTokens
				completionTokens = chunk.Usage.CompletionTokens
			}

			stopReason := mapAnthropicStopReason(choice.FinishReason)

			// Ensure content_block_stop is sent before message_delta
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

			// Record stats with timing
			server.stats.Record(model, promptTokens, completionTokens, time.Duration(timings.evalDuration()))

			return
		}
	}

	if err := scanner.Err(); err != nil {
		log.Printf("anthropic stream scanner error: %v", err)
	}

	// If we never got a finish reason, send cleanup events
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

	// Record stats even for truncated stream (in case usage was seen)
	if completionTokens > 0 {
		server.stats.Record(model, promptTokens, completionTokens, time.Duration(timings.evalDuration()))
	}
}

// translateAnthropicToOpenAI converts an Anthropic Messages API request to
// an OpenAI chat completions request.
func translateAnthropicToOpenAI(anthropicReq types.AnthropicMessageRequest, maxTokens int) (types.OpenAIChatRequest, error) {
	openAIReq := types.OpenAIChatRequest{
		Stream:    anthropicReq.Stream,
		MaxTokens: &maxTokens,
	}

	// Translate temperature, top_p, top_k
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

	// Translate messages
	var msgs []types.OpenAIMessage

	// Handle system prompt (can be a string or array of content blocks)
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

// translateAnthropicMsg converts a single Anthropic message to an OpenAI message.
func translateAnthropicMsg(msg types.AnthropicMessage) types.OpenAIMessage {
	openAIMsg := types.OpenAIMessage{
		Role: msg.Role,
	}

	// Parse the content - can be a string or []AnthropicContentBlock
	var contentStr string
	var contentBlocks []types.AnthropicContentBlock

	// Try to unmarshal as a string first
	if err := json.Unmarshal(msg.Content, &contentStr); err == nil {
		contentJSON, _ := json.Marshal(contentStr)
		openAIMsg.Content = contentJSON
		return openAIMsg
	}

	// Try as array of content blocks
	if err := json.Unmarshal(msg.Content, &contentBlocks); err != nil {
		// Fallback: pass through as-is
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
			// tool_result maps to a "tool" role message in OpenAI.
			// We need to extract the content from the tool result.
			var resultContent string
			if err := json.Unmarshal(block.Content, &resultContent); err == nil {
				textParts = append(textParts, resultContent)
			} else {
				// Try as array of content blocks
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
		// When there are tool calls, content must be non-null
		if len(textParts) == 0 && len(imageParts) == 0 {
			contentJSON, _ := json.Marshal("")
			openAIMsg.Content = contentJSON
		}
	}

	// Build content
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

// convertOpenAIToAnthropic converts an OpenAI chat response to Anthropic format.
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

// convertOpenAIMsgToAnthropicContent converts an OpenAI response message to
// Anthropic content blocks.
func convertOpenAIMsgToAnthropicContent(msg *types.OpenAIRespMsg) []types.AnthropicContentBlock {
	var blocks []types.AnthropicContentBlock

	// Extract text content
	contentText := ""
	if msg.Content != nil {
		contentText = *msg.Content
	}

	// Extract reasoning content (maps to thinking in Anthropic)
	reasoningText := ""
	if msg.ReasoningContent != nil {
		reasoningText = *msg.ReasoningContent
	} else if msg.Reasoning != nil {
		reasoningText = *msg.Reasoning
	}

	// Add thinking block if reasoning content exists
	if reasoningText != "" {
		blocks = append(blocks, types.AnthropicContentBlock{
			Type: "thinking",
			Text: reasoningText,
		})
	}

	// Add text content block
	if contentText != "" {
		blocks = append(blocks, types.AnthropicContentBlock{
			Type: "text",
			Text: contentText,
		})
	}

	// Add tool_use blocks for tool calls
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

// mapAnthropicStopReason converts OpenAI finish reasons to Anthropic stop reasons.
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

// extractAnthropicSystemText extracts the text from the Anthropic system field,
// which can be a plain string or an array of content blocks (newer API format).
func extractAnthropicSystemText(system json.RawMessage) string {
	if len(system) == 0 || string(system) == "null" {
		return ""
	}

	// Try as plain string first
	var text string
	if err := json.Unmarshal(system, &text); err == nil {
		return text
	}

	// Try as array of content blocks
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

// writeAnthropicError writes an Anthropic-compatible error response.
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

// writeAnthropicSSE writes an Anthropic-style SSE event (event: + data:).
func writeAnthropicSSE(w http.ResponseWriter, eventType string, data any) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		log.Printf("anthropic SSE marshal error: %v", err)
		return
	}

	_, _ = fmt.Fprintf(w, "event: %s\n", eventType)
	_, _ = fmt.Fprintf(w, "data: %s\n\n", string(jsonData))
}
