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
	baseURL, apiKey, upstreamModel, _, passthrough := server.resolveRouteForModelPassthrough(anthropicReq.Model)
	apiKey = server.resolveEffectiveAPIKey(apiKey, passthrough, applogging.ExtractAPIKey(r))
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

	st := &anthropicStreamState{
		writer:    w,
		flusher:   flusher,
		messageID: fmt.Sprintf("msg_%s_%d", model, time.Now().UnixMilli()),
		model:     model,
	}

	var finishReason *string
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

		if chunk.Usage != nil {
			promptTokens = chunk.Usage.PromptTokens
			completionTokens = chunk.Usage.CompletionTokens
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]
		st.emitMessageStart()

		delta := choice.Delta
		if delta != nil {
			reasoning := ""
			if delta.ReasoningContent != nil {
				reasoning = *delta.ReasoningContent
			} else if delta.Reasoning != nil {
				reasoning = *delta.Reasoning
			}
			if reasoning != "" {
				timings.markFirstVisibleOutput()
				st.emitThinkingDelta(reasoning)
			}

			if delta.Content != nil && *delta.Content != "" {
				timings.markFirstVisibleOutput()
				st.emitTextDelta(*delta.Content)
			}

			if len(delta.ToolCalls) > 0 {
				timings.markFirstVisibleOutput()
				for _, tc := range delta.ToolCalls {
					st.emitToolCallDelta(tc)
				}
			}
		}

		if choice.FinishReason != nil {
			finishReason = choice.FinishReason
		}
	}

	if err := scanner.Err(); err != nil {
		log.Printf("anthropic stream scanner error: %v", err)
	}

	timings.markComplete()

	// Ensure the stream is well-formed even when upstream sent nothing.
	st.emitMessageStart()
	st.closeOpenBlock()

	stopReason := "end_turn"
	if finishReason != nil {
		stopReason = mapAnthropicStopReason(finishReason)
	}

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

	writeAnthropicSSE(w, "message_stop", types.AnthropicMessageStopEvent{Type: "message_stop"})
	flusher.Flush()

	if completionTokens > 0 || promptTokens > 0 {
		server.stats.Record(model, promptTokens, completionTokens, time.Duration(timings.evalDuration()))
	}
}

// anthropicStreamState tracks the open content block while translating an
// OpenAI SSE stream into Anthropic Messages streaming events. Anthropic
// requires content blocks to be opened, delta'd, and closed in sequential
// index order, so only one block is open at a time.
type anthropicStreamState struct {
	writer    http.ResponseWriter
	flusher   http.Flusher
	messageID string
	model     string

	startedMessage bool
	nextIndex      int

	// open block tracking. blockKind is "", "text", "thinking", or "tool".
	blockKind     string
	blockIndex    int
	toolCallIndex int // OpenAI tool-call index of the currently open tool block
}

func (st *anthropicStreamState) emitMessageStart() {
	if st.startedMessage {
		return
	}
	st.startedMessage = true
	writeAnthropicSSE(st.writer, "message_start", types.AnthropicMessageStartEvent{
		Type: "message_start",
		Message: types.AnthropicMessageResponse{
			ID:      st.messageID,
			Type:    "message",
			Role:    "assistant",
			Model:   st.model,
			Content: []types.AnthropicContentBlock{},
		},
	})
	st.flusher.Flush()
}

func (st *anthropicStreamState) closeOpenBlock() {
	if st.blockKind == "" {
		return
	}
	writeAnthropicSSE(st.writer, "content_block_stop", types.AnthropicContentBlockStopEvent{
		Type:  "content_block_stop",
		Index: st.blockIndex,
	})
	st.flusher.Flush()
	st.blockKind = ""
}

func (st *anthropicStreamState) openBlock(kind string, block types.AnthropicContentBlock) {
	st.blockKind = kind
	st.blockIndex = st.nextIndex
	st.nextIndex++
	writeAnthropicSSE(st.writer, "content_block_start", types.AnthropicContentBlockStartEvent{
		Type:         "content_block_start",
		Index:        st.blockIndex,
		ContentBlock: block,
	})
	st.flusher.Flush()
}

func (st *anthropicStreamState) emitThinkingDelta(text string) {
	if st.blockKind != "thinking" {
		st.closeOpenBlock()
		st.openBlock("thinking", types.AnthropicContentBlock{Type: "thinking"})
	}
	writeAnthropicSSE(st.writer, "content_block_delta", types.AnthropicThinkingDeltaEvent{
		Type:  "content_block_delta",
		Index: st.blockIndex,
		Delta: types.AnthropicThinkingDelta{Type: "thinking_delta", Thinking: text},
	})
	st.flusher.Flush()
}

func (st *anthropicStreamState) emitTextDelta(text string) {
	if st.blockKind != "text" {
		st.closeOpenBlock()
		st.openBlock("text", types.AnthropicContentBlock{Type: "text", Text: ""})
	}
	writeAnthropicSSE(st.writer, "content_block_delta", types.AnthropicContentBlockDeltaEvent{
		Type:  "content_block_delta",
		Index: st.blockIndex,
		Delta: types.AnthropicTextDelta{Type: "text_delta", Text: text},
	})
	st.flusher.Flush()
}

func (st *anthropicStreamState) emitToolCallDelta(tc types.OpenAIToolCall) {
	tcIndex := 0
	if tc.Index != nil {
		tcIndex = *tc.Index
	}

	// Open a new tool_use block when this is a different tool call than the
	// one currently open (new index, or a fresh call signalled by a name).
	needNewBlock := st.blockKind != "tool" || st.toolCallIndex != tcIndex
	if needNewBlock {
		st.closeOpenBlock()
		st.openBlock("tool", types.AnthropicContentBlock{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: json.RawMessage(`{}`),
		})
		st.toolCallIndex = tcIndex
	}

	if tc.Function.Arguments != "" {
		writeAnthropicSSE(st.writer, "content_block_delta", types.AnthropicInputJSONDeltaEvent{
			Type:  "content_block_delta",
			Index: st.blockIndex,
			Delta: types.AnthropicInputJSONDelta{Type: "input_json_delta", PartialJSON: tc.Function.Arguments},
		})
		st.flusher.Flush()
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
	if tools, err := convertAnthropicTools(anthropicReq.Tools); err != nil {
		return openAIReq, err
	} else if tools != nil {
		openAIReq.Tools = tools
	}
	if choice := convertAnthropicToolChoice(anthropicReq.ToolChoice); choice != nil {
		openAIReq.ToolChoice = choice
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

// convertAnthropicTools translates the Anthropic `tools` array
// (`{name, description, input_schema}`) into the OpenAI function-tool schema
// (`{type:"function", function:{name, description, parameters}}`). It returns
// nil when there are no tools so the field is omitted from the upstream
// request (some backends reject empty tool arrays).
func convertAnthropicTools(raw json.RawMessage) (json.RawMessage, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}

	var anthropicTools []types.AnthropicTool
	if err := json.Unmarshal(raw, &anthropicTools); err != nil {
		return nil, fmt.Errorf("invalid tools: %w", err)
	}
	if len(anthropicTools) == 0 {
		return nil, nil
	}

	type openAIFunction struct {
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters,omitempty"`
	}
	type openAITool struct {
		Type     string         `json:"type"`
		Function openAIFunction `json:"function"`
	}

	openAITools := make([]openAITool, 0, len(anthropicTools))
	for _, tool := range anthropicTools {
		if tool.Name == "" {
			continue
		}
		params := tool.InputSchema
		if len(params) == 0 || string(params) == "null" {
			params = json.RawMessage(`{"type":"object"}`)
		}
		openAITools = append(openAITools, openAITool{
			Type: "function",
			Function: openAIFunction{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  params,
			},
		})
	}

	if len(openAITools) == 0 {
		return nil, nil
	}

	return json.Marshal(openAITools)
}

// convertAnthropicToolChoice maps Anthropic `tool_choice` to its OpenAI
// equivalent. Anthropic uses {"type":"auto"|"any"|"tool", "name":"..."}.
// OpenAI uses "auto"/"required"/"none" strings or
// {"type":"function","function":{"name":"..."}}.
func convertAnthropicToolChoice(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}

	var choice types.AnthropicToolChoice
	if err := json.Unmarshal(raw, &choice); err != nil {
		return nil
	}

	switch choice.Type {
	case "auto":
		return json.RawMessage(`"auto"`)
	case "any":
		return json.RawMessage(`"required"`)
	case "none":
		return json.RawMessage(`"none"`)
	case "tool":
		if choice.Name == "" {
			return json.RawMessage(`"auto"`)
		}
		out, err := json.Marshal(map[string]any{
			"type":     "function",
			"function": map[string]string{"name": choice.Name},
		})
		if err != nil {
			return nil
		}
		return out
	default:
		return nil
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
