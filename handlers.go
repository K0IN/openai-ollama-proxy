package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

type modelMetadata struct {
	ContextLength int
	Family        string
	ParentModel   string
	Format        string
	ParameterSize string
	Quantization  string
}

func currentModelMetadata(ctx context.Context) modelMetadata {
	metadata := fallbackModelMetadata()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, cfg.VLLMBaseURL+"/v1/models", nil)
	if err != nil {
		return metadata
	}
	if cfg.VLLMAPIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.VLLMAPIKey)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return metadata
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return metadata
	}

	var list OpenAIModelListResponse
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		return metadata
	}

	for _, model := range list.Data {
		if model.ID != cfg.VLLMModel && model.Root != cfg.VLLMModel {
			continue
		}
		if model.MaxModelLen > 0 {
			metadata.ContextLength = model.MaxModelLen
		}
		if model.Root != "" {
			metadata.ParentModel = model.Root
			applyModelNameHints(&metadata, model.Root)
		} else {
			applyModelNameHints(&metadata, model.ID)
		}
		return metadata
	}

	return metadata
}

func fallbackModelMetadata() modelMetadata {
	metadata := modelMetadata{
		ContextLength: cfg.ModelContextLength,
		Family:        "transformer",
		ParentModel:   cfg.VLLMModel,
		Format:        "unknown",
		ParameterSize: "unknown",
		Quantization:  "unknown",
	}
	applyModelNameHints(&metadata, cfg.VLLMModel)
	return metadata
}

func applyModelNameHints(metadata *modelMetadata, name string) {
	lower := strings.ToLower(name)
	if strings.Contains(lower, "qwen") {
		metadata.Family = "qwen3"
	}
	if strings.Contains(lower, "awq") {
		metadata.Quantization = "AWQ-4bit"
	}
	if strings.Contains(lower, "fp8") {
		metadata.Quantization = "FP8"
	}
	if strings.Contains(lower, "nvfp4") {
		metadata.Quantization = "NVFP4"
	}
	if strings.Contains(name, "35B") {
		metadata.ParameterSize = "35B"
	}
	if strings.Contains(lower, "safetensors") || strings.Contains(lower, "awq") || strings.Contains(lower, "fp8") || strings.Contains(lower, "nvfp4") {
		metadata.Format = "safetensors"
	}
}

func ollamaModelInfo(metadata modelMetadata) map[string]any {
	info := map[string]any{
		"general.architecture":         metadata.Family,
		"general.parameter_count":      metadata.ParameterSize,
		"general.quantization_version": metadata.Quantization,
	}
	if metadata.ContextLength > 0 {
		info[metadata.Family+".context_length"] = metadata.ContextLength
		info["general.context_length"] = metadata.ContextLength
	}
	return info
}

func probeVLLMHealth(ctx context.Context) (bool, error) {
	healthCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()

	upstream, err := http.NewRequestWithContext(healthCtx, http.MethodGet,
		strings.TrimRight(cfg.VLLMBaseURL, "/")+"/health", nil)
	if err != nil {
		return false, err
	}
	if cfg.VLLMAPIKey != "" {
		upstream.Header.Set("Authorization", "Bearer "+cfg.VLLMAPIKey)
	}

	resp, err := httpClient.Do(upstream)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)

	return resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices, nil
}

func doUpstreamChatWithRetry(ctx context.Context, payload []byte) (*http.Response, error) {
	deadline := time.Now().Add(cfg.UpstreamStartupWait)
	if cfg.UpstreamRetryInterval <= 0 {
		cfg.UpstreamRetryInterval = 2 * time.Second
	}

	var lastErr error

	for {
		upstream, err := http.NewRequestWithContext(ctx, http.MethodPost,
			cfg.VLLMBaseURL+"/v1/chat/completions", bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		upstream.Header.Set("Content-Type", "application/json")
		if cfg.VLLMAPIKey != "" {
			upstream.Header.Set("Authorization", "Bearer "+cfg.VLLMAPIKey)
		}

		resp, err := httpClient.Do(upstream)
		if err == nil {
			if resp.StatusCode != http.StatusServiceUnavailable {
				return resp, nil
			}

			errBody, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			lastErr = fmt.Errorf("upstream returned 503: %s", strings.TrimSpace(string(errBody)))
		} else {
			lastErr = err
		}

		if time.Now().After(deadline) {
			break
		}

		select {
		case <-ctx.Done():
			return nil, errors.New("request canceled")
		case <-time.After(cfg.UpstreamRetryInterval):
		}
	}

	if lastErr == nil {
		lastErr = errors.New("upstream not ready")
	}

	return nil, fmt.Errorf("vLLM unavailable after %s: %w", cfg.UpstreamStartupWait, lastErr)
}

func handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	var olReq OllamaChatRequest
	if err := json.Unmarshal(body, &olReq); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Handle load/unload: empty messages array means load or unload the model.
	if len(olReq.Messages) == 0 {
		doneReason := "load"
		if isZeroKeepAlive(olReq.KeepAlive) {
			doneReason = "unload"
		}
		resp := OllamaChatResponse{
			Model:      olReq.Model,
			CreatedAt:  time.Now().UTC().Format(time.RFC3339Nano),
			Done:       true,
			DoneReason: doneReason,
			Message:    OllamaMessage{Role: "assistant"},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
		return
	}

	oaiReq, err := ollamaChatToOpenAI(olReq)
	if err != nil {
		http.Error(w, "translation error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	oaiBody, err := json.Marshal(oaiReq)
	if err != nil {
		http.Error(w, "marshal error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	if cfg.Debug {
		log.Printf(">>> UPSTREAM POST %s/v1/chat/completions (%d bytes):\n  %s", cfg.VLLMBaseURL, len(oaiBody), string(oaiBody))
	}

	resp, err := doUpstreamChatWithRetry(r.Context(), oaiBody)
	if err != nil {
		log.Printf("upstream chat error: %v", err)
		http.Error(w, "upstream not ready: "+err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	if cfg.Debug {
		log.Printf("<<< UPSTREAM %d | content-type=%q", resp.StatusCode, resp.Header.Get("Content-Type"))
	}

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(resp.Body)
		log.Printf("upstream chat error %d: %s | sent: %s", resp.StatusCode, string(errBody), string(oaiBody))
		http.Error(w, fmt.Sprintf("upstream error: %d", resp.StatusCode), resp.StatusCode)
		return
	}

	ollamaModel := olReq.Model
	stream := true
	if olReq.Stream != nil {
		stream = *olReq.Stream
	}

	if stream {
		handleChatStream(w, resp.Body, ollamaModel)
	} else {
		handleChatNonStream(w, resp.Body, ollamaModel)
	}
}

func handleChatNonStream(w http.ResponseWriter, body io.Reader, model string) {
	rawBody, err := io.ReadAll(body)
	if err != nil {
		http.Error(w, "read error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	if cfg.Debug {
		log.Printf("<<< UPSTREAM BODY (non-stream, %d bytes): %s", len(rawBody), string(rawBody))
	}

	var oaiResp OpenAIChatResponse
	if err := json.Unmarshal(rawBody, &oaiResp); err != nil {
		http.Error(w, "decode error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	olResp := openAIChatToOllama(oaiResp, model)

	if cfg.Debug {
		out, _ := json.Marshal(olResp)
		log.Printf("<<< OLLAMA RESPONSE: %s", string(out))
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(olResp)
}

func handleChatStream(w http.ResponseWriter, body io.Reader, model string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	scanner := bufio.NewScanner(body)
	// SSE lines can be large with tool calls / big content
	scanner.Buffer(make([]byte, 0, 256*1024), 1024*1024)
	pendingDoneReason := ""
	sentFinal := false
	chunkIdx := 0
	var streamedToolCalls []streamedToolCallState

	for scanner.Scan() {
		line := scanner.Text()
		if cfg.Debug {
			log.Printf("  STREAM RAW LINE [%d]: %s", chunkIdx, line)
		}
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			if cfg.Debug {
				log.Printf("  STREAM [DONE] received")
			}
			break
		}

		var chunk OpenAIChatResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			log.Printf("stream decode error: %v, data: %s", err, data)
			continue
		}

		toolCallDeltas := chunkToolCalls(chunk)
		if len(toolCallDeltas) > 0 {
			streamedToolCalls = appendStreamedToolCalls(streamedToolCalls, toolCallDeltas)
			if cfg.Debug {
				log.Printf("  STREAM TOOL CALL DELTA [%d]: parts=%d accumulated=%d", chunkIdx, len(toolCallDeltas), len(streamedToolCalls))
			}
			chunkIdx++
			continue
		}

		if chunkFinishReason(chunk) == "tool_calls" && len(streamedToolCalls) > 0 {
			toolChunk, ok := buildOllamaStreamToolCallChunk(streamedToolCalls, model)
			streamedToolCalls = nil
			if ok {
				out, err := json.Marshal(toolChunk)
				if err != nil {
					log.Printf("stream tool-call encode error: %v", err)
				} else {
					if cfg.Debug {
						log.Printf("  STREAM EMIT TOOL [%d]: %s", chunkIdx, string(out))
					}
					w.Write(out)
					w.Write([]byte("\n"))
					flusher.Flush()
					chunkIdx++
				}
			}
		}

		olChunk := openAIStreamChunkToOllama(chunk, model)

		if cfg.Debug {
			olOut, _ := json.Marshal(olChunk)
			isEmpty := isEmptyStreamChunk(olChunk)
			noUsage := respChunkHasNoUsage(chunk)
			log.Printf("  STREAM CHUNK [%d]: done=%t doneReason=%q empty=%t noUsage=%t content=%q thinking=%q toolCalls=%d -> %s",
				chunkIdx, olChunk.Done, olChunk.DoneReason, isEmpty, noUsage,
				olChunk.Message.Content, olChunk.Message.Thinking, len(olChunk.Message.ToolCalls),
				string(olOut))
		}

		if olChunk.DoneReason != "" && isEmptyStreamChunk(olChunk) && respChunkHasNoUsage(chunk) {
			pendingDoneReason = olChunk.DoneReason
			if cfg.Debug {
				log.Printf("  STREAM CHUNK [%d]: deferred done_reason=%q", chunkIdx, pendingDoneReason)
			}
			chunkIdx++
			continue
		}
		if olChunk.Done && olChunk.DoneReason == "" && pendingDoneReason != "" {
			olChunk.DoneReason = pendingDoneReason
		}
		if isEmptyStreamChunk(olChunk) && !olChunk.Done {
			if cfg.Debug {
				log.Printf("  STREAM CHUNK [%d]: skipped (empty, not done)", chunkIdx)
			}
			chunkIdx++
			continue
		}
		if olChunk.Done {
			sentFinal = true
		}
		out, err := json.Marshal(olChunk)
		if err != nil {
			log.Printf("stream encode error: %v", err)
			chunkIdx++
			continue
		}
		if cfg.Debug {
			log.Printf("  STREAM EMIT  [%d]: %s", chunkIdx, string(out))
		}
		w.Write(out)
		w.Write([]byte("\n"))
		flusher.Flush()
		chunkIdx++
	}

	if err := scanner.Err(); err != nil {
		log.Printf("stream scanner error: %v", err)
	}

	// Ensure a final done message is sent
	if !sentFinal {
		doneReason := pendingDoneReason
		if doneReason == "" {
			doneReason = "stop"
		}
		final := OllamaChatResponse{
			Model:      model,
			CreatedAt:  time.Now().UTC().Format(time.RFC3339Nano),
			Done:       true,
			DoneReason: doneReason,
			Message:    OllamaMessage{Role: "assistant"},
		}
		out, _ := json.Marshal(final)
		w.Write(out)
		w.Write([]byte("\n"))
		flusher.Flush()
	}
}

func isEmptyStreamChunk(chunk OllamaChatResponse) bool {
	return chunk.Message.Content == "" && chunk.Message.Thinking == "" && len(chunk.Message.ToolCalls) == 0
}

func respChunkHasNoUsage(chunk OpenAIChatResponse) bool {
	return chunk.Usage == nil
}

func isEmptyGenerateStreamChunk(chunk OllamaGenerateResponse) bool {
	return chunk.Response == "" && chunk.Thinking == "" && len(chunk.ToolCalls) == 0
}

func isZeroKeepAlive(raw json.RawMessage) bool {
	trimmed := strings.TrimSpace(string(raw))
	return trimmed == "0" || trimmed == "\"0\""
}

type streamedToolCallState struct {
	Name      string
	Arguments string
}

func appendStreamedToolCalls(states []streamedToolCallState, toolCalls []OpenAIToolCall) []streamedToolCallState {
	for idx, toolCall := range toolCalls {
		stateIndex := idx
		if toolCall.Index != nil && *toolCall.Index >= 0 {
			stateIndex = *toolCall.Index
		}
		for len(states) <= stateIndex {
			states = append(states, streamedToolCallState{})
		}
		if toolCall.Function.Name != "" {
			states[stateIndex].Name += toolCall.Function.Name
		}
		if toolCall.Function.Arguments != "" {
			states[stateIndex].Arguments += toolCall.Function.Arguments
		}
	}
	return states
}

func buildOllamaStreamToolCallChunk(states []streamedToolCallState, model string) (OllamaChatResponse, bool) {
	chunk := OllamaChatResponse{
		Model:     model,
		CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
		Done:      false,
		Message:   OllamaMessage{Role: "assistant"},
	}

	for _, state := range states {
		if state.Name == "" {
			continue
		}

		args := strings.TrimSpace(state.Arguments)
		rawArgs := json.RawMessage("{}")
		if args != "" {
			if json.Valid([]byte(args)) {
				rawArgs = json.RawMessage(args)
			} else if cfg.Debug {
				log.Printf("  STREAM TOOL CALL: invalid arguments JSON for %q, emitting empty object | args=%q", state.Name, truncateForLog(args, 200))
			}
		}

		chunk.Message.ToolCalls = append(chunk.Message.ToolCalls, OllamaToolCall{
			Function: OllamaToolCallFunction{
				Name:      state.Name,
				Arguments: rawArgs,
			},
		})
	}

	return chunk, len(chunk.Message.ToolCalls) > 0
}

func chunkToolCalls(chunk OpenAIChatResponse) []OpenAIToolCall {
	if len(chunk.Choices) == 0 || chunk.Choices[0].Delta == nil {
		return nil
	}
	return chunk.Choices[0].Delta.ToolCalls
}

func chunkFinishReason(chunk OpenAIChatResponse) string {
	if len(chunk.Choices) == 0 || chunk.Choices[0].FinishReason == nil {
		return ""
	}
	return *chunk.Choices[0].FinishReason
}

func rewriteRequestModel(body []byte) ([]byte, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}
	payload["model"] = cfg.VLLMModel
	return json.Marshal(payload)
}

// rewriteRequestForChat rewrites the model and disables thinking by default so
// that OpenAI passthrough clients receive plain content instead of vLLM's
// reasoning field. Clients that explicitly set chat_template_kwargs keep their
// setting, and tool-related fields are preserved.
func rewriteRequestForChat(body []byte, _ string) ([]byte, bool, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, false, err
	}
	payload["model"] = cfg.VLLMModel
	if _, ok := payload["chat_template_kwargs"]; !ok {
		payload["chat_template_kwargs"] = map[string]any{"enable_thinking": false}
	}

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
	content, _ := message["content"].(string)
	reasoningContent, _ := message["reasoning_content"].(string)
	reasoning, _ := message["reasoning"].(string)
	if content == "" {
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

func normalizeOpenAIJSON(payload []byte) ([]byte, error) {
	var value map[string]any
	if err := json.Unmarshal(payload, &value); err != nil {
		return nil, err
	}

	if _, ok := value["model"]; ok {
		value["model"] = cfg.ModelName
	}

	// vLLM may include large prompt token debug fields that downstream OpenAI
	// clients do not expect on chat completions responses.
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

func normalizeOpenAIStreamLine(line string) string {
	if !strings.HasPrefix(line, "data: ") {
		return line
	}

	data := strings.TrimPrefix(line, "data: ")
	if data == "" || data == "[DONE]" {
		return line
	}

	normalized, err := normalizeOpenAIJSON([]byte(data))
	if err != nil {
		if cfg.Debug {
			log.Printf("openai chat normalize skipped invalid chunk: %v | line=%s", err, truncateForLog(line, 200))
		}
		return line
	}

	return "data: " + string(normalized)
}

func copyResponseHeaders(dst http.ResponseWriter, src *http.Response) {
	for key, values := range src.Header {
		for _, value := range values {
			dst.Header().Add(key, value)
		}
	}
}

func handleOpenAIModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	metadata := currentModelMetadata(r.Context())
	resp := OpenAIModelListResponse{
		Object: "list",
		Data: []OpenAIModel{
			{
				Object:      "model",
				ID:          cfg.ModelName,
				OwnedBy:     "openai-ollama-proxy",
				Root:        metadata.ParentModel,
				MaxModelLen: metadata.ContextLength,
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleOpenAIChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	payload, strippedTools, err := rewriteRequestForChat(body, r.Header.Get("User-Agent"))
	if err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}
	reqSummary := requestDebugSummary(payload)
	if cfg.Debug {
		log.Printf("openai chat request start | %s | accept=%q content-type=%q", reqSummary, r.Header.Get("Accept"), r.Header.Get("Content-Type"))
		if strippedTools {
			log.Printf("openai chat request normalized for GitHubCopilotChat | tools stripped for direct-response compatibility")
		}
	}

	if cfg.Debug {
		log.Printf(">>> UPSTREAM (openai passthrough) POST %s/v1/chat/completions (%d bytes):\n  %s", cfg.VLLMBaseURL, len(payload), string(payload))
	}

	resp, err := doUpstreamChatWithRetry(r.Context(), payload)
	if err != nil {
		log.Printf("upstream openai-chat error: %v | %s", err, reqSummary)
		http.Error(w, "upstream not ready: "+err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	if cfg.Debug {
		log.Printf("<<< UPSTREAM (openai passthrough) %d | content-type=%q", resp.StatusCode, resp.Header.Get("Content-Type"))
	}

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(resp.Body)
		log.Printf("upstream openai-chat error %d: %s | %s | sent: %s", resp.StatusCode, string(errBody), reqSummary, string(payload))
		http.Error(w, fmt.Sprintf("upstream error: %d", resp.StatusCode), resp.StatusCode)
		return
	}

	if !strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") {
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			http.Error(w, "decode error: "+err.Error(), http.StatusInternalServerError)
			return
		}
		if cfg.Debug {
			log.Printf("<<< UPSTREAM (openai passthrough) non-stream body (%d bytes): %s", len(body), string(body))
		}
		normalized, err := normalizeOpenAIJSON(body)
		if err != nil {
			http.Error(w, "decode error: "+err.Error(), http.StatusInternalServerError)
			return
		}

		if cfg.Debug {
			log.Printf("<<< RESPONSE (openai passthrough) normalized: %s", string(normalized))
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		if _, err := w.Write(normalized); err != nil {
			log.Printf("openai chat proxy encode error | %s: %v", reqSummary, err)
		}
		return
	}

	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache, no-transform")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(resp.StatusCode)

	// Flush immediately so the client receives headers before generation starts.
	// Then stream line-by-line and flush after each line so SSE clients (e.g.
	// VS Code Copilot) receive tokens in real time instead of all at once.
	flusher, canFlush := w.(http.Flusher)
	if canFlush {
		flusher.Flush()
	}

	streamStart := time.Now()
	firstChunkLatency := time.Duration(0)
	chunkCount := 0
	byteCount := 0
	loggedFirstChunk := false

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 256*1024), 1024*1024)
	for scanner.Scan() {
		lineText := normalizeOpenAIStreamLine(scanner.Text())
		line := []byte(lineText)
		chunkCount++
		byteCount += len(line) + 1
		if !loggedFirstChunk {
			firstChunkLatency = time.Since(streamStart)
			loggedFirstChunk = true
			if cfg.Debug {
				log.Printf("openai chat first chunk after %s | canFlush=%t content-type=%q | %s | line=%s",
					firstChunkLatency.Round(time.Millisecond), canFlush, resp.Header.Get("Content-Type"), reqSummary, truncateForLog(string(line), 200))
			}
		}
		if _, err := w.Write(line); err != nil {
			log.Printf("openai chat write error after %d chunks %d bytes | %s: %v", chunkCount, byteCount, reqSummary, err)
			return
		}
		if _, err := w.Write([]byte("\n")); err != nil {
			log.Printf("openai chat newline write error after %d chunks %d bytes | %s: %v", chunkCount, byteCount, reqSummary, err)
			return
		}
		if canFlush {
			flusher.Flush()
		}
	}
	if err := scanner.Err(); err != nil {
		log.Printf("openai chat proxy stream error after %s firstChunk=%s chunks=%d bytes=%d canFlush=%t | %s: %v",
			time.Since(streamStart).Round(time.Millisecond), firstChunkLatency.Round(time.Millisecond), chunkCount, byteCount, canFlush, reqSummary, err)
		return
	}

	log.Printf("openai chat stream complete in %s firstChunk=%s chunks=%d bytes=%d canFlush=%t content-type=%q | %s",
		time.Since(streamStart).Round(time.Millisecond), firstChunkLatency.Round(time.Millisecond), chunkCount, byteCount, canFlush, resp.Header.Get("Content-Type"), reqSummary)
}

func handleOpenAIEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	payload, err := rewriteRequestModel(body)
	if err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	upstream, err := http.NewRequestWithContext(r.Context(), http.MethodPost,
		cfg.VLLMBaseURL+"/v1/embeddings", bytes.NewReader(payload))
	if err != nil {
		http.Error(w, "request error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	upstream.Header.Set("Content-Type", "application/json")
	if cfg.VLLMAPIKey != "" {
		upstream.Header.Set("Authorization", "Bearer "+cfg.VLLMAPIKey)
	}

	resp, err := httpClient.Do(upstream)
	if err != nil {
		http.Error(w, "upstream error: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	copyResponseHeaders(w, resp)
	w.WriteHeader(resp.StatusCode)
	if _, err := io.Copy(w, resp.Body); err != nil {
		log.Printf("openai embeddings proxy copy error: %v", err)
	}
}

func handlePull(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	defer r.Body.Close()
	var req OllamaPullRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && !errors.Is(err, io.EOF) {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	json.NewEncoder(w).Encode(OllamaProgressResponse{Status: "success"})
}

func handleCreate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()
	io.Copy(io.Discard, r.Body)

	w.Header().Set("Content-Type", "application/x-ndjson")
	json.NewEncoder(w).Encode(OllamaProgressResponse{Status: "success"})
}

func handleCopy(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()
	io.Copy(io.Discard, r.Body)

	w.WriteHeader(http.StatusOK)
}

func handleDelete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()
	io.Copy(io.Discard, r.Body)

	w.WriteHeader(http.StatusOK)
}

func handleBlobs(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()
	io.Copy(io.Discard, r.Body)

	switch r.Method {
	case http.MethodHead:
		w.WriteHeader(http.StatusOK)
	case http.MethodPost:
		w.WriteHeader(http.StatusCreated)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func handleGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	var olReq OllamaGenerateRequest
	if err := json.Unmarshal(body, &olReq); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	if olReq.Prompt == "" {
		doneReason := "load"
		if isZeroKeepAlive(olReq.KeepAlive) {
			doneReason = "unload"
		}
		resp := OllamaGenerateResponse{
			Model:      olReq.Model,
			CreatedAt:  time.Now().UTC().Format(time.RFC3339Nano),
			Response:   "",
			Done:       true,
			DoneReason: doneReason,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
		return
	}

	oaiReq, err := ollamaGenerateToOpenAI(olReq)
	if err != nil {
		http.Error(w, "translation error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	oaiBody, err := json.Marshal(oaiReq)
	if err != nil {
		http.Error(w, "marshal error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	resp, err := doUpstreamChatWithRetry(r.Context(), oaiBody)
	if err != nil {
		http.Error(w, "upstream not ready: "+err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(resp.Body)
		log.Printf("upstream generate error %d: %s | sent: %s", resp.StatusCode, string(errBody), string(oaiBody))
		http.Error(w, fmt.Sprintf("upstream error: %d", resp.StatusCode), resp.StatusCode)
		return
	}

	stream := true
	if olReq.Stream != nil {
		stream = *olReq.Stream
	}

	if stream {
		handleGenerateStream(w, resp.Body, olReq.Model)
	} else {
		handleGenerateNonStream(w, resp.Body, olReq.Model)
	}
}

func handleGenerateNonStream(w http.ResponseWriter, body io.Reader, model string) {
	var oaiResp OpenAIChatResponse
	if err := json.NewDecoder(body).Decode(&oaiResp); err != nil {
		http.Error(w, "decode error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	olResp := openAIChatToOllamaGenerate(oaiResp, model)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(olResp)
}

func handleGenerateStream(w http.ResponseWriter, body io.Reader, model string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	scanner := bufio.NewScanner(body)
	// SSE lines can be large with tool calls / big content
	scanner.Buffer(make([]byte, 0, 256*1024), 1024*1024)
	pendingDoneReason := ""
	sentFinal := false

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var chunk OpenAIChatResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			log.Printf("generate stream decode error: %v, data: %s", err, data)
			continue
		}

		olChunk := openAIStreamChunkToOllamaGenerate(chunk, model)
		if olChunk.DoneReason != "" && isEmptyGenerateStreamChunk(olChunk) && respChunkHasNoUsage(chunk) {
			pendingDoneReason = olChunk.DoneReason
			continue
		}
		if olChunk.Done && olChunk.DoneReason == "" && pendingDoneReason != "" {
			olChunk.DoneReason = pendingDoneReason
		}
		if isEmptyGenerateStreamChunk(olChunk) && !olChunk.Done {
			continue
		}
		if olChunk.Done {
			sentFinal = true
		}

		out, err := json.Marshal(olChunk)
		if err != nil {
			log.Printf("generate stream encode error: %v", err)
			continue
		}
		w.Write(out)
		w.Write([]byte("\n"))
		flusher.Flush()
	}

	if !sentFinal {
		doneReason := pendingDoneReason
		if doneReason == "" {
			doneReason = "stop"
		}
		final := OllamaGenerateResponse{
			Model:      model,
			CreatedAt:  time.Now().UTC().Format(time.RFC3339Nano),
			Response:   "",
			Done:       true,
			DoneReason: doneReason,
		}
		out, _ := json.Marshal(final)
		w.Write(out)
		w.Write([]byte("\n"))
		flusher.Flush()
	}
}

func handleEmbed(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	var olReq OllamaEmbedRequest
	if err := json.Unmarshal(body, &olReq); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Build OpenAI embedding request
	oaiReq := OpenAIEmbedRequest{
		Model: resolveModel(olReq.Model),
		Input: olReq.Input,
	}

	// Handle legacy /api/embeddings with "prompt" field
	if len(olReq.Input) == 0 && olReq.Prompt != "" {
		b, _ := json.Marshal(olReq.Prompt)
		oaiReq.Input = b
	}

	oaiBody, err := json.Marshal(oaiReq)
	if err != nil {
		http.Error(w, "marshal error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	upstream, err := http.NewRequestWithContext(r.Context(), http.MethodPost,
		cfg.VLLMBaseURL+"/v1/embeddings", bytes.NewReader(oaiBody))
	if err != nil {
		http.Error(w, "request error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	upstream.Header.Set("Content-Type", "application/json")
	if cfg.VLLMAPIKey != "" {
		upstream.Header.Set("Authorization", "Bearer "+cfg.VLLMAPIKey)
	}

	resp, err := httpClient.Do(upstream)
	if err != nil {
		http.Error(w, "upstream error: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(resp.Body)
		log.Printf("upstream embed error %d: %s", resp.StatusCode, string(errBody))
		http.Error(w, fmt.Sprintf("upstream error: %d", resp.StatusCode), resp.StatusCode)
		return
	}

	var oaiResp OpenAIEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&oaiResp); err != nil {
		http.Error(w, "decode error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	olResp := openAIEmbedToOllama(oaiResp, olReq.Model)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(olResp)
}

// handleEmbeddings handles the deprecated /api/embeddings endpoint (single prompt → single embedding).
func handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	var olReq OllamaEmbedRequest
	if err := json.Unmarshal(body, &olReq); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	input := olReq.Input
	if len(input) == 0 && olReq.Prompt != "" {
		b, _ := json.Marshal(olReq.Prompt)
		input = b
	}

	oaiReq := OpenAIEmbedRequest{
		Model: resolveModel(olReq.Model),
		Input: input,
	}

	oaiBody, err := json.Marshal(oaiReq)
	if err != nil {
		http.Error(w, "marshal error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	upstream, err := http.NewRequestWithContext(r.Context(), http.MethodPost,
		cfg.VLLMBaseURL+"/v1/embeddings", bytes.NewReader(oaiBody))
	if err != nil {
		http.Error(w, "request error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	upstream.Header.Set("Content-Type", "application/json")
	if cfg.VLLMAPIKey != "" {
		upstream.Header.Set("Authorization", "Bearer "+cfg.VLLMAPIKey)
	}

	resp, err := httpClient.Do(upstream)
	if err != nil {
		http.Error(w, "upstream error: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(resp.Body)
		log.Printf("upstream embeddings error %d: %s", resp.StatusCode, string(errBody))
		http.Error(w, fmt.Sprintf("upstream error: %d", resp.StatusCode), resp.StatusCode)
		return
	}

	var oaiResp OpenAIEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&oaiResp); err != nil {
		http.Error(w, "decode error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Return single embedding in legacy format
	result := OllamaEmbeddingsResponse{}
	if len(oaiResp.Data) > 0 {
		result.Embedding = oaiResp.Data[0].Embedding
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func handleTags(w http.ResponseWriter, r *http.Request) {
	metadata := currentModelMetadata(r.Context())
	resp := OllamaTagsResponse{
		Models: []OllamaModelInfo{
			{
				Name:       cfg.ModelName,
				Model:      cfg.ModelName,
				ModifiedAt: time.Now().UTC().Format(time.RFC3339),
				Size:       0,
				Digest:     "proxy",
				Details: OllamaModelDetails{
					ParentModel:       metadata.ParentModel,
					Format:            metadata.Format,
					Family:            metadata.Family,
					Families:          []string{metadata.Family},
					ParameterSize:     metadata.ParameterSize,
					QuantizationLevel: metadata.Quantization,
				},
			},
		},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleShow(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	metadata := currentModelMetadata(r.Context())

	resp := OllamaShowResponse{
		Modelfile:  "# proxied model",
		Parameters: fmt.Sprintf("num_ctx %d", metadata.ContextLength),
		Template:   "",
		Details: OllamaModelDetails{
			ParentModel:       metadata.ParentModel,
			Format:            metadata.Format,
			Family:            metadata.Family,
			Families:          []string{metadata.Family},
			ParameterSize:     metadata.ParameterSize,
			QuantizationLevel: metadata.Quantization,
		},
		ModelInfo:    ollamaModelInfo(metadata),
		Capabilities: []string{"completion", "tools"},
	}
	if resp.Parameters == "" {
		resp.Parameters = fmt.Sprintf("num_ctx %d", metadata.ContextLength)
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handleVersion(w http.ResponseWriter, r *http.Request) {
	resp := OllamaVersionResponse{Version: cfg.OllamaVersion}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func handlePs(w http.ResponseWriter, r *http.Request) {
	metadata := currentModelMetadata(r.Context())
	resp := OllamaPsResponse{
		Models: []OllamaPsModel{
			{
				Name:   cfg.ModelName,
				Model:  cfg.ModelName,
				Size:   0,
				Digest: "proxy",
				Details: OllamaModelDetails{
					ParentModel:       metadata.ParentModel,
					Format:            metadata.Format,
					Family:            metadata.Family,
					Families:          []string{metadata.Family},
					ParameterSize:     metadata.ParameterSize,
					QuantizationLevel: metadata.Quantization,
				},
				ExpiresAt: time.Now().Add(24 * time.Hour).UTC().Format(time.RFC3339),
				SizeVRAM:  0,
			},
		},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleHead responds to HEAD requests (Ollama client health checks).
func handleHead(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	healthy, err := probeVLLMHealth(r.Context())
	if err != nil || !healthy {
		w.WriteHeader(http.StatusServiceUnavailable)
		return
	}

	w.WriteHeader(http.StatusOK)
}

// handleRoot responds to GET / with "Ollama is running".
func handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	w.Header().Set("Content-Type", "text/plain")
	healthy, err := probeVLLMHealth(r.Context())
	if err != nil || !healthy {
		w.WriteHeader(http.StatusServiceUnavailable)
		fmt.Fprint(w, "Ollama is down")
		return
	}

	fmt.Fprint(w, "Ollama is running")
}
