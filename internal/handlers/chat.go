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
	"github.com/k0in/openai-ollama-proxy/internal/streaming"
	"github.com/k0in/openai-ollama-proxy/internal/translate"
	"github.com/k0in/openai-ollama-proxy/internal/types"
)

func (server *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read body", http.StatusBadRequest)
		return
	}
	defer func() { _ = r.Body.Close() }()

	var ollamaReq types.OllamaChatRequest
	if err := json.Unmarshal(body, &ollamaReq); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	if len(ollamaReq.Messages) == 0 {
		doneReason := "load"
		if isZeroKeepAlive(ollamaReq.KeepAlive) {
			doneReason = "unload"
		}
		resp := types.OllamaChatResponse{
			Model:      ollamaReq.Model,
			CreatedAt:  time.Now().UTC().Format(time.RFC3339Nano),
			Done:       true,
			DoneReason: doneReason,
			Message:    types.OllamaMessage{Role: "assistant"},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
		return
	}

	openAIReq, err := translate.OllamaChatToOpenAI(ollamaReq)
	if err != nil {
		http.Error(w, "translation error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Resolve route for the requested model.
	baseURL, apiKey, upstreamModel, _, passthrough, found := server.resolveRouteForModelPassthrough(ollamaReq.Model)
	if !found {
		http.Error(w, fmt.Sprintf("model not configured: %q", ollamaReq.Model), http.StatusNotFound)
		return
	}
	apiKey = server.resolveEffectiveAPIKey(apiKey, passthrough, applogging.ExtractAPIKey(r))
	openAIReq.Model = upstreamModel

	// Inject model-level thinking level if no explicit think was set by client.
	if ollamaReq.Think == nil {
		if entry, ok := server.router.Lookup(ollamaReq.Model); ok && entry.ThinkingLevel != "" {
			level := entry.ThinkingLevel
			openAIReq.ReasoningEffort = &level
		}
	}

	openAIBody, err := json.Marshal(openAIReq)
	if err != nil {
		http.Error(w, "marshal error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	if server.cfg.Debug {
		log.Printf(">>> UPSTREAM POST %s/v1/chat/completions (%d bytes, model=%q)", baseURL, len(openAIBody), upstreamModel)
	}

	timings := newObservedTimings()

	resp, err := server.doUpstreamChatWithRetryForRoute(r.Context(), openAIBody, baseURL, apiKey, server.shouldRetryOnError(ollamaReq.Model))
	if err != nil {
		log.Printf("upstream chat error: %v", err)
		http.Error(w, "upstream not ready: "+err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer func() { _ = resp.Body.Close() }()

	if server.cfg.Debug {
		log.Printf("<<< UPSTREAM %d | content-type=%q", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	timings.markResponseStart()

	if resp.StatusCode != http.StatusOK {
		_, _ = io.ReadAll(resp.Body)
		log.Printf("upstream chat error %d | sent: %d bytes", resp.StatusCode, len(openAIBody))
		http.Error(w, fmt.Sprintf("upstream error: %d", resp.StatusCode), resp.StatusCode)
		return
	}

	stream := true
	if ollamaReq.Stream != nil {
		stream = *ollamaReq.Stream
	}

	if stream {
		server.handleChatStream(w, resp.Body, ollamaReq.Model, timings)
		return
	}

	server.handleChatNonStream(w, resp.Body, ollamaReq.Model, timings)
}

func (server *Server) handleChatNonStream(w http.ResponseWriter, body io.Reader, model string, timings *observedTimings) {
	rawBody, err := io.ReadAll(body)
	if err != nil {
		http.Error(w, "read error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	if server.cfg.Debug {
		log.Printf("<<< UPSTREAM BODY (non-stream, %d bytes)", len(rawBody))
	}

	var openAIResp types.OpenAIChatResponse
	if err := json.Unmarshal(rawBody, &openAIResp); err != nil {
		http.Error(w, "decode error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	timings.markComplete()

	if openAIResp.Usage != nil {
		server.stats.Record(model, openAIResp.Usage.PromptTokens, openAIResp.Usage.CompletionTokens, time.Duration(timings.evalDuration()))
		if server.cfg.Debug && openAIResp.Usage.CompletionTokensDetails != nil {
			log.Printf("<<< UPSTREAM reasoning_tokens=%d", openAIResp.Usage.CompletionTokensDetails.ReasoningTokens)
		}
	}

	ollamaResp := translate.OpenAIChatToOllama(openAIResp, model)
	applyObservedChatTimings(&ollamaResp, timings)

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(ollamaResp)
}

func (server *Server) handleChatStream(w http.ResponseWriter, body io.Reader, model string, timings *observedTimings) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	scanner := bufio.NewScanner(body)
	// SSE lines from upstream can be large when streaming long tool-call deltas or
	// reasoning blocks. Allow up to 10 MiB per line; bufio.ErrTooLong is
	// surfaced via scanner.Err() at the end of the loop.
	scanner.Buffer(make([]byte, 0, 256*1024), 10*1024*1024)
	pendingDoneReason := ""
	sentFinal := false
	chunkIndex := 0
	var toolCallStates []streaming.ToolCallState
	var lastReasoningTokens int

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
			log.Printf("stream decode error: %v", err)
			continue
		}

		// Track reasoning_tokens from usage-bearing chunks (usually the final chunk).
		if chunk.Usage != nil && chunk.Usage.CompletionTokensDetails != nil {
			lastReasoningTokens = chunk.Usage.CompletionTokensDetails.ReasoningTokens
		}

		toolCallDeltas := chunkToolCalls(chunk)
		if len(toolCallDeltas) > 0 {
			toolCallStates = streaming.AppendToolCalls(toolCallStates, toolCallDeltas)
			chunkIndex++
			continue
		}

		if chunkFinishReason(chunk) == "tool_calls" && len(toolCallStates) > 0 {
			toolChunk, ok := streaming.BuildOllamaToolCallChunk(toolCallStates, model, server.cfg.Debug)
			toolCallStates = nil
			if ok {
				timings.markFirstVisibleOutput()
				out, err := json.Marshal(toolChunk)
				if err != nil {
					log.Printf("stream tool-call encode error: %v", err)
				} else {
					_, _ = w.Write(out)
					_, _ = w.Write([]byte("\n"))
					flusher.Flush()
					chunkIndex++
				}
			}
		}

		ollamaChunk := translate.OpenAIStreamChunkToOllama(chunk, model)

		if ollamaChunk.DoneReason != "" && isEmptyStreamChunk(ollamaChunk) && respChunkHasNoUsage(chunk) {
			pendingDoneReason = ollamaChunk.DoneReason
			chunkIndex++
			continue
		}
		if ollamaChunk.Done && ollamaChunk.DoneReason == "" && pendingDoneReason != "" {
			ollamaChunk.DoneReason = pendingDoneReason
		}
		if isEmptyStreamChunk(ollamaChunk) && !ollamaChunk.Done {
			chunkIndex++
			continue
		}
		if !ollamaChunk.Done {
			timings.markFirstVisibleOutput()
		}
		if ollamaChunk.Done {
			timings.markComplete()
			applyObservedChatTimings(&ollamaChunk, timings)

			server.stats.Record(model, ollamaChunk.PromptEvalCount, ollamaChunk.EvalCount, time.Duration(timings.evalDuration()))
			sentFinal = true
		}

		out, err := json.Marshal(ollamaChunk)
		if err != nil {
			log.Printf("stream encode error: %v", err)
			chunkIndex++
			continue
		}
		_, _ = w.Write(out)
		_, _ = w.Write([]byte("\n"))
		flusher.Flush()
		chunkIndex++
	}

	if err := scanner.Err(); err != nil {
		log.Printf("stream scanner error: %v", err)
	}

	if server.cfg.Debug && lastReasoningTokens > 0 {
		log.Printf("<<< UPSTREAM reasoning_tokens=%d", lastReasoningTokens)
	}

	if !sentFinal {
		doneReason := pendingDoneReason
		if doneReason == "" {
			doneReason = "stop"
		}
		final := types.OllamaChatResponse{
			Model:      model,
			CreatedAt:  time.Now().UTC().Format(time.RFC3339Nano),
			Done:       true,
			DoneReason: doneReason,
			Message:    types.OllamaMessage{Role: "assistant"},
		}
		timings.markComplete()
		applyObservedChatTimings(&final, timings)
		out, _ := json.Marshal(final)
		_, _ = w.Write(out)
		_, _ = w.Write([]byte("\n"))
		flusher.Flush()
	}
}

func isEmptyStreamChunk(chunk types.OllamaChatResponse) bool {
	return chunk.Message.Content == "" && chunk.Message.Thinking == "" && len(chunk.Message.ToolCalls) == 0
}

func respChunkHasNoUsage(chunk types.OpenAIChatResponse) bool {
	return chunk.Usage == nil
}

func chunkToolCalls(chunk types.OpenAIChatResponse) []types.OpenAIToolCall {
	if len(chunk.Choices) == 0 || chunk.Choices[0].Delta == nil {
		return nil
	}
	return chunk.Choices[0].Delta.ToolCalls
}

func chunkFinishReason(chunk types.OpenAIChatResponse) string {
	if len(chunk.Choices) == 0 || chunk.Choices[0].FinishReason == nil {
		return ""
	}
	return *chunk.Choices[0].FinishReason
}
