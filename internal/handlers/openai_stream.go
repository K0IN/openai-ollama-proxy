package handlers

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/k0in/openai-ollama-proxy/internal/types"
)

func (server *Server) proxyOpenAIStream(w http.ResponseWriter, resp *http.Response, reqSummary string, timings *observedTimings) {
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache, no-transform")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(resp.StatusCode)

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
	// See chat.go for sizing rationale.
	scanner.Buffer(make([]byte, 0, 256*1024), 10*1024*1024)

	var lastChunkWithUsage types.OpenAIChatResponse
	var upstreamModelForStats string // tracked from any chunk, used as fallback

	for scanner.Scan() {
		lineText := server.normalizeOpenAIStreamLine(scanner.Text())
		line := []byte(lineText)
		chunkCount++
		byteCount += len(line) + 1

		// Try to parse the line as a JSON chunk
		if strings.HasPrefix(lineText, "data: ") {
			data := strings.TrimPrefix(lineText, "data: ")
			if data != "[DONE]" {
				var chunk types.OpenAIChatResponse
				if err := json.Unmarshal([]byte(data), &chunk); err == nil {
					if chunk.Usage != nil {
						lastChunkWithUsage = chunk
					}
					if chunk.Model != "" {
						upstreamModelForStats = chunk.Model
					}
					if len(chunk.Choices) > 0 && chunk.Choices[0].Delta != nil && chunk.Choices[0].Delta.Content != nil && *chunk.Choices[0].Delta.Content != "" {
						timings.markFirstVisibleOutput()
					}
				}
			}
		}

		if !loggedFirstChunk {
			firstChunkLatency = time.Since(streamStart)
			loggedFirstChunk = true
			if server.cfg.Debug {
				log.Printf("openai chat first chunk after %s | canFlush=%t content-type=%q | %s",
					firstChunkLatency.Round(time.Millisecond), canFlush, resp.Header.Get("Content-Type"), reqSummary)
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

	timings.markComplete()

	// Use the model name from the final usage chunk if available, otherwise
	// fall back to any model name seen in the stream (some providers omit
	// "model" from the usage-bearing chunk).
	statsModel := lastChunkWithUsage.Model
	if statsModel == "" {
		statsModel = upstreamModelForStats
	}
	if lastChunkWithUsage.Usage != nil && statsModel != "" {
		cachedInput := 0
		if lastChunkWithUsage.Usage.PromptTokensDetails != nil {
			cachedInput = lastChunkWithUsage.Usage.PromptTokensDetails.CachedTokens
		}
		server.stats.Record(statsModel, lastChunkWithUsage.Usage.PromptTokens, lastChunkWithUsage.Usage.CompletionTokens, cachedInput, time.Duration(timings.evalDuration()))
	}

	var reasoningTokensStr string
	if lastChunkWithUsage.Usage != nil && lastChunkWithUsage.Usage.CompletionTokensDetails != nil {
		reasoningTokensStr = fmt.Sprintf(" reasoning_tokens=%d", lastChunkWithUsage.Usage.CompletionTokensDetails.ReasoningTokens)
	}

	log.Printf("openai chat stream complete in %s firstChunk=%s chunks=%d bytes=%d canFlush=%t content-type=%q%s | %s",
		time.Since(streamStart).Round(time.Millisecond), firstChunkLatency.Round(time.Millisecond), chunkCount, byteCount, canFlush, resp.Header.Get("Content-Type"), reasoningTokensStr, reqSummary)
}
