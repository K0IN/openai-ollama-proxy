package handlers

import (
	"bufio"
	"log"
	"net/http"
	"time"
)

func (server *Server) proxyOpenAIStream(w http.ResponseWriter, resp *http.Response, reqSummary string) {
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
	for scanner.Scan() {
		lineText := server.normalizeOpenAIStreamLine(scanner.Text())
		line := []byte(lineText)
		chunkCount++
		byteCount += len(line) + 1
		if !loggedFirstChunk {
			firstChunkLatency = time.Since(streamStart)
			loggedFirstChunk = true
			if server.cfg.Debug {
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
