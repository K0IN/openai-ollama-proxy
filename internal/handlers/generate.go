package handlers

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/k0in/openai-ollama-proxy/internal/translate"
	"github.com/k0in/openai-ollama-proxy/internal/types"
)

func (server *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
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

	var ollamaReq types.OllamaGenerateRequest
	if err := json.Unmarshal(body, &ollamaReq); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	if ollamaReq.Prompt == "" {
		doneReason := ""
		if isZeroKeepAlive(ollamaReq.KeepAlive) {
			doneReason = "unload"
		}
		resp := types.OllamaGenerateResponse{
			Model:      ollamaReq.Model,
			CreatedAt:  time.Now().UTC().Format(time.RFC3339Nano),
			Response:   "",
			Done:       true,
			DoneReason: doneReason,
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
		return
	}

	openAIReq, err := translate.OllamaGenerateToOpenAI(ollamaReq)
	if err != nil {
		http.Error(w, "translation error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	openAIReq.Model = server.cfg.VLLMModel

	openAIBody, err := json.Marshal(openAIReq)
	if err != nil {
		http.Error(w, "marshal error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	timings := newObservedTimings()

	resp, err := server.doUpstreamChatWithRetry(r.Context(), openAIBody)
	if err != nil {
		http.Error(w, "upstream not ready: "+err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(resp.Body)
		log.Printf("upstream generate error %d: %s | sent: %s", resp.StatusCode, string(errBody), string(openAIBody))
		http.Error(w, fmt.Sprintf("upstream error: %d", resp.StatusCode), resp.StatusCode)
		return
	}
	timings.markResponseStart()

	stream := true
	if ollamaReq.Stream != nil {
		stream = *ollamaReq.Stream
	}

	if stream {
		server.handleGenerateStream(w, resp.Body, ollamaReq.Model, timings)
		return
	}

	server.handleGenerateNonStream(w, resp.Body, ollamaReq.Model, timings)
}

func (server *Server) handleGenerateNonStream(w http.ResponseWriter, body io.Reader, model string, timings *observedTimings) {
	var openAIResp types.OpenAIChatResponse
	if err := json.NewDecoder(body).Decode(&openAIResp); err != nil {
		http.Error(w, "decode error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	timings.markComplete()

	ollamaResp := translate.OpenAIChatToOllamaGenerate(openAIResp, model)
	applyObservedGenerateTimings(&ollamaResp, timings)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(ollamaResp)
}

func (server *Server) handleGenerateStream(w http.ResponseWriter, body io.Reader, model string, timings *observedTimings) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 256*1024), 1024*1024)
	pendingDoneReason := ""
	sentFinal := false

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
			log.Printf("generate stream decode error: %v, data: %s", err, data)
			continue
		}

		ollamaChunk := translate.OpenAIStreamChunkToOllamaGenerate(chunk, model)
		if ollamaChunk.DoneReason != "" && isEmptyGenerateStreamChunk(ollamaChunk) && respChunkHasNoUsage(chunk) {
			pendingDoneReason = ollamaChunk.DoneReason
			continue
		}
		if ollamaChunk.Done && ollamaChunk.DoneReason == "" && pendingDoneReason != "" {
			ollamaChunk.DoneReason = pendingDoneReason
		}
		if isEmptyGenerateStreamChunk(ollamaChunk) && !ollamaChunk.Done {
			continue
		}
		if !ollamaChunk.Done {
			timings.markFirstVisibleOutput()
		}
		if ollamaChunk.Done {
			timings.markComplete()
			applyObservedGenerateTimings(&ollamaChunk, timings)
			sentFinal = true
		}

		out, err := json.Marshal(ollamaChunk)
		if err != nil {
			log.Printf("generate stream encode error: %v", err)
			continue
		}
		_, _ = w.Write(out)
		_, _ = w.Write([]byte("\n"))
		flusher.Flush()
	}

	if err := scanner.Err(); err != nil {
		log.Printf("generate stream scanner error: %v", err)
	}

	if !sentFinal {
		doneReason := pendingDoneReason
		if doneReason == "" {
			doneReason = "stop"
		}
		final := types.OllamaGenerateResponse{
			Model:      model,
			CreatedAt:  time.Now().UTC().Format(time.RFC3339Nano),
			Response:   "",
			Done:       true,
			DoneReason: doneReason,
		}
		timings.markComplete()
		applyObservedGenerateTimings(&final, timings)
		out, _ := json.Marshal(final)
		_, _ = w.Write(out)
		_, _ = w.Write([]byte("\n"))
		flusher.Flush()
	}
}

func isEmptyGenerateStreamChunk(chunk types.OllamaGenerateResponse) bool {
	return chunk.Response == "" && chunk.Thinking == "" && len(chunk.ToolCalls) == 0
}

func (server *Server) handleEmbed(w http.ResponseWriter, r *http.Request) {
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

	var ollamaReq types.OllamaEmbedRequest
	if err := json.Unmarshal(body, &ollamaReq); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	openAIReq := types.OpenAIEmbedRequest{Model: server.cfg.VLLMModel, Input: ollamaReq.Input}
	if len(ollamaReq.Input) == 0 && ollamaReq.Prompt != "" {
		input, _ := json.Marshal(ollamaReq.Prompt)
		openAIReq.Input = input
	}

	openAIBody, err := json.Marshal(openAIReq)
	if err != nil {
		http.Error(w, "marshal error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	upstream, err := http.NewRequestWithContext(r.Context(), http.MethodPost, server.cfg.VLLMBaseURL+"/v1/embeddings", bytes.NewReader(openAIBody))
	if err != nil {
		http.Error(w, "request error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	upstream.Header.Set("Content-Type", "application/json")
	if server.cfg.VLLMAPIKey != "" {
		upstream.Header.Set("Authorization", "Bearer "+server.cfg.VLLMAPIKey)
	}
	timings := newObservedTimings()

	resp, err := server.client.Do(upstream)
	if err != nil {
		http.Error(w, "upstream error: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	timings.markResponseStart()

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(resp.Body)
		log.Printf("upstream embed error %d: %s", resp.StatusCode, string(errBody))
		http.Error(w, fmt.Sprintf("upstream error: %d", resp.StatusCode), resp.StatusCode)
		return
	}

	var openAIResp types.OpenAIEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&openAIResp); err != nil {
		http.Error(w, "decode error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	timings.markComplete()

	ollamaResp := translate.OpenAIEmbedToOllama(openAIResp, ollamaReq.Model)
	applyObservedEmbedTimings(&ollamaResp, timings)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(ollamaResp)
}

func (server *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
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

	var ollamaReq types.OllamaEmbedRequest
	if err := json.Unmarshal(body, &ollamaReq); err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	input := ollamaReq.Input
	if len(input) == 0 && ollamaReq.Prompt != "" {
		input, _ = json.Marshal(ollamaReq.Prompt)
	}

	openAIReq := types.OpenAIEmbedRequest{Model: server.cfg.VLLMModel, Input: input}
	openAIBody, err := json.Marshal(openAIReq)
	if err != nil {
		http.Error(w, "marshal error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	upstream, err := http.NewRequestWithContext(r.Context(), http.MethodPost, server.cfg.VLLMBaseURL+"/v1/embeddings", bytes.NewReader(openAIBody))
	if err != nil {
		http.Error(w, "request error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	upstream.Header.Set("Content-Type", "application/json")
	if server.cfg.VLLMAPIKey != "" {
		upstream.Header.Set("Authorization", "Bearer "+server.cfg.VLLMAPIKey)
	}

	resp, err := server.client.Do(upstream)
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

	var openAIResp types.OpenAIEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&openAIResp); err != nil {
		http.Error(w, "decode error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	result := types.OllamaEmbeddingsResponse{}
	if len(openAIResp.Data) > 0 {
		result.Embedding = openAIResp.Data[0].Embedding
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(result)
}
