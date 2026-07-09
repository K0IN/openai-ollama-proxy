package handlers

import (
	"bytes"
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

func (server *Server) handleOpenAIModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	models := server.router.AllModels()
	resp := types.OpenAIModelListResponse{
		Object: "list",
		Data:   make([]types.OpenAIModel, 0, len(models)),
	}
	for _, m := range models {
		entry, ok := server.router.Lookup(m)
		ctxLen := server.cfg.ModelContextLength
		supportsVision := false
		supportsThinking := false
		if ok {
			if entry.ContextLength > 0 {
				ctxLen = entry.ContextLength
			}
			supportsVision = entry.SupportsVision
			supportsThinking = entry.SupportsThinking
		}
		resp.Data = append(resp.Data, types.OpenAIModel{
			Object:           "model",
			ID:               m,
			OwnedBy:          "openai-ollama-proxy",
			MaxModelLen:      ctxLen,
			SupportsVision:   supportsVision,
			SupportsThinking: supportsThinking,
		})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (server *Server) handleOpenAIChat(w http.ResponseWriter, r *http.Request) {
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

	// Extract the model name from the original request before rewriting.
	var origPayload map[string]any
	_ = json.Unmarshal(body, &origPayload)
	reqModel, _ := origPayload["model"].(string)
	baseURL, apiKey, _, _, passthrough, found := server.resolveRouteForModelPassthrough(reqModel)
	if !found {
		http.Error(w, fmt.Sprintf("model not configured: %q", reqModel), http.StatusNotFound)
		return
	}
	apiKey = server.resolveEffectiveAPIKey(apiKey, passthrough, applogging.ExtractAPIKey(r))

	payload, strippedTools, err := server.rewriteRequestForChat(body)
	if err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}
	reqSummary := requestDebugSummary(payload)

	if server.cfg.Debug {
		log.Printf("openai chat request start | %s | accept=%q content-type=%q", reqSummary, applogging.SanitizeForLog(r.Header.Get("Accept")), applogging.SanitizeForLog(r.Header.Get("Content-Type"))) // #nosec G706 -- inputs sanitized via SanitizeForLog
		if strippedTools {
			log.Printf("openai chat request normalized | tools stripped for direct-response compatibility")
		}
		log.Printf(">>> UPSTREAM (openai passthrough) POST %s/v1/chat/completions (%d bytes)", baseURL, len(payload))
	}

	timings := newObservedTimings()

	resp, err := server.doUpstreamChatWithRetryForRoute(r.Context(), payload, baseURL, apiKey, server.shouldRetryOnError(reqModel))
	if err != nil {
		log.Printf("upstream openai-chat error: %v | %s", err, reqSummary)
		http.Error(w, "upstream not ready: "+err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer func() { _ = resp.Body.Close() }()
	timings.markResponseStart()

	if server.cfg.Debug {
		log.Printf("<<< UPSTREAM (openai passthrough) %d | content-type=%q", resp.StatusCode, resp.Header.Get("Content-Type"))
	}

	if resp.StatusCode != http.StatusOK {
		_, _ = io.ReadAll(resp.Body)
		log.Printf("upstream openai-chat error %d | %s | sent: %d bytes", resp.StatusCode, reqSummary, len(payload))
		http.Error(w, fmt.Sprintf("upstream error: %d", resp.StatusCode), resp.StatusCode)
		return
	}

	if !strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") {
		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			http.Error(w, "decode error: "+err.Error(), http.StatusInternalServerError)
			return
		}
		if server.cfg.Debug {
			log.Printf("<<< UPSTREAM (openai passthrough) non-stream body (%d bytes)", len(respBody))
		}

		normalized, err := server.normalizeOpenAIJSON(respBody)
		if err != nil {
			http.Error(w, "decode error: "+err.Error(), http.StatusInternalServerError)
			return
		}

		if server.cfg.Debug {
			log.Printf("<<< RESPONSE (openai passthrough) normalized: %d bytes", len(normalized))
		}

		timings.markComplete()

		var openAIResp types.OpenAIChatResponse
		statsModel := openAIResp.Model
		if statsModel == "" {
			statsModel = reqModel
		}
		if err := json.Unmarshal(normalized, &openAIResp); err == nil && openAIResp.Usage != nil {
			cachedInput := 0
			if openAIResp.Usage.PromptTokensDetails != nil {
				cachedInput = openAIResp.Usage.PromptTokensDetails.CachedTokens
			}
			server.stats.Record(statsModel, openAIResp.Usage.PromptTokens, openAIResp.Usage.CompletionTokens, cachedInput, time.Duration(timings.evalDuration()))
			if server.cfg.Debug && openAIResp.Usage.CompletionTokensDetails != nil {
				log.Printf("<<< UPSTREAM (openai passthrough) reasoning_tokens=%d", openAIResp.Usage.CompletionTokensDetails.ReasoningTokens)
			}
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		if _, err := w.Write(normalized); err != nil {
			log.Printf("openai chat proxy encode error | %s: %v", reqSummary, err)
		}
		return
	}

	server.proxyOpenAIStream(w, resp, reqSummary, timings)
}

func (server *Server) handleOpenAIEmbeddings(w http.ResponseWriter, r *http.Request) {
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

	// Extract the model name from the original request before rewriting.
	var origPayload map[string]any
	_ = json.Unmarshal(body, &origPayload)
	model, _ := origPayload["model"].(string)
	baseURL, apiKey, _, _, passthrough, found := server.resolveRouteForModelPassthrough(model)
	if !found {
		http.Error(w, fmt.Sprintf("model not configured: %q", model), http.StatusNotFound)
		return
	}
	apiKey = server.resolveEffectiveAPIKey(apiKey, passthrough, applogging.ExtractAPIKey(r))

	payload, err := server.rewriteRequestModel(body)
	if err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	upstream, err := http.NewRequestWithContext(r.Context(), http.MethodPost, baseURL+"/v1/embeddings", bytes.NewReader(payload))
	if err != nil {
		http.Error(w, "request error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	upstream.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		upstream.Header.Set("Authorization", "Bearer "+apiKey)
	}

	resp, err := server.requestClient.Do(upstream)
	if err != nil {
		http.Error(w, "upstream error: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer func() { _ = resp.Body.Close() }()

	copyResponseHeaders(w, resp)
	w.WriteHeader(resp.StatusCode)
	if _, err := io.Copy(w, resp.Body); err != nil {
		log.Printf("openai embeddings proxy copy error: %v", err)
	}
}
