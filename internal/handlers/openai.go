package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	applogging "github.com/k0in/openai-ollama-proxy/internal/logging"
	"github.com/k0in/openai-ollama-proxy/internal/types"
)

func (server *Server) handleOpenAIModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	metadata := server.currentModelMetadata(r.Context())
	resp := types.OpenAIModelListResponse{
		Object: "list",
		Data: []types.OpenAIModel{{
			Object:      "model",
			ID:          server.cfg.ModelName,
			OwnedBy:     "openai-ollama-proxy",
			Root:        metadata.ParentModel,
			MaxModelLen: metadata.ContextLength,
		}},
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
		log.Printf(">>> UPSTREAM (openai passthrough) POST %s/v1/chat/completions (%d bytes):\n  %s", server.cfg.VLLMBaseURL, len(payload), string(applogging.RedactJSONForLog(payload)))
	}

	resp, err := server.doUpstreamChatWithRetry(r.Context(), payload)
	if err != nil {
		log.Printf("upstream openai-chat error: %v | %s", err, reqSummary)
		http.Error(w, "upstream not ready: "+err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer func() { _ = resp.Body.Close() }()

	if server.cfg.Debug {
		log.Printf("<<< UPSTREAM (openai passthrough) %d | content-type=%q", resp.StatusCode, resp.Header.Get("Content-Type"))
	}

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(resp.Body)
		log.Printf("upstream openai-chat error %d: %s | %s | sent: %s", resp.StatusCode, string(errBody), reqSummary, string(applogging.RedactJSONForLog(payload)))
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
			log.Printf("<<< UPSTREAM (openai passthrough) non-stream body (%d bytes): %s", len(respBody), string(respBody))
		}

		normalized, err := server.normalizeOpenAIJSON(respBody)
		if err != nil {
			http.Error(w, "decode error: "+err.Error(), http.StatusInternalServerError)
			return
		}

		if server.cfg.Debug {
			log.Printf("<<< RESPONSE (openai passthrough) normalized: %s", string(normalized))
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		if _, err := w.Write(normalized); err != nil {
			log.Printf("openai chat proxy encode error | %s: %v", reqSummary, err)
		}
		return
	}

	server.proxyOpenAIStream(w, resp, reqSummary)
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

	payload, err := server.rewriteRequestModel(body)
	if err != nil {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	upstream, err := http.NewRequestWithContext(r.Context(), http.MethodPost, server.cfg.VLLMBaseURL+"/v1/embeddings", bytes.NewReader(payload))
	if err != nil {
		http.Error(w, "request error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	upstream.Header.Set("Content-Type", "application/json")
	if server.cfg.VLLMAPIKey != "" {
		upstream.Header.Set("Authorization", "Bearer "+server.cfg.VLLMAPIKey)
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
