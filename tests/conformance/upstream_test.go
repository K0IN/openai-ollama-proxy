package conformance

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/k0in/openai-ollama-proxy/internal/types"
)

// handleUpstream is the stub OpenAI-compatible backend. It records every
// request body and returns minimal valid responses so the SDK clients see a
// well-formed reply. Streaming is detected via the request's "stream" flag.
func (h *harness) handleUpstream(w http.ResponseWriter, r *http.Request) {
	switch {
	case r.URL.Path == "/v1/models" && r.Method == http.MethodGet:
		h.writeModels(w)
	case r.URL.Path == "/v1/chat/completions" && r.Method == http.MethodPost:
		h.handleChatUpstream(w, r)
	case r.URL.Path == "/v1/embeddings" && r.Method == http.MethodPost:
		h.handleEmbedUpstream(w, r)
	default:
		http.Error(w, "not found: "+r.URL.Path, http.StatusNotFound)
	}
}

func (h *harness) writeModels(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(types.OpenAIModelListResponse{
		Object: "list",
		Data: []types.OpenAIModel{
			{ID: upstreamModel, Object: "model", OwnedBy: "test", MaxModelLen: 65536},
		},
	})
}

func (h *harness) handleChatUpstream(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)

	var chat types.OpenAIChatRequest
	_ = json.Unmarshal(body, &chat)

	h.mu.Lock()
	h.captured = append(h.captured, capturedRequest{
		Path: r.URL.Path,
		Raw:  json.RawMessage(body),
		Chat: chat,
	})
	h.mu.Unlock()

	if chat.Stream {
		h.writeChatStream(w)
		return
	}
	h.writeChatNonStream(w)
}

func (h *harness) writeChatNonStream(w http.ResponseWriter) {
	content := "Hello from the stub upstream."
	stop := "stop"
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(types.OpenAIChatResponse{
		ID:      "chatcmpl-stub",
		Object:  "chat.completion",
		Created: 1700000000,
		Model:   upstreamModel,
		Choices: []types.OpenAIChoice{{
			Index:        0,
			Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
			FinishReason: &stop,
		}},
		Usage: &types.OpenAIUsage{PromptTokens: 8, CompletionTokens: 5, TotalTokens: 13},
	})
}

func (h *harness) writeChatStream(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)

	chunks := []string{
		`{"id":"1","object":"chat.completion.chunk","model":"` + upstreamModel + `","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}`,
		`{"id":"1","object":"chat.completion.chunk","model":"` + upstreamModel + `","choices":[{"index":0,"delta":{"content":" world"}}]}`,
		`{"id":"1","object":"chat.completion.chunk","model":"` + upstreamModel + `","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`{"id":"1","object":"chat.completion.chunk","model":"` + upstreamModel + `","choices":[],"usage":{"prompt_tokens":8,"completion_tokens":5,"total_tokens":13}}`,
	}
	for _, c := range chunks {
		_, _ = fmt.Fprintf(w, "data: %s\n\n", c)
		if flusher != nil {
			flusher.Flush()
		}
	}
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
}

func (h *harness) handleEmbedUpstream(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)

	var embed map[string]any
	_ = json.Unmarshal(body, &embed)

	h.mu.Lock()
	h.captured = append(h.captured, capturedRequest{
		Path:  r.URL.Path,
		Raw:   json.RawMessage(body),
		Embed: embed,
	})
	h.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(types.OpenAIEmbedResponse{
		Object: "list",
		Data: []types.OpenAIEmbedData{
			{Object: "embedding", Index: 0, Embedding: []float64{0.1, 0.2, 0.3}},
		},
		Model: upstreamModel,
		Usage: &types.OpenAIUsage{PromptTokens: 4},
	})
}
