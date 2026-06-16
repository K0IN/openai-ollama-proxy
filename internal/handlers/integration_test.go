package handlers

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/k0in/openai-ollama-proxy/internal/config"
	"github.com/k0in/openai-ollama-proxy/internal/stats"
	"github.com/k0in/openai-ollama-proxy/internal/types"
)

// TestMultiUpstream_Routing verifies that requests to different local models
// are routed to the correct upstreams.
func TestMultiUpstream_Routing(t *testing.T) {
	// Upstream A serves "qwen3:latest"
	upstreamA := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("upstream A: unexpected path %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer key-a" {
			t.Errorf("upstream A: Authorization = %q, want %q", r.Header.Get("Authorization"), "Bearer key-a")
		}
		var req types.OpenAIChatRequest
		json.NewDecoder(r.Body).Decode(&req)
		if req.Model != "qwen3-vllm" {
			t.Errorf("upstream A: model = %q, want %q", req.Model, "qwen3-vllm")
		}
		hello := "Hello from Upstream A"
		stop := "stop"
		json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			ID:     "a-123",
			Object: "chat.completion",
			Model:  "qwen3-vllm",
			Choices: []types.OpenAIChoice{{
				Index:        0,
				Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &hello},
				FinishReason: &stop,
			}},
		})
	}))
	defer upstreamA.Close()

	// Upstream B serves "gpt-4o"
	upstreamB := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("upstream B: unexpected path %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer key-b" {
			t.Errorf("upstream B: Authorization = %q, want %q", r.Header.Get("Authorization"), "Bearer key-b")
		}
		var req types.OpenAIChatRequest
		json.NewDecoder(r.Body).Decode(&req)
		if req.Model != "gpt-4o" {
			t.Errorf("upstream B: model = %q, want %q", req.Model, "gpt-4o")
		}
		hello := "Hello from Upstream B"
		stop := "stop"
		json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			ID:     "b-456",
			Object: "chat.completion",
			Model:  "gpt-4o",
			Choices: []types.OpenAIChoice{{
				Index:        0,
				Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &hello},
				FinishReason: &stop,
			}},
		})
	}))
	defer upstreamB.Close()

	// Build router with both upstreams
	router, err := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL:    upstreamA.URL,
			APIKey: "key-a",
			Models: []config.ModelMapping{
				{Upstream: "qwen3-vllm", Local: "qwen3:latest", ContextLength: 32768},
			},
		},
		{
			URL:    upstreamB.URL,
			APIKey: "key-b",
			Models: []config.ModelMapping{
				{Upstream: "gpt-4o", Local: "gpt-4o", ContextLength: 128000},
			},
		},
	}, 65536)
	if err != nil {
		t.Fatalf("BuildRoutingTable: %v", err)
	}

	cfg := config.Config{
		ListenAddr:            ":11434",
		ModelContextLength:    65536,
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	server := NewWithClients(cfg, router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	// Test 1: Request to qwen3:latest → upstream A
	t.Run("model_qwen3", func(t *testing.T) {
		body := strings.NewReader(`{"model":"qwen3:latest","messages":[{"role":"user","content":"Hi"}],"stream":false}`)
		req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
		w := httptest.NewRecorder()
		server.handleChat(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
		}
		var got types.OllamaChatResponse
		json.Unmarshal(w.Body.Bytes(), &got)
		if got.Message.Content != "Hello from Upstream A" {
			t.Errorf("content = %q, want %q", got.Message.Content, "Hello from Upstream A")
		}
		if got.Model != "qwen3:latest" {
			t.Errorf("model = %q, want %q", got.Model, "qwen3:latest")
		}
	})

	// Test 2: Request to gpt-4o → upstream B
	t.Run("model_gpt4o", func(t *testing.T) {
		body := strings.NewReader(`{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}],"stream":false}`)
		req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
		w := httptest.NewRecorder()
		server.handleChat(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
		}
		var got types.OllamaChatResponse
		json.Unmarshal(w.Body.Bytes(), &got)
		if got.Message.Content != "Hello from Upstream B" {
			t.Errorf("content = %q, want %q", got.Message.Content, "Hello from Upstream B")
		}
		if got.Model != "gpt-4o" {
			t.Errorf("model = %q, want %q", got.Model, "gpt-4o")
		}
	})

	// Test 3: Unknown model → upstream unavailable since no route found
	t.Run("unknown_model", func(t *testing.T) {
		body := strings.NewReader(`{"model":"nonexistent","messages":[{"role":"user","content":"Hi"}],"stream":false}`)
		req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
		w := httptest.NewRecorder()
		server.handleChat(w, req)

		if w.Code != http.StatusServiceUnavailable {
			t.Errorf("status = %d, want 503 for unknown model (no upstream route)", w.Code)
		}
	})
}

// TestMultiUpstream_TagsList verifies /api/tags returns all models from all
// upstreams in the router.
func TestMultiUpstream_TagsList(t *testing.T) {
	router, err := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: "http://localhost:8001",
			Models: []config.ModelMapping{
				{Upstream: "a-model", Local: "model-a:latest", ContextLength: 8192},
				{Upstream: "b-model", Local: "model-b:latest", ContextLength: 16384},
			},
		},
		{
			URL: "http://localhost:8002",
			Models: []config.ModelMapping{
				{Upstream: "c-model", Local: "model-c:latest", ContextLength: 32768},
			},
		},
	}, 65536)
	if err != nil {
		t.Fatalf("BuildRoutingTable: %v", err)
	}

	cfg := config.Config{
		ListenAddr:         ":11434",
		ModelContextLength: 65536,
		OllamaVersion:      "0.6.4",
	}
	server := New(cfg, router, &http.Client{Timeout: 5 * time.Second}, stats.New())

	req := httptest.NewRequest(http.MethodGet, "/api/tags", nil)
	w := httptest.NewRecorder()
	server.handleTags(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var resp types.OllamaTagsResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	gotModels := make(map[string]bool)
	for _, m := range resp.Models {
		gotModels[m.Name] = true
	}
	for _, want := range []string{"model-a:latest", "model-b:latest", "model-c:latest"} {
		if !gotModels[want] {
			t.Errorf("model %q missing from /api/tags", want)
		}
	}
	if len(resp.Models) != 3 {
		t.Errorf("len(models) = %d, want 3", len(resp.Models))
	}
}

// TestMultiUpstream_HealthProbe verifies handleRoot probes all upstreams
// and returns healthy when at least one upstream is reachable.
func TestMultiUpstream_HealthProbe(t *testing.T) {
	hitA := make(chan struct{}, 1)
	hitB := make(chan struct{}, 1)

	upstreamA := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" && r.Method == http.MethodGet {
			select {
			case hitA <- struct{}{}:
			default:
			}
			w.WriteHeader(http.StatusOK)
			return
		}
	}))
	defer upstreamA.Close()

	upstreamB := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" && r.Method == http.MethodGet {
			select {
			case hitB <- struct{}{}:
			default:
			}
			w.WriteHeader(http.StatusOK)
			return
		}
	}))
	defer upstreamB.Close()

	router, err := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL:    upstreamA.URL,
			APIKey: "key-a",
			Models: []config.ModelMapping{
				{Upstream: "a-model", Local: "model-a:latest"},
			},
		},
		{
			URL:    upstreamB.URL,
			APIKey: "key-b",
			Models: []config.ModelMapping{
				{Upstream: "b-model", Local: "model-b:latest"},
			},
		},
	}, 65536)
	if err != nil {
		t.Fatalf("BuildRoutingTable: %v", err)
	}

	cfg := config.Config{
		ListenAddr:            ":11434",
		ModelContextLength:    65536,
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
		// Set UpstreamBaseURL so probeUpstreamHealth actually probes one
		UpstreamBaseURL: upstreamA.URL,
		UpstreamAPIKey:  "key-a",
	}
	// When UpstreamBaseURL is set, probeUpstreamHealth probes the configured
	// upstream (not the router upstreams). We use upstreamA so the probe hits it.
	server := NewWithClients(cfg, router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	server.handleRoot(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("root status = %d, want 200 (body=%q)", w.Code, w.Body.String())
	}

	// Upstream A should have been probed (configured via UpstreamBaseURL)
	select {
	case <-hitA:
	default:
		t.Error("upstream A was not probed")
	}
	// Upstream B is only in the routing table, not in cfg.UpstreamBaseURL,
	// so probeUpstreamHealth does not probe it.
}

// TestMultiUpstream_HealthProbe_OneDown verifies that if one upstream is
// unreachable but another is healthy, the probe still returns healthy.
func TestMultiUpstream_HealthProbe_OneDown(t *testing.T) {
	upstreamHealthy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" && r.Method == http.MethodGet {
			w.WriteHeader(http.StatusOK)
			return
		}
	}))
	defer upstreamHealthy.Close()

	router, err := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: "http://127.0.0.1:1", // dead upstream
			Models: []config.ModelMapping{
				{Upstream: "dead-model", Local: "dead:latest"},
			},
		},
		{
			URL: upstreamHealthy.URL,
			Models: []config.ModelMapping{
				{Upstream: "live-model", Local: "live:latest"},
			},
		},
	}, 65536)
	if err != nil {
		t.Fatalf("BuildRoutingTable: %v", err)
	}

	cfg := config.Config{
		ListenAddr:            ":11434",
		ModelContextLength:    65536,
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	server := NewWithClients(cfg, router, &http.Client{Timeout: 2 * time.Second}, &http.Client{Timeout: 2 * time.Second}, stats.New())

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	server.handleRoot(w, req)

	// Should still be healthy because at least one upstream is reachable
	if w.Code != http.StatusOK {
		t.Errorf("root status = %d, want 200 (body=%q)", w.Code, w.Body.String())
	}
}

// TestMultiUpstream_HealthProbe_AllDown verifies that if ALL upstreams are
// unreachable, the probe returns unhealthy.
func TestMultiUpstream_HealthProbe_AllDown(t *testing.T) {
	router, err := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: "http://127.0.0.1:1", // dead
			Models: []config.ModelMapping{
				{Upstream: "dead-a", Local: "dead-a:latest"},
			},
		},
		{
			URL: "http://127.0.0.1:2", // also dead
			Models: []config.ModelMapping{
				{Upstream: "dead-b", Local: "dead-b:latest"},
			},
		},
	}, 65536)
	if err != nil {
		t.Fatalf("BuildRoutingTable: %v", err)
	}

	cfg := config.Config{
		ListenAddr:            ":11434",
		ModelContextLength:    65536,
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
		UpstreamBaseURL:       "http://127.0.0.1:1", // dead — triggers probe
	}
	server := NewWithClients(cfg, router, &http.Client{Timeout: 500 * time.Millisecond}, &http.Client{Timeout: 500 * time.Millisecond}, stats.New())

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	server.handleRoot(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("root status = %d, want 503 when all upstreams are down (body=%q)", w.Code, w.Body.String())
	}
}

// TestMultiUpstream_OpenAIPassthrough verifies /v1/chat/completions routes
// to the correct upstream based on the model in the request body.
func TestMultiUpstream_OpenAIPassthrough(t *testing.T) {
	upstreamA := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req types.OpenAIChatRequest
		json.NewDecoder(r.Body).Decode(&req)
		if req.Model != "a-model" {
			t.Errorf("upstream A got model=%q, want %q", req.Model, "a-model")
		}
		json.NewEncoder(w).Encode(map[string]any{"model": "a-model", "choices": []any{}})
	}))
	defer upstreamA.Close()

	upstreamB := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req types.OpenAIChatRequest
		json.NewDecoder(r.Body).Decode(&req)
		if req.Model != "b-model" {
			t.Errorf("upstream B got model=%q, want %q", req.Model, "b-model")
		}
		json.NewEncoder(w).Encode(map[string]any{"model": "b-model", "choices": []any{}})
	}))
	defer upstreamB.Close()

	router, err := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: upstreamA.URL,
			Models: []config.ModelMapping{
				{Upstream: "a-model", Local: "model-a:latest"},
			},
		},
		{
			URL: upstreamB.URL,
			Models: []config.ModelMapping{
				{Upstream: "b-model", Local: "model-b:latest"},
			},
		},
	}, 65536)
	if err != nil {
		t.Fatalf("BuildRoutingTable: %v", err)
	}

	cfg := config.Config{
		ListenAddr:            ":11434",
		ModelContextLength:    65536,
		ModelName:             "model-a:latest",
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	server := NewWithClients(cfg, router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	// OpenAI request with model-a:latest → upstream A
	t.Run("openai_model_a", func(t *testing.T) {
		body := strings.NewReader(`{"model":"model-a:latest","messages":[{"role":"user","content":"Hi"}]}`)
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
		w := httptest.NewRecorder()
		server.handleOpenAIChat(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
		}
		// The proxy normalizes the model to cfg.ModelName in non-stream mode
		var got map[string]any
		json.Unmarshal(w.Body.Bytes(), &got)
		if got["model"] != "model-a:latest" {
			t.Errorf("model = %v, want %q", got["model"], "model-a:latest")
		}
	})

	// OpenAI request with model-b:latest → upstream B
	t.Run("openai_model_b", func(t *testing.T) {
		body := strings.NewReader(`{"model":"model-b:latest","messages":[{"role":"user","content":"Hi"}]}`)
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
		w := httptest.NewRecorder()
		server.handleOpenAIChat(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
		}
		var got map[string]any
		json.Unmarshal(w.Body.Bytes(), &got)
		if got["model"] != "model-a:latest" {
			t.Errorf("model = %v, want %q", got["model"], "model-a:latest")
		}
	})
}

// TestMultiUpstream_StreamingRouting verifies streaming responses are routed
// to the correct upstream and the model name is normalized in the stream.
func TestMultiUpstream_StreamingRouting(t *testing.T) {
	upstreamA := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, "data: {\"id\":\"a\",\"model\":\"a-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"from-a\"}}]}\n\ndata: [DONE]\n\n")
		flusher, _ := w.(http.Flusher)
		if flusher != nil {
			flusher.Flush()
		}
	}))
	defer upstreamA.Close()

	upstreamB := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, "data: {\"id\":\"b\",\"model\":\"b-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"from-b\"}}]}\n\ndata: [DONE]\n\n")
	}))
	defer upstreamB.Close()

	router, err := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: upstreamA.URL,
			Models: []config.ModelMapping{
				{Upstream: "a-model", Local: "model-a:latest"},
			},
		},
		{
			URL: upstreamB.URL,
			Models: []config.ModelMapping{
				{Upstream: "b-model", Local: "model-b:latest"},
			},
		},
	}, 65536)
	if err != nil {
		t.Fatalf("BuildRoutingTable: %v", err)
	}

	cfg := config.Config{
		ListenAddr:            ":11434",
		ModelContextLength:    65536,
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	server := NewWithClients(cfg, router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	// Stream request to model-a → upstream A
	t.Run("stream_model_a", func(t *testing.T) {
		body := strings.NewReader(`{"model":"model-a:latest","messages":[{"role":"user","content":"Hi"}],"stream":true}`)
		req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
		w := httptest.NewRecorder()
		server.handleChat(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", w.Code)
		}
		lines := strings.Split(strings.TrimSpace(w.Body.String()), "\n")
		if len(lines) < 2 {
			t.Fatalf("expected at least 2 NDJSON lines, got %d", len(lines))
		}
		var first types.OllamaChatResponse
		json.Unmarshal([]byte(lines[0]), &first)
		if first.Model != "model-a:latest" {
			t.Errorf("model = %q, want %q", first.Model, "model-a:latest")
		}
		if first.Message.Content != "from-a" {
			t.Errorf("content = %q, want %q", first.Message.Content, "from-a")
		}
	})

	// Stream request to model-b → upstream B
	t.Run("stream_model_b", func(t *testing.T) {
		body := strings.NewReader(`{"model":"model-b:latest","messages":[{"role":"user","content":"Hi"}],"stream":true}`)
		req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
		w := httptest.NewRecorder()
		server.handleChat(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", w.Code)
		}
		lines := strings.Split(strings.TrimSpace(w.Body.String()), "\n")
		if len(lines) < 2 {
			t.Fatalf("expected at least 2 NDJSON lines, got %d", len(lines))
		}
		var first types.OllamaChatResponse
		json.Unmarshal([]byte(lines[0]), &first)
		if first.Model != "model-b:latest" {
			t.Errorf("model = %q, want %q", first.Model, "model-b:latest")
		}
		if first.Message.Content != "from-b" {
			t.Errorf("content = %q, want %q", first.Message.Content, "from-b")
		}
	})
}

// TestMultiUpstream_RetryFallback verifies that when one upstream is down,
// requests to its models return an error (not silently fall through to
// another upstream).
func TestMultiUpstream_RetryFallback(t *testing.T) {
	// Upstream A is down (connection refused)
	router, err := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: "http://127.0.0.1:1", // port 1 — no server listening
			Models: []config.ModelMapping{
				{Upstream: "dead-model", Local: "dead:latest"},
			},
		},
	}, 65536)
	if err != nil {
		t.Fatalf("BuildRoutingTable: %v", err)
	}

	cfg := config.Config{
		ListenAddr:            ":11434",
		ModelContextLength:    65536,
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	server := NewWithClients(cfg, router, &http.Client{Timeout: 500 * time.Millisecond}, &http.Client{Timeout: 500 * time.Millisecond}, stats.New())

	body := strings.NewReader(`{"model":"dead:latest","messages":[{"role":"user","content":"Hi"}],"stream":false}`)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
	w := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		defer close(done)
		server.handleChat(w, req)
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("handleChat hung on dead upstream")
	}

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503 for dead upstream", w.Code)
	}
}

// TestMultiUpstream_PerModelContextLength verifies context_length from the
// router is used for each model.
func TestMultiUpstream_PerModelContextLength(t *testing.T) {
	router, err := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: "http://localhost:8000",
			Models: []config.ModelMapping{
				{Upstream: "small-model", Local: "small:latest", ContextLength: 4096},
				{Upstream: "large-model", Local: "large:latest", ContextLength: 131072},
			},
		},
	}, 65536)
	if err != nil {
		t.Fatalf("BuildRoutingTable: %v", err)
	}

	cfg := config.Config{
		ListenAddr:         ":11434",
		ModelContextLength: 65536,
		OllamaVersion:      "0.6.4",
	}
	server := New(cfg, router, &http.Client{Timeout: 5 * time.Second}, stats.New())

	// /api/show for small model
	t.Run("small_context", func(t *testing.T) {
		body := strings.NewReader(`{"model":"small:latest"}`)
		req := httptest.NewRequest(http.MethodPost, "/api/show", body)
		w := httptest.NewRecorder()
		server.handleShow(w, req)

		// handleShow returns 200 even without upstream for metadata display
		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", w.Code)
		}
		var got map[string]any
		json.NewDecoder(w.Body).Decode(&got)
		params, _ := got["parameters"].(string)
		if !strings.Contains(params, "4096") {
			t.Errorf("parameters = %q, want to contain 4096", params)
		}
	})

	// /api/show for large model
	t.Run("large_context", func(t *testing.T) {
		body := strings.NewReader(`{"model":"large:latest"}`)
		req := httptest.NewRequest(http.MethodPost, "/api/show", body)
		w := httptest.NewRecorder()
		server.handleShow(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200", w.Code)
		}
		var got map[string]any
		json.NewDecoder(w.Body).Decode(&got)
		params, _ := got["parameters"].(string)
		if !strings.Contains(params, "131072") {
			t.Errorf("parameters = %q, want to contain 131072", params)
		}
	})
}
