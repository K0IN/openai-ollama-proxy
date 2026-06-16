// Package conformance contains integration tests that drive the proxy with the
// official Go SDKs of each supported provider (OpenAI, Anthropic, Ollama) and
// assert that the client's request is correctly translated and forwarded to the
// upstream OpenAI-compatible backend.
//
// The suite lives in its own Go module (see go.mod) so the heavy SDK
// dependencies never enter the core proxy module. Run it with:
//
//	cd tests/conformance && go test ./...
package conformance

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/k0in/openai-ollama-proxy/internal/config"
	"github.com/k0in/openai-ollama-proxy/internal/handlers"
	"github.com/k0in/openai-ollama-proxy/internal/stats"
	"github.com/k0in/openai-ollama-proxy/internal/types"
)

// localModel is the model alias clients send; upstreamModel is what the proxy
// must rewrite it to before forwarding upstream.
const (
	localModel    = "qwen3:latest"
	upstreamModel = "test-upstream-model"
)

// capturedRequest holds the decoded OpenAI request body the proxy sent upstream
// along with the raw bytes and request path, so tests can assert on exactly
// what was upstreamed.
type capturedRequest struct {
	Path string
	Raw  json.RawMessage
	Chat types.OpenAIChatRequest
	// Embed is populated for /v1/embeddings calls.
	Embed map[string]any
}

// harness wires a capturing stub upstream behind the real proxy handlers and
// exposes the proxy's base URL plus accessors for the last captured upstream
// request.
type harness struct {
	ProxyURL string

	mu       sync.Mutex
	captured []capturedRequest

	proxy    *httptest.Server
	upstream *httptest.Server
}

// newHarness starts a stub upstream and the proxy in front of it. Callers must
// defer Close.
func newHarness(t *testing.T) *harness {
	t.Helper()

	h := &harness{}
	h.upstream = httptest.NewServer(http.HandlerFunc(h.handleUpstream))

	router, err := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: h.upstream.URL,
			Models: []config.ModelMapping{
				{Upstream: upstreamModel, Local: localModel, ContextLength: 65536},
			},
		},
	}, 65536)
	if err != nil {
		h.upstream.Close()
		t.Fatalf("BuildRoutingTable: %v", err)
	}

	cfg := config.Config{
		ListenAddr:            ":0",
		ModelContextLength:    65536,
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	srv := handlers.NewWithClients(cfg, router,
		&http.Client{Timeout: 30 * time.Second},
		&http.Client{Timeout: 30 * time.Second},
		stats.New())

	h.proxy = httptest.NewServer(srv.Routes())
	h.ProxyURL = h.proxy.URL
	return h
}

func (h *harness) Close() {
	if h.proxy != nil {
		h.proxy.Close()
	}
	if h.upstream != nil {
		h.upstream.Close()
	}
}

// LastChat returns the most recent captured chat-completions request. It fails
// the test if none was captured.
func (h *harness) LastChat(t *testing.T) capturedRequest {
	t.Helper()
	h.mu.Lock()
	defer h.mu.Unlock()
	for i := len(h.captured) - 1; i >= 0; i-- {
		if h.captured[i].Path == "/v1/chat/completions" {
			return h.captured[i]
		}
	}
	t.Fatal("no chat-completions request was captured upstream")
	return capturedRequest{}
}

// LastEmbed returns the most recent captured embeddings request.
func (h *harness) LastEmbed(t *testing.T) capturedRequest {
	t.Helper()
	h.mu.Lock()
	defer h.mu.Unlock()
	for i := len(h.captured) - 1; i >= 0; i-- {
		if h.captured[i].Path == "/v1/embeddings" {
			return h.captured[i]
		}
	}
	t.Fatal("no embeddings request was captured upstream")
	return capturedRequest{}
}
