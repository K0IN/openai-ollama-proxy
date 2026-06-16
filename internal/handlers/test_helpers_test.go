package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/k0in/openai-ollama-proxy/internal/config"
	"github.com/k0in/openai-ollama-proxy/internal/stats"
	"github.com/k0in/openai-ollama-proxy/internal/types"
)

func newTestServer() *Server {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(types.OpenAIModelListResponse{
			Object: "list",
			Data:   []types.OpenAIModel{{ID: "test-model", Object: "model", OwnedBy: "test", MaxModelLen: 65536}},
		})
	}))

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: upstream.URL,
			Models: []config.ModelMapping{
				{Upstream: "test-model", Local: "qwen3:latest", ContextLength: 65536},
			},
		},
	}, 65536)

	_ = upstream // close after server is created — the router holds the URL
	// We need the upstream to stay alive for the test, so we leave it running.
	// The server will use upstream.URL from the routing table.

	cfg := config.Config{
		ListenAddr:            ":11434",
		ModelContextLength:    65536,
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	return New(cfg, router, &http.Client{Timeout: 5 * time.Second}, stats.New())
}

func withUpstreamHealthServer(t *testing.T, server *Server, statusCode int, body string) func() {
	t.Helper()

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Fatalf("method = %q, want %q", r.Method, http.MethodGet)
		}
		if r.URL.Path != "/v1/models" {
			t.Fatalf("path = %q, want %q", r.URL.Path, "/v1/models")
		}

		w.WriteHeader(statusCode)
		if body != "" {
			_, _ = w.Write([]byte(body))
		}
	}))

	// Replace the router with one that points to this new upstream.
	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL:    upstream.URL,
			APIKey: "test-key",
			Models: []config.ModelMapping{
				{Upstream: "test-model", Local: "qwen3:latest", ContextLength: 65536},
			},
		},
	}, 65536)
	origRouter := server.router
	server.router = router

	return func() {
		server.router = origRouter
		upstream.Close()
	}
}

// upstreamRouter creates a RoutingTable pointing to the given upstream URL.
// Useful for test helpers that need to route requests to a test upstream.
func upstreamRouter(upstreamURL, apiKey string) *config.RoutingTable {
	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL:    upstreamURL,
			APIKey: apiKey,
			Models: []config.ModelMapping{
				{Upstream: "test-model", Local: "qwen3:latest", ContextLength: 65536},
			},
		},
	}, 65536)
	return router
}

func assertRFC3339Timestamp(t *testing.T, value string) {
	t.Helper()

	if value == "" {
		t.Fatal("timestamp should not be empty")
	}
	if _, err := time.Parse(time.RFC3339Nano, value); err != nil {
		t.Fatalf("timestamp %q is not valid RFC3339Nano: %v", value, err)
	}
}

func assertModelDetailsContract(t *testing.T, value any) {
	t.Helper()

	details, ok := value.(map[string]any)
	if !ok {
		t.Fatalf("details = %#v, want object", value)
	}

	for _, field := range []string{"parent_model", "format", "family", "parameter_size", "quantization_level"} {
		if _, ok := details[field].(string); !ok {
			t.Fatalf("details[%q] = %#v, want string", field, details[field])
		}
	}

	families, ok := details["families"].([]any)
	if !ok || len(families) == 0 {
		t.Fatalf("details[%q] = %#v, want non-empty array", "families", details["families"])
	}
	for i, family := range families {
		if _, ok := family.(string); !ok {
			t.Fatalf("details[%q][%d] = %#v, want string", "families", i, family)
		}
	}
}

func decodeProgressStream(t *testing.T, body string) []types.OllamaProgressResponse {
	t.Helper()

	lines := strings.Split(strings.TrimSpace(body), "\n")
	if len(lines) == 1 && lines[0] == "" {
		t.Fatal("expected streamed progress responses, got empty body")
	}

	responses := make([]types.OllamaProgressResponse, len(lines))
	for i, line := range lines {
		if err := json.Unmarshal([]byte(line), &responses[i]); err != nil {
			t.Fatalf("decode progress line %d: %v", i, err)
		}
	}

	return responses
}
