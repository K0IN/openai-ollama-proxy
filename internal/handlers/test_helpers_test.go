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
	cfg := config.Config{
		ListenAddr:            ":11434",
		UpstreamBaseURL:       "http://127.0.0.1:0",
		UpstreamAPIKey:        "",
		UpstreamModel:         "test-model",
		ModelName:             "generic:latest",
		ModelContextLength:    65536,
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   2 * time.Second,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	router, err := config.BuildRoutingTable(nil, cfg.ModelContextLength)
	if err != nil {
		// Legacy flat-config tests: create an empty routing table.
		router = &config.RoutingTable{}
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
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Fatalf("authorization = %q, want %q", r.Header.Get("Authorization"), "Bearer test-key")
		}

		w.WriteHeader(statusCode)
		if body != "" {
			_, _ = w.Write([]byte(body))
		}
	}))

	origURL := server.cfg.UpstreamBaseURL
	origKey := server.cfg.UpstreamAPIKey
	server.cfg.UpstreamBaseURL = upstream.URL
	server.cfg.UpstreamAPIKey = "test-key"

	return func() {
		server.cfg.UpstreamBaseURL = origURL
		server.cfg.UpstreamAPIKey = origKey
		upstream.Close()
	}
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
