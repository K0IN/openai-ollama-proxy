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

func TestHandleRoot_Healthy(t *testing.T) {
	server := newTestServer()
	cleanup := withUpstreamHealthServer(t, server, http.StatusOK, "ok")
	defer cleanup()

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	server.handleRoot(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Body.String(); got != "Ollama is running" {
		t.Fatalf("body = %q, want %q", got, "Ollama is running")
	}
}

func TestHandleRoot_Unhealthy(t *testing.T) {
	server := newTestServer()
	cleanup := withUpstreamHealthServer(t, server, http.StatusServiceUnavailable, "loading")
	defer cleanup()

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	server.handleRoot(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", w.Code)
	}
	if got := w.Body.String(); got != "Ollama is down" {
		t.Fatalf("body = %q, want %q", got, "Ollama is down")
	}
}

func TestHandleHead_Healthy(t *testing.T) {
	server := newTestServer()
	cleanup := withUpstreamHealthServer(t, server, http.StatusOK, "ok")
	defer cleanup()

	req := httptest.NewRequest(http.MethodHead, "/", nil)
	w := httptest.NewRecorder()
	server.handleHead(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Body.String(); got != "" {
		t.Fatalf("body = %q, want empty string", got)
	}
}

func TestHandleHead_Unhealthy(t *testing.T) {
	server := newTestServer()
	cleanup := withUpstreamHealthServer(t, server, http.StatusServiceUnavailable, "loading")
	defer cleanup()

	req := httptest.NewRequest(http.MethodHead, "/", nil)
	w := httptest.NewRecorder()
	server.handleHead(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", w.Code)
	}
	if got := w.Body.String(); got != "" {
		t.Fatalf("body = %q, want empty string", got)
	}
}

func TestHandleVersion(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/api/version", nil)
	w := httptest.NewRecorder()
	server.handleVersion(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if got := resp.Header.Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}
	var got map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	version, ok := got["version"].(string)
	if !ok || version == "" {
		t.Fatalf("version = %#v, want non-empty string", got["version"])
	}
}

func TestHandleTags(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/api/tags", nil)
	w := httptest.NewRecorder()
	server.handleTags(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if got := resp.Header.Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}
	var got map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	models, ok := got["models"].([]any)
	if !ok {
		t.Fatalf("models = %#v, want array", got["models"])
	}
	if len(models) == 0 {
		t.Fatal("models should contain at least one entry")
	}
	first, ok := models[0].(map[string]any)
	if !ok {
		t.Fatalf("models[0] = %#v, want object", models[0])
	}
	for _, field := range []string{"name", "model", "modified_at", "digest"} {
		value, ok := first[field].(string)
		if !ok || value == "" {
			t.Fatalf("%s = %#v, want non-empty string", field, first[field])
		}
	}
	assertRFC3339Timestamp(t, first["modified_at"].(string))
	if _, ok := first["size"].(float64); !ok {
		t.Fatalf("size = %#v, want number", first["size"])
	}
	assertModelDetailsContract(t, first["details"])
}

func TestHandleShow(t *testing.T) {
	server := newTestServer()
	body := `{"model":"qwen3:latest"}`
	req := httptest.NewRequest(http.MethodPost, "/api/show", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleShow(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if got := resp.Header.Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}
	var got map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"modelfile", "parameters", "template"} {
		if _, ok := got[field].(string); !ok {
			t.Fatalf("%s = %#v, want string", field, got[field])
		}
	}
	assertModelDetailsContract(t, got["details"])
	modelInfo, ok := got["model_info"].(map[string]any)
	if !ok {
		t.Fatalf("model_info = %#v, want object", got["model_info"])
	}
	if len(modelInfo) == 0 {
		t.Fatal("model_info should not be empty")
	}
	capabilities, ok := got["capabilities"].([]any)
	if !ok || len(capabilities) == 0 {
		t.Fatalf("capabilities = %#v, want non-empty array", got["capabilities"])
	}
	for i, capability := range capabilities {
		if _, ok := capability.(string); !ok {
			t.Fatalf("capabilities[%d] = %#v, want string", i, capability)
		}
	}
}

func TestHandleShow_ParameterCountIsNumeric(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(types.OpenAIModelListResponse{
			Object: "list",
			Data:   []types.OpenAIModel{{ID: "Qwen3-35B-FP8", Object: "model", OwnedBy: "test", MaxModelLen: 65536}},
		})
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: upstream.URL,
			Models: []config.ModelMapping{
				{Upstream: "Qwen3-35B-FP8", Local: "qwen3:latest", ContextLength: 65536},
			},
		},
	}, 65536)

	cfg := config.Config{
		ListenAddr:            ":11434",
		ModelContextLength:    65536,
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	server := New(cfg, router, &http.Client{Timeout: 5 * time.Second}, stats.New())

	req := httptest.NewRequest(http.MethodPost, "/api/show", strings.NewReader(`{"model":"qwen3:latest"}`))
	w := httptest.NewRecorder()
	server.handleShow(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var got map[string]any
	if err := json.NewDecoder(w.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}

	details, ok := got["details"].(map[string]any)
	if !ok {
		t.Fatalf("details = %#v, want object", got["details"])
	}
	if gotSize, ok := details["parameter_size"].(string); !ok || gotSize != "35B" {
		t.Fatalf("details[parameter_size] = %#v, want %q", details["parameter_size"], "35B")
	}

	modelInfo, ok := got["model_info"].(map[string]any)
	if !ok {
		t.Fatalf("model_info = %#v, want object", got["model_info"])
	}
	if gotCount, ok := modelInfo["general.parameter_count"].(float64); !ok || gotCount != 35_000_000_000 {
		t.Fatalf("model_info[general.parameter_count] = %#v, want %d", modelInfo["general.parameter_count"], int64(35_000_000_000))
	}
}

func TestHandleShow_MethodNotAllowed(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/api/show", nil)
	w := httptest.NewRecorder()
	server.handleShow(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", w.Code)
	}
}

func TestHandleShow_VisionCapability(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(types.OpenAIModelListResponse{
			Object: "list",
			Data:   []types.OpenAIModel{{ID: "vision-model", Object: "model", OwnedBy: "test", MaxModelLen: 65536}},
		})
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: upstream.URL,
			Models: []config.ModelMapping{
				{Upstream: "vision-model", Local: "vision-model:latest", ContextLength: 65536, SupportsVision: true},
			},
		},
	}, 65536)

	cfg := config.Config{
		ListenAddr:            ":11434",
		ModelContextLength:    65536,
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	server := New(cfg, router, &http.Client{Timeout: 5 * time.Second}, stats.New())

	req := httptest.NewRequest(http.MethodPost, "/api/show", strings.NewReader(`{"model":"vision-model:latest"}`))
	w := httptest.NewRecorder()
	server.handleShow(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var got map[string]any
	if err := json.NewDecoder(w.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}

	capabilities, ok := got["capabilities"].([]any)
	if !ok {
		t.Fatalf("capabilities = %#v, want array", got["capabilities"])
	}

	hasCompletion := false
	hasTools := false
	hasVision := false
	for _, c := range capabilities {
		switch c.(string) {
		case "completion":
			hasCompletion = true
		case "tools":
			hasTools = true
		case "vision":
			hasVision = true
		}
	}
	if !hasCompletion {
		t.Error("capabilities missing 'completion'")
	}
	if !hasTools {
		t.Error("capabilities missing 'tools'")
	}
	if !hasVision {
		t.Error("capabilities missing 'vision'")
	}
}

func TestHandleShow_NoVisionCapability(t *testing.T) {
	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: "http://localhost:8000",
			Models: []config.ModelMapping{
				{Upstream: "text-model", Local: "text-model:latest", ContextLength: 32768, SupportsVision: false},
			},
		},
	}, 32768)

	cfg := config.Config{
		ListenAddr:            ":11434",
		ModelContextLength:    32768,
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	server := New(cfg, router, &http.Client{Timeout: 5 * time.Second}, stats.New())

	req := httptest.NewRequest(http.MethodPost, "/api/show", strings.NewReader(`{"model":"text-model:latest"}`))
	w := httptest.NewRecorder()
	server.handleShow(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var got map[string]any
	if err := json.NewDecoder(w.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}

	capabilities, ok := got["capabilities"].([]any)
	if !ok {
		t.Fatalf("capabilities = %#v, want array", got["capabilities"])
	}

	for _, c := range capabilities {
		if c.(string) == "vision" {
			t.Error("capabilities should NOT include 'vision' when SupportsVision=false")
		}
	}
}

func TestHandleShow_ThinkingCapability(t *testing.T) {
	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: "http://localhost:8000",
			Models: []config.ModelMapping{
				{Upstream: "thinking-model", Local: "thinking-model:latest", ContextLength: 65536, SupportsThinking: []string{"medium"}},
			},
		},
	}, 65536)

	cfg := config.Config{
		ListenAddr:            ":11434",
		ModelContextLength:    65536,
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	server := New(cfg, router, &http.Client{Timeout: 5 * time.Second}, stats.New())

	req := httptest.NewRequest(http.MethodPost, "/api/show", strings.NewReader(`{"model":"thinking-model:latest-medium"}`))
	w := httptest.NewRecorder()
	server.handleShow(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var got map[string]any
	if err := json.NewDecoder(w.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}

	capabilities, ok := got["capabilities"].([]any)
	if !ok {
		t.Fatalf("capabilities = %#v, want array", got["capabilities"])
	}

	hasCompletion := false
	hasTools := false
	hasThinking := false
	hasVision := false
	for _, c := range capabilities {
		switch c.(string) {
		case "completion":
			hasCompletion = true
		case "tools":
			hasTools = true
		case "thinking":
			hasThinking = true
		case "vision":
			hasVision = true
		}
	}
	if !hasCompletion {
		t.Error("capabilities missing 'completion'")
	}
	if !hasTools {
		t.Error("capabilities missing 'tools'")
	}
	if !hasThinking {
		t.Error("capabilities missing 'thinking' when SupportsThinking=true")
	}
	if hasVision {
		t.Error("capabilities should NOT include 'vision' when SupportsVision=false")
	}
}

func TestHandleShow_NoThinkingCapability(t *testing.T) {
	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{
			URL: "http://localhost:8000",
			Models: []config.ModelMapping{
				{Upstream: "text-model", Local: "text-model:latest", ContextLength: 32768},
			},
		},
	}, 32768)

	cfg := config.Config{
		ListenAddr:            ":11434",
		ModelContextLength:    32768,
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	server := New(cfg, router, &http.Client{Timeout: 5 * time.Second}, stats.New())

	req := httptest.NewRequest(http.MethodPost, "/api/show", strings.NewReader(`{"model":"text-model:latest"}`))
	w := httptest.NewRecorder()
	server.handleShow(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var got map[string]any
	if err := json.NewDecoder(w.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}

	capabilities, ok := got["capabilities"].([]any)
	if !ok {
		t.Fatalf("capabilities = %#v, want array", got["capabilities"])
	}

	for _, c := range capabilities {
		if c.(string) == "thinking" {
			t.Error("capabilities should NOT include 'thinking' when SupportsThinking=false")
		}
	}
}

func TestHandlePs(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/api/ps", nil)
	w := httptest.NewRecorder()
	server.handlePs(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if got := resp.Header.Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}
	var got map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	models, ok := got["models"].([]any)
	if !ok {
		t.Fatalf("models = %#v, want array", got["models"])
	}
	if len(models) == 0 {
		t.Fatal("models should contain at least one entry")
	}
	first, ok := models[0].(map[string]any)
	if !ok {
		t.Fatalf("models[0] = %#v, want object", models[0])
	}
	for _, field := range []string{"name", "model", "digest", "expires_at"} {
		value, ok := first[field].(string)
		if !ok || value == "" {
			t.Fatalf("%s = %#v, want non-empty string", field, first[field])
		}
	}
	assertRFC3339Timestamp(t, first["expires_at"].(string))
	for _, field := range []string{"size", "size_vram"} {
		if _, ok := first[field].(float64); !ok {
			t.Fatalf("%s = %#v, want number", field, first[field])
		}
	}
	assertModelDetailsContract(t, first["details"])
}
