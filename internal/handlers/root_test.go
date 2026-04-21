package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHandleRoot_Healthy(t *testing.T) {
	server := newTestServer()
	cleanup := withVLLMHealthServer(t, server, http.StatusOK, "ok")
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
	cleanup := withVLLMHealthServer(t, server, http.StatusServiceUnavailable, "loading")
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
	cleanup := withVLLMHealthServer(t, server, http.StatusOK, "ok")
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
	cleanup := withVLLMHealthServer(t, server, http.StatusServiceUnavailable, "loading")
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
	server := newTestServer()
	server.cfg.VLLMModel = "Qwen3-35B-FP8"

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
