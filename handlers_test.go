package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func withVLLMHealthServer(t *testing.T, statusCode int, body string) func() {
	t.Helper()

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Fatalf("method = %q, want %q", r.Method, http.MethodGet)
		}
		if r.URL.Path != "/health" {
			t.Fatalf("path = %q, want %q", r.URL.Path, "/health")
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Fatalf("authorization = %q, want %q", r.Header.Get("Authorization"), "Bearer test-key")
		}

		w.WriteHeader(statusCode)
		if body != "" {
			_, _ = w.Write([]byte(body))
		}
	}))

	origURL := cfg.VLLMBaseURL
	origKey := cfg.VLLMAPIKey
	cfg.VLLMBaseURL = upstream.URL
	cfg.VLLMAPIKey = "test-key"

	return func() {
		cfg.VLLMBaseURL = origURL
		cfg.VLLMAPIKey = origKey
		upstream.Close()
	}
}

func TestHandleRoot_Healthy(t *testing.T) {
	cleanup := withVLLMHealthServer(t, http.StatusOK, "ok")
	defer cleanup()

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	handleRoot(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Body.String(); got != "Ollama is running" {
		t.Fatalf("body = %q, want %q", got, "Ollama is running")
	}
}

func TestHandleRoot_Unhealthy(t *testing.T) {
	cleanup := withVLLMHealthServer(t, http.StatusServiceUnavailable, "loading")
	defer cleanup()

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	handleRoot(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", w.Code)
	}
	if got := w.Body.String(); got != "Ollama is down" {
		t.Fatalf("body = %q, want %q", got, "Ollama is down")
	}
}

func TestHandleHead_Healthy(t *testing.T) {
	cleanup := withVLLMHealthServer(t, http.StatusOK, "ok")
	defer cleanup()

	req := httptest.NewRequest(http.MethodHead, "/", nil)
	w := httptest.NewRecorder()
	handleHead(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Body.String(); got != "" {
		t.Fatalf("body = %q, want empty string", got)
	}
}

func TestHandleHead_Unhealthy(t *testing.T) {
	cleanup := withVLLMHealthServer(t, http.StatusServiceUnavailable, "loading")
	defer cleanup()

	req := httptest.NewRequest(http.MethodHead, "/", nil)
	w := httptest.NewRecorder()
	handleHead(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", w.Code)
	}
	if got := w.Body.String(); got != "" {
		t.Fatalf("body = %q, want empty string", got)
	}
}

func TestHandleVersion(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/api/version", nil)
	w := httptest.NewRecorder()
	handleVersion(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
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
	req := httptest.NewRequest(http.MethodGet, "/api/tags", nil)
	w := httptest.NewRecorder()
	handleTags(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
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
	if _, ok := first["size"].(float64); !ok {
		t.Fatalf("size = %#v, want number", first["size"])
	}
	if _, ok := first["details"].(map[string]any); !ok {
		t.Fatalf("details = %#v, want object", first["details"])
	}
}

func TestHandleShow(t *testing.T) {
	body := `{"model":"qwen3:latest"}`
	req := httptest.NewRequest(http.MethodPost, "/api/show", strings.NewReader(body))
	w := httptest.NewRecorder()
	handleShow(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
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
	if _, ok := got["details"].(map[string]any); !ok {
		t.Fatalf("details = %#v, want object", got["details"])
	}
	if _, ok := got["model_info"].(map[string]any); !ok {
		t.Fatalf("model_info = %#v, want object", got["model_info"])
	}
	capabilities, ok := got["capabilities"].([]any)
	if !ok || len(capabilities) == 0 {
		t.Fatalf("capabilities = %#v, want non-empty array", got["capabilities"])
	}
}

func TestHandleShow_MethodNotAllowed(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/api/show", nil)
	w := httptest.NewRecorder()
	handleShow(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", w.Code)
	}
}

func TestHandlePs(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/api/ps", nil)
	w := httptest.NewRecorder()
	handlePs(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
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
	for _, field := range []string{"size", "size_vram"} {
		if _, ok := first[field].(float64); !ok {
			t.Fatalf("%s = %#v, want number", field, first[field])
		}
	}
	if _, ok := first["details"].(map[string]any); !ok {
		t.Fatalf("details = %#v, want object", first["details"])
	}
}

func TestHandleChat_MethodNotAllowed(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/api/chat", nil)
	w := httptest.NewRecorder()
	handleChat(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", w.Code)
	}
}

// TestHandleChat_NonStream tests the full proxy path with a mock upstream.
func TestHandleChat_NonStream(t *testing.T) {
	content := "Hello from vLLM!"
	stop := "stop"
	mockResp := OpenAIChatResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1700000000,
		Model:   "test-model",
		Choices: []OpenAIChoice{
			{
				Index:        0,
				Message:      &OpenAIRespMsg{Role: "assistant", Content: &content},
				FinishReason: &stop,
			},
		},
		Usage: &OpenAIUsage{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8},
	}

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("Authorization = %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(mockResp)
	}))
	defer upstream.Close()

	origURL := cfg.VLLMBaseURL
	origKey := cfg.VLLMAPIKey
	cfg.VLLMBaseURL = upstream.URL
	cfg.VLLMAPIKey = "test-key"
	defer func() {
		cfg.VLLMBaseURL = origURL
		cfg.VLLMAPIKey = origKey
	}()

	ollamaReq := `{"model":"qwen3:latest","messages":[{"role":"user","content":"Hi"}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(ollamaReq))
	w := httptest.NewRecorder()
	handleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var got OllamaChatResponse
	if err := json.NewDecoder(w.Result().Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got.Message.Role != "assistant" {
		t.Errorf("Role = %q, want %q", got.Message.Role, "assistant")
	}
	if got.Message.Content != "Hello from vLLM!" {
		t.Errorf("Content = %q, want %q", got.Message.Content, "Hello from vLLM!")
	}
	if !got.Done {
		t.Error("Done should be true")
	}
	if got.Model != "qwen3:latest" {
		t.Errorf("Model = %q, want %q", got.Model, "qwen3:latest")
	}
}

// TestHandleChat_Stream tests streaming with a mock SSE upstream.
func TestHandleChat_Stream(t *testing.T) {
	sseData := strings.Join([]string{
		`data: {"id":"1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"delta":{"content":" world"}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"finish_reason":"stop","delta":{}}]}`,
		``,
		`data: {"id":"1","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(sseData))
	}))
	defer upstream.Close()

	origURL := cfg.VLLMBaseURL
	cfg.VLLMBaseURL = upstream.URL
	defer func() { cfg.VLLMBaseURL = origURL }()

	ollamaReq := `{"model":"qwen3:latest","messages":[{"role":"user","content":"Hi"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(ollamaReq))
	w := httptest.NewRecorder()
	handleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	// Parse NDJSON lines
	lines := strings.Split(strings.TrimSpace(w.Body.String()), "\n")
	if len(lines) < 2 {
		t.Fatalf("expected at least 2 NDJSON lines, got %d: %s", len(lines), w.Body.String())
	}

	// First chunk should have content "Hello"
	var first OllamaChatResponse
	json.Unmarshal([]byte(lines[0]), &first)
	if first.Message.Role != "assistant" {
		t.Errorf("first chunk role = %q, want %q", first.Message.Role, "assistant")
	}
	if first.Message.Content != "Hello" {
		t.Errorf("first chunk content = %q, want %q", first.Message.Content, "Hello")
	}
	if first.Done {
		t.Error("first chunk should not be done")
	}

	// Last line should be the final done message
	var last OllamaChatResponse
	json.Unmarshal([]byte(lines[len(lines)-1]), &last)
	if !last.Done {
		t.Error("last chunk should be done")
	}
	if last.Message.Role != "assistant" {
		t.Errorf("last chunk role = %q, want %q", last.Message.Role, "assistant")
	}
	if last.Message.Content != "" {
		t.Errorf("last chunk content = %q, want empty string", last.Message.Content)
	}
}

func TestHandleChat_Stream_ToolCalls(t *testing.T) {
	sseData := strings.Join([]string{
		`data: {"id":"1","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{"}}]}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"city\":\"Tokyo\""}}]}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]}}]}`,
		``,
		`data: {"id":"1","choices":[{"index":0,"finish_reason":"tool_calls","delta":{}}]}`,
		``,
		`data: {"id":"1","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(sseData))
	}))
	defer upstream.Close()

	origURL := cfg.VLLMBaseURL
	cfg.VLLMBaseURL = upstream.URL
	defer func() { cfg.VLLMBaseURL = origURL }()

	ollamaReq := `{"model":"qwen3:latest","messages":[{"role":"user","content":"Use the tool"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(ollamaReq))
	w := httptest.NewRecorder()
	handleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	lines := strings.Split(strings.TrimSpace(w.Body.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 NDJSON lines, got %d: %s", len(lines), w.Body.String())
	}

	var toolChunk OllamaChatResponse
	if err := json.Unmarshal([]byte(lines[0]), &toolChunk); err != nil {
		t.Fatal(err)
	}
	if toolChunk.Done {
		t.Fatal("tool chunk should not be done")
	}
	if toolChunk.Message.Content != "" {
		t.Fatalf("tool chunk content = %q, want empty string", toolChunk.Message.Content)
	}
	if len(toolChunk.Message.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(toolChunk.Message.ToolCalls))
	}
	if toolChunk.Message.ToolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("tool name = %q, want %q", toolChunk.Message.ToolCalls[0].Function.Name, "get_weather")
	}
	var args map[string]string
	if err := json.Unmarshal(toolChunk.Message.ToolCalls[0].Function.Arguments, &args); err != nil {
		t.Fatal(err)
	}
	if args["city"] != "Tokyo" {
		t.Fatalf("city = %q, want %q", args["city"], "Tokyo")
	}

	var final OllamaChatResponse
	if err := json.Unmarshal([]byte(lines[1]), &final); err != nil {
		t.Fatal(err)
	}
	if !final.Done {
		t.Fatal("final chunk should be done")
	}
	if final.DoneReason != "stop" {
		t.Fatalf("DoneReason = %q, want %q", final.DoneReason, "stop")
	}
	if final.EvalCount != 2 {
		t.Fatalf("EvalCount = %d, want 2", final.EvalCount)
	}
}

func TestHandleEmbed_MethodNotAllowed(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/api/embed", nil)
	w := httptest.NewRecorder()
	handleEmbed(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", w.Code)
	}
}

func TestHandlePull_StreamDefault(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/api/pull", strings.NewReader(`{"model":"llama3.2"}`))
	w := httptest.NewRecorder()
	handlePull(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	lines := strings.Split(strings.TrimSpace(w.Body.String()), "\n")
	if len(lines) < 2 {
		t.Fatalf("expected streamed progress responses, got %d line(s): %s", len(lines), w.Body.String())
	}

	var first OllamaProgressResponse
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatal(err)
	}
	if first.Status != "pulling manifest" {
		t.Fatalf("first status = %q, want %q", first.Status, "pulling manifest")
	}

	var last OllamaProgressResponse
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &last); err != nil {
		t.Fatal(err)
	}
	if last.Status != "success" {
		t.Fatalf("last status = %q, want %q", last.Status, "success")
	}
}

func TestHandlePull_NoStream(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/api/pull", strings.NewReader(`{"model":"llama3.2","stream":false}`))
	w := httptest.NewRecorder()
	handlePull(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var got OllamaProgressResponse
	if err := json.NewDecoder(w.Result().Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got.Status != "success" {
		t.Fatalf("status = %q, want %q", got.Status, "success")
	}
}

func TestHandleGenerate_Load(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"qwen3:latest"}`))
	w := httptest.NewRecorder()
	handleGenerate(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var got OllamaGenerateResponse
	if err := json.NewDecoder(w.Result().Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if !got.Done {
		t.Fatal("Done should be true")
	}
	if got.Response != "" {
		t.Fatalf("Response = %q, want empty string", got.Response)
	}
	if got.DoneReason != "" {
		t.Fatalf("DoneReason = %q, want empty string", got.DoneReason)
	}
}

func TestHandleCreate_StreamDefault(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/api/create", strings.NewReader(`{"model":"mario","from":"llama3.2"}`))
	w := httptest.NewRecorder()
	handleCreate(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	lines := strings.Split(strings.TrimSpace(w.Body.String()), "\n")
	if len(lines) < 2 {
		t.Fatalf("expected streamed progress responses, got %d line(s): %s", len(lines), w.Body.String())
	}

	var first OllamaProgressResponse
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatal(err)
	}
	if first.Status != "reading model metadata" {
		t.Fatalf("first status = %q, want %q", first.Status, "reading model metadata")
	}

	var last OllamaProgressResponse
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &last); err != nil {
		t.Fatal(err)
	}
	if last.Status != "success" {
		t.Fatalf("last status = %q, want %q", last.Status, "success")
	}
}

func TestHandleCreate_NoStream(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/api/create", strings.NewReader(`{"model":"mario","from":"llama3.2","stream":false}`))
	w := httptest.NewRecorder()
	handleCreate(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	var got OllamaProgressResponse
	if err := json.NewDecoder(w.Result().Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got.Status != "success" {
		t.Fatalf("Status = %q, want %q", got.Status, "success")
	}
}

func TestHandleCopy(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/api/copy", strings.NewReader(`{"source":"a","destination":"b"}`))
	w := httptest.NewRecorder()
	handleCopy(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
}

func TestHandleDelete(t *testing.T) {
	req := httptest.NewRequest(http.MethodDelete, "/api/delete", strings.NewReader(`{"model":"test"}`))
	w := httptest.NewRecorder()
	handleDelete(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
}

func TestHandleBlobs_Head(t *testing.T) {
	req := httptest.NewRequest(http.MethodHead, "/api/blobs/sha256:abc123", nil)
	w := httptest.NewRecorder()
	handleBlobs(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
}

func TestHandleBlobs_Post(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/api/blobs/sha256:abc123", strings.NewReader("data"))
	w := httptest.NewRecorder()
	handleBlobs(w, req)

	if w.Code != http.StatusCreated {
		t.Fatalf("status = %d, want 201", w.Code)
	}
}
