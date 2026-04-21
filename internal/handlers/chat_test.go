package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/k0in/openai-ollama-proxy/internal/types"
)

func TestHandleChat_MethodNotAllowed(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/api/chat", nil)
	w := httptest.NewRecorder()
	server.handleChat(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", w.Code)
	}
}

func TestHandleChat_Load(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"qwen3:latest","messages":[]}`))
	w := httptest.NewRecorder()
	server.handleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Result().Header.Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}

	var got types.OllamaChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Model != "qwen3:latest" {
		t.Fatalf("Model = %q, want %q", got.Model, "qwen3:latest")
	}
	assertRFC3339Timestamp(t, got.CreatedAt)
	if got.Message.Role != "assistant" {
		t.Fatalf("Role = %q, want %q", got.Message.Role, "assistant")
	}
	if got.Message.Content != "" {
		t.Fatalf("Content = %q, want empty string", got.Message.Content)
	}
	if !got.Done {
		t.Fatal("Done should be true")
	}
	if got.DoneReason != "load" {
		t.Fatalf("DoneReason = %q, want %q", got.DoneReason, "load")
	}
}

func TestHandleChat_Unload(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"qwen3:latest","messages":[],"keep_alive":0}`))
	w := httptest.NewRecorder()
	server.handleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var got types.OllamaChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	assertRFC3339Timestamp(t, got.CreatedAt)
	if !got.Done {
		t.Fatal("Done should be true")
	}
	if got.DoneReason != "unload" {
		t.Fatalf("DoneReason = %q, want %q", got.DoneReason, "unload")
	}
	if got.Message.Role != "assistant" {
		t.Fatalf("Role = %q, want %q", got.Message.Role, "assistant")
	}
	if got.Message.Content != "" {
		t.Fatalf("Content = %q, want empty string", got.Message.Content)
	}
}

func TestHandleChat_NonStream(t *testing.T) {
	server := newTestServer()
	content := "Hello from vLLM!"
	stop := "stop"
	mockResp := types.OpenAIChatResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1700000000,
		Model:   "test-model",
		Choices: []types.OpenAIChoice{{
			Index:        0,
			Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
			FinishReason: &stop,
		}},
		Usage: &types.OpenAIUsage{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8},
	}

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("Authorization = %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(mockResp)
	}))
	defer upstream.Close()

	origURL := server.cfg.VLLMBaseURL
	origKey := server.cfg.VLLMAPIKey
	server.cfg.VLLMBaseURL = upstream.URL
	server.cfg.VLLMAPIKey = "test-key"
	defer func() {
		server.cfg.VLLMBaseURL = origURL
		server.cfg.VLLMAPIKey = origKey
	}()

	ollamaReq := `{"model":"qwen3:latest","messages":[{"role":"user","content":"Hi"}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(ollamaReq))
	w := httptest.NewRecorder()
	server.handleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Result().Header.Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}

	var got types.OllamaChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	assertRFC3339Timestamp(t, got.CreatedAt)
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
	if got.PromptEvalCount != 5 {
		t.Errorf("PromptEvalCount = %d, want 5", got.PromptEvalCount)
	}
	if got.EvalCount != 3 {
		t.Errorf("EvalCount = %d, want 3", got.EvalCount)
	}
	if got.TotalDuration <= 0 {
		t.Errorf("TotalDuration = %d, want > 0", got.TotalDuration)
	}
	if got.PromptEvalDuration < 0 {
		t.Errorf("PromptEvalDuration = %d, want >= 0", got.PromptEvalDuration)
	}
	if got.EvalDuration < 0 {
		t.Errorf("EvalDuration = %d, want >= 0", got.EvalDuration)
	}
	if got.PromptEvalDuration+got.EvalDuration > got.TotalDuration {
		t.Errorf("phase durations = %d, want <= total duration %d", got.PromptEvalDuration+got.EvalDuration, got.TotalDuration)
	}
	if got.LoadDuration != 0 {
		t.Errorf("LoadDuration = %d, want 0 when upstream load time is unknown", got.LoadDuration)
	}
}

func TestHandleChat_Stream(t *testing.T) {
	server := newTestServer()
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
		_, _ = w.Write([]byte(sseData))
	}))
	defer upstream.Close()

	origURL := server.cfg.VLLMBaseURL
	server.cfg.VLLMBaseURL = upstream.URL
	defer func() { server.cfg.VLLMBaseURL = origURL }()

	ollamaReq := `{"model":"qwen3:latest","messages":[{"role":"user","content":"Hi"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(ollamaReq))
	w := httptest.NewRecorder()
	server.handleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Result().Header.Get("Content-Type"); !strings.HasPrefix(got, "application/x-ndjson") {
		t.Fatalf("Content-Type = %q, want application/x-ndjson", got)
	}

	lines := strings.Split(strings.TrimSpace(w.Body.String()), "\n")
	if len(lines) < 2 {
		t.Fatalf("expected at least 2 NDJSON lines, got %d: %s", len(lines), w.Body.String())
	}

	var first types.OllamaChatResponse
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatal(err)
	}
	assertRFC3339Timestamp(t, first.CreatedAt)
	if first.Model != "qwen3:latest" {
		t.Fatalf("first chunk model = %q, want %q", first.Model, "qwen3:latest")
	}
	if first.Message.Role != "assistant" {
		t.Errorf("first chunk role = %q, want %q", first.Message.Role, "assistant")
	}
	if first.Message.Content != "Hello" {
		t.Errorf("first chunk content = %q, want %q", first.Message.Content, "Hello")
	}
	if first.Done {
		t.Error("first chunk should not be done")
	}

	var last types.OllamaChatResponse
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &last); err != nil {
		t.Fatal(err)
	}
	assertRFC3339Timestamp(t, last.CreatedAt)
	if last.Model != "qwen3:latest" {
		t.Fatalf("last chunk model = %q, want %q", last.Model, "qwen3:latest")
	}
	if !last.Done {
		t.Error("last chunk should be done")
	}
	if last.Message.Role != "assistant" {
		t.Errorf("last chunk role = %q, want %q", last.Message.Role, "assistant")
	}
	if last.Message.Content != "" {
		t.Errorf("last chunk content = %q, want empty string", last.Message.Content)
	}
	if last.PromptEvalCount != 5 {
		t.Errorf("PromptEvalCount = %d, want 5", last.PromptEvalCount)
	}
	if last.EvalCount != 2 {
		t.Errorf("EvalCount = %d, want 2", last.EvalCount)
	}
	if last.TotalDuration <= 0 {
		t.Errorf("TotalDuration = %d, want > 0", last.TotalDuration)
	}
	if last.PromptEvalDuration < 0 {
		t.Errorf("PromptEvalDuration = %d, want >= 0", last.PromptEvalDuration)
	}
	if last.EvalDuration < 0 {
		t.Errorf("EvalDuration = %d, want >= 0", last.EvalDuration)
	}
	if last.PromptEvalDuration+last.EvalDuration > last.TotalDuration {
		t.Errorf("phase durations = %d, want <= total duration %d", last.PromptEvalDuration+last.EvalDuration, last.TotalDuration)
	}
	if last.LoadDuration != 0 {
		t.Errorf("LoadDuration = %d, want 0 when upstream load time is unknown", last.LoadDuration)
	}
}

func TestHandleChat_Stream_ToolCalls(t *testing.T) {
	server := newTestServer()
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
		_, _ = w.Write([]byte(sseData))
	}))
	defer upstream.Close()

	origURL := server.cfg.VLLMBaseURL
	server.cfg.VLLMBaseURL = upstream.URL
	defer func() { server.cfg.VLLMBaseURL = origURL }()

	ollamaReq := `{"model":"qwen3:latest","messages":[{"role":"user","content":"Use the tool"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(ollamaReq))
	w := httptest.NewRecorder()
	server.handleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Result().Header.Get("Content-Type"); !strings.HasPrefix(got, "application/x-ndjson") {
		t.Fatalf("Content-Type = %q, want application/x-ndjson", got)
	}

	lines := strings.Split(strings.TrimSpace(w.Body.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 NDJSON lines, got %d: %s", len(lines), w.Body.String())
	}

	var toolChunk types.OllamaChatResponse
	if err := json.Unmarshal([]byte(lines[0]), &toolChunk); err != nil {
		t.Fatal(err)
	}
	assertRFC3339Timestamp(t, toolChunk.CreatedAt)
	if toolChunk.Model != "qwen3:latest" {
		t.Fatalf("tool chunk model = %q, want %q", toolChunk.Model, "qwen3:latest")
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

	var final types.OllamaChatResponse
	if err := json.Unmarshal([]byte(lines[1]), &final); err != nil {
		t.Fatal(err)
	}
	assertRFC3339Timestamp(t, final.CreatedAt)
	if final.Model != "qwen3:latest" {
		t.Fatalf("final chunk model = %q, want %q", final.Model, "qwen3:latest")
	}
	if !final.Done {
		t.Fatal("final chunk should be done")
	}
	if final.DoneReason != "stop" {
		t.Fatalf("DoneReason = %q, want %q", final.DoneReason, "stop")
	}
	if final.PromptEvalCount != 5 {
		t.Fatalf("PromptEvalCount = %d, want 5", final.PromptEvalCount)
	}
	if final.EvalCount != 2 {
		t.Fatalf("EvalCount = %d, want 2", final.EvalCount)
	}
	if final.TotalDuration <= 0 {
		t.Fatalf("TotalDuration = %d, want > 0", final.TotalDuration)
	}
	if final.PromptEvalDuration < 0 {
		t.Fatalf("PromptEvalDuration = %d, want >= 0", final.PromptEvalDuration)
	}
	if final.EvalDuration < 0 {
		t.Fatalf("EvalDuration = %d, want >= 0", final.EvalDuration)
	}
	if final.PromptEvalDuration+final.EvalDuration > final.TotalDuration {
		t.Fatalf("phase durations = %d, want <= total duration %d", final.PromptEvalDuration+final.EvalDuration, final.TotalDuration)
	}
	if final.LoadDuration != 0 {
		t.Fatalf("LoadDuration = %d, want 0 when upstream load time is unknown", final.LoadDuration)
	}
}
