package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/k0in/openai-ollama-proxy/internal/types"
)

func TestHandleEmbed_MethodNotAllowed(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/api/embed", nil)
	w := httptest.NewRecorder()
	server.handleEmbed(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", w.Code)
	}
}

func TestHandleEmbed_SingleInput(t *testing.T) {
	server := newTestServer()

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %q, want %q", r.Method, http.MethodPost)
		}
		if r.URL.Path != "/v1/embeddings" {
			t.Fatalf("path = %q, want %q", r.URL.Path, "/v1/embeddings")
		}

		var got types.OpenAIEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		if got.Model != server.cfg.UpstreamModel {
			t.Fatalf("upstream model = %q, want %q", got.Model, server.cfg.UpstreamModel)
		}

		var input string
		if err := json.Unmarshal(got.Input, &input); err != nil {
			t.Fatal(err)
		}
		if input != "Why is the sky blue?" {
			t.Fatalf("input = %q, want %q", input, "Why is the sky blue?")
		}
		time.Sleep(10 * time.Millisecond)

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(types.OpenAIEmbedResponse{
			Data:  []types.OpenAIEmbedData{{Embedding: []float64{0.1, 0.2, 0.3}, Index: 0}},
			Usage: &types.OpenAIUsage{PromptTokens: 8},
		})
	}))
	defer upstream.Close()

	origURL := server.cfg.UpstreamBaseURL
	server.cfg.UpstreamBaseURL = upstream.URL
	defer func() { server.cfg.UpstreamBaseURL = origURL }()

	req := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(`{"model":"all-minilm","input":"Why is the sky blue?"}`))
	w := httptest.NewRecorder()
	server.handleEmbed(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Result().Header.Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}

	var raw map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &raw); err != nil {
		t.Fatal(err)
	}
	if _, ok := raw["embeddings"].([]any); !ok {
		t.Fatalf("embeddings = %#v, want array", raw["embeddings"])
	}
	if _, ok := raw["embedding"]; ok {
		t.Fatalf("embedding should not be present on /api/embed response: %#v", raw["embedding"])
	}

	var got types.OllamaEmbedResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Model != "all-minilm" {
		t.Fatalf("Model = %q, want %q", got.Model, "all-minilm")
	}
	if len(got.Embeddings) != 1 {
		t.Fatalf("len(Embeddings) = %d, want 1", len(got.Embeddings))
	}
	if got.TotalDuration <= 0 {
		t.Fatalf("TotalDuration = %d, want > 0", got.TotalDuration)
	}
	if got.TotalDuration < int64(10*time.Millisecond) {
		t.Fatalf("TotalDuration = %d, want at least %d", got.TotalDuration, int64(10*time.Millisecond))
	}
	if got.LoadDuration != 0 {
		t.Fatalf("LoadDuration = %d, want 0 when upstream load time is unknown", got.LoadDuration)
	}
	if got.PromptEvalCount != 8 {
		t.Fatalf("PromptEvalCount = %d, want 8", got.PromptEvalCount)
	}
}

func TestHandleEmbed_MultipleInput(t *testing.T) {
	server := newTestServer()

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var got types.OpenAIEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}

		var input []string
		if err := json.Unmarshal(got.Input, &input); err != nil {
			t.Fatal(err)
		}
		if len(input) != 2 {
			t.Fatalf("len(input) = %d, want 2", len(input))
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(types.OpenAIEmbedResponse{
			Data: []types.OpenAIEmbedData{
				{Embedding: []float64{0.1, 0.2}, Index: 0},
				{Embedding: []float64{0.3, 0.4}, Index: 1},
			},
		})
	}))
	defer upstream.Close()

	origURL := server.cfg.UpstreamBaseURL
	server.cfg.UpstreamBaseURL = upstream.URL
	defer func() { server.cfg.UpstreamBaseURL = origURL }()

	req := httptest.NewRequest(http.MethodPost, "/api/embed", strings.NewReader(`{"model":"all-minilm","input":["Why is the sky blue?","Why is the grass green?"]}`))
	w := httptest.NewRecorder()
	server.handleEmbed(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var got types.OllamaEmbedResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Embeddings) != 2 {
		t.Fatalf("len(Embeddings) = %d, want 2", len(got.Embeddings))
	}
}

func TestHandleEmbeddings_DeprecatedEndpoint(t *testing.T) {
	server := newTestServer()

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var got types.OpenAIEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}

		var input string
		if err := json.Unmarshal(got.Input, &input); err != nil {
			t.Fatal(err)
		}
		if input != "Here is an article about llamas..." {
			t.Fatalf("input = %q, want %q", input, "Here is an article about llamas...")
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(types.OpenAIEmbedResponse{
			Data: []types.OpenAIEmbedData{{Embedding: []float64{0.5, 0.6, 0.7}, Index: 0}},
		})
	}))
	defer upstream.Close()

	origURL := server.cfg.UpstreamBaseURL
	server.cfg.UpstreamBaseURL = upstream.URL
	defer func() { server.cfg.UpstreamBaseURL = origURL }()

	req := httptest.NewRequest(http.MethodPost, "/api/embeddings", strings.NewReader(`{"model":"all-minilm","prompt":"Here is an article about llamas..."}`))
	w := httptest.NewRecorder()
	server.handleEmbeddings(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Result().Header.Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}

	var raw map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &raw); err != nil {
		t.Fatal(err)
	}
	if _, ok := raw["embedding"].([]any); !ok {
		t.Fatalf("embedding = %#v, want array", raw["embedding"])
	}
	if _, ok := raw["embeddings"]; ok {
		t.Fatalf("embeddings should not be present on /api/embeddings response: %#v", raw["embeddings"])
	}
}

func TestHandlePull_StreamDefault(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/api/pull", strings.NewReader(`{"model":"llama3.2"}`))
	w := httptest.NewRecorder()
	server.handlePull(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Result().Header.Get("Content-Type"); !strings.HasPrefix(got, "application/x-ndjson") {
		t.Fatalf("Content-Type = %q, want application/x-ndjson", got)
	}

	responses := decodeProgressStream(t, w.Body.String())
	if len(responses) < 5 {
		t.Fatalf("expected manifest, download progress, and 4 terminal statuses, got %d line(s): %s", len(responses), w.Body.String())
	}

	if responses[0].Status != "pulling manifest" {
		t.Fatalf("first status = %q, want %q", responses[0].Status, "pulling manifest")
	}

	download := responses[1]
	if download.Status == "" {
		t.Fatal("download status should not be empty")
	}
	if download.Digest == "" {
		t.Fatal("download digest should not be empty")
	}
	if download.Total <= 0 {
		t.Fatalf("download total = %d, want > 0", download.Total)
	}

	wantTail := []string{"verifying sha256 digest", "writing manifest", "removing any unused layers", "success"}
	tail := responses[len(responses)-len(wantTail):]
	for i, want := range wantTail {
		if tail[i].Status != want {
			t.Fatalf("tail[%d].Status = %q, want %q", i, tail[i].Status, want)
		}
	}
}

func TestHandlePull_NoStream(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/api/pull", strings.NewReader(`{"model":"llama3.2","stream":false}`))
	w := httptest.NewRecorder()
	server.handlePull(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Result().Header.Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}

	var got types.OllamaProgressResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Status != "success" {
		t.Fatalf("status = %q, want %q", got.Status, "success")
	}
}

func TestHandleGenerate_MethodNotAllowed(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodGet, "/api/generate", nil)
	w := httptest.NewRecorder()
	server.handleGenerate(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", w.Code)
	}
}

func TestHandleGenerate_Load(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"qwen3:latest"}`))
	w := httptest.NewRecorder()
	server.handleGenerate(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Result().Header.Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}

	var got types.OllamaGenerateResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Model != "qwen3:latest" {
		t.Fatalf("Model = %q, want %q", got.Model, "qwen3:latest")
	}
	assertRFC3339Timestamp(t, got.CreatedAt)
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

func TestHandleGenerate_Unload(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"qwen3:latest","keep_alive":0}`))
	w := httptest.NewRecorder()
	server.handleGenerate(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var got types.OllamaGenerateResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	assertRFC3339Timestamp(t, got.CreatedAt)
	if got.Response != "" {
		t.Fatalf("Response = %q, want empty string", got.Response)
	}
	if !got.Done {
		t.Fatal("Done should be true")
	}
	if got.DoneReason != "unload" {
		t.Fatalf("DoneReason = %q, want %q", got.DoneReason, "unload")
	}
}

func TestHandleGenerate_NonStream(t *testing.T) {
	server := newTestServer()
	content := "Hello from upstream!"
	stop := "stop"
	mockResp := types.OpenAIChatResponse{
		Choices: []types.OpenAIChoice{{
			Index:        0,
			Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
			FinishReason: &stop,
		}},
		Usage: &types.OpenAIUsage{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8},
	}

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Fatalf("Authorization = %q, want %q", r.Header.Get("Authorization"), "Bearer test-key")
		}

		var got types.OpenAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		if got.Model != server.cfg.UpstreamModel {
			t.Fatalf("upstream model = %q, want %q", got.Model, server.cfg.UpstreamModel)
		}
		if got.Stream {
			t.Fatal("upstream stream should be false")
		}
		if len(got.Messages) != 1 {
			t.Fatalf("len(Messages) = %d, want 1", len(got.Messages))
		}
		if got.Messages[0].Role != "user" {
			t.Fatalf("Messages[0].Role = %q, want %q", got.Messages[0].Role, "user")
		}

		var prompt string
		if err := json.Unmarshal(got.Messages[0].Content, &prompt); err != nil {
			t.Fatal(err)
		}
		if prompt != "Hi" {
			t.Fatalf("prompt = %q, want %q", prompt, "Hi")
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(mockResp)
	}))
	defer upstream.Close()

	origURL := server.cfg.UpstreamBaseURL
	origKey := server.cfg.UpstreamAPIKey
	server.cfg.UpstreamBaseURL = upstream.URL
	server.cfg.UpstreamAPIKey = "test-key"
	defer func() {
		server.cfg.UpstreamBaseURL = origURL
		server.cfg.UpstreamAPIKey = origKey
	}()

	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"qwen3:latest","prompt":"Hi","stream":false}`))
	w := httptest.NewRecorder()
	server.handleGenerate(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Result().Header.Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}

	var got types.OllamaGenerateResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	assertRFC3339Timestamp(t, got.CreatedAt)
	if got.Model != "qwen3:latest" {
		t.Fatalf("Model = %q, want %q", got.Model, "qwen3:latest")
	}
	if got.Response != "Hello from upstream!" {
		t.Fatalf("Response = %q, want %q", got.Response, "Hello from upstream!")
	}
	if !got.Done {
		t.Fatal("Done should be true")
	}
	if got.PromptEvalCount != 5 {
		t.Fatalf("PromptEvalCount = %d, want 5", got.PromptEvalCount)
	}
	if got.EvalCount != 3 {
		t.Fatalf("EvalCount = %d, want 3", got.EvalCount)
	}
	if got.TotalDuration <= 0 {
		t.Fatalf("TotalDuration = %d, want > 0", got.TotalDuration)
	}
	if got.PromptEvalDuration < 0 {
		t.Fatalf("PromptEvalDuration = %d, want >= 0", got.PromptEvalDuration)
	}
	if got.EvalDuration < 0 {
		t.Fatalf("EvalDuration = %d, want >= 0", got.EvalDuration)
	}
	if got.PromptEvalDuration+got.EvalDuration > got.TotalDuration {
		t.Fatalf("phase durations = %d, want <= total duration %d", got.PromptEvalDuration+got.EvalDuration, got.TotalDuration)
	}
	if got.LoadDuration != 0 {
		t.Fatalf("LoadDuration = %d, want 0 when upstream load time is unknown", got.LoadDuration)
	}
}

func TestHandleGenerate_Stream(t *testing.T) {
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

	origURL := server.cfg.UpstreamBaseURL
	server.cfg.UpstreamBaseURL = upstream.URL
	defer func() { server.cfg.UpstreamBaseURL = origURL }()

	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"qwen3:latest","prompt":"Hi","stream":true}`))
	w := httptest.NewRecorder()
	server.handleGenerate(w, req)

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

	var first types.OllamaGenerateResponse
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatal(err)
	}
	assertRFC3339Timestamp(t, first.CreatedAt)
	if first.Model != "qwen3:latest" {
		t.Fatalf("first chunk model = %q, want %q", first.Model, "qwen3:latest")
	}
	if first.Response != "Hello" {
		t.Fatalf("first chunk response = %q, want %q", first.Response, "Hello")
	}
	if first.Done {
		t.Fatal("first chunk should not be done")
	}

	var last types.OllamaGenerateResponse
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &last); err != nil {
		t.Fatal(err)
	}
	assertRFC3339Timestamp(t, last.CreatedAt)
	if !last.Done {
		t.Fatal("last chunk should be done")
	}
	if last.Response != "" {
		t.Fatalf("last chunk response = %q, want empty string", last.Response)
	}
	if last.PromptEvalCount != 5 {
		t.Fatalf("PromptEvalCount = %d, want 5", last.PromptEvalCount)
	}
	if last.EvalCount != 2 {
		t.Fatalf("EvalCount = %d, want 2", last.EvalCount)
	}
	if last.TotalDuration <= 0 {
		t.Fatalf("TotalDuration = %d, want > 0", last.TotalDuration)
	}
	if last.PromptEvalDuration < 0 {
		t.Fatalf("PromptEvalDuration = %d, want >= 0", last.PromptEvalDuration)
	}
	if last.EvalDuration < 0 {
		t.Fatalf("EvalDuration = %d, want >= 0", last.EvalDuration)
	}
	if last.PromptEvalDuration+last.EvalDuration > last.TotalDuration {
		t.Fatalf("phase durations = %d, want <= total duration %d", last.PromptEvalDuration+last.EvalDuration, last.TotalDuration)
	}
	if last.LoadDuration != 0 {
		t.Fatalf("LoadDuration = %d, want 0 when upstream load time is unknown", last.LoadDuration)
	}
}

func TestHandleCreate_StreamDefault(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/api/create", strings.NewReader(`{"model":"mario","from":"llama3.2","system":"You are Mario from Super Mario Bros."}`))
	w := httptest.NewRecorder()
	server.handleCreate(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Result().Header.Get("Content-Type"); !strings.HasPrefix(got, "application/x-ndjson") {
		t.Fatalf("Content-Type = %q, want application/x-ndjson", got)
	}

	responses := decodeProgressStream(t, w.Body.String())
	if len(responses) < 4 {
		t.Fatalf("expected model metadata, system layer, manifest, and success statuses, got %d line(s): %s", len(responses), w.Body.String())
	}

	if responses[0].Status != "reading model metadata" {
		t.Fatalf("first status = %q, want %q", responses[0].Status, "reading model metadata")
	}
	if responses[1].Status != "creating system layer" {
		t.Fatalf("second status = %q, want %q", responses[1].Status, "creating system layer")
	}

	hasLayerStatus := false
	for _, response := range responses[2 : len(responses)-2] {
		if strings.HasPrefix(response.Status, "using ") || strings.HasPrefix(response.Status, "writing layer ") {
			hasLayerStatus = true
			break
		}
	}
	if !hasLayerStatus {
		t.Fatal("expected at least one layer status between system layer creation and manifest write")
	}
	if responses[len(responses)-2].Status != "writing manifest" {
		t.Fatalf("penultimate status = %q, want %q", responses[len(responses)-2].Status, "writing manifest")
	}
	if responses[len(responses)-1].Status != "success" {
		t.Fatalf("last status = %q, want %q", responses[len(responses)-1].Status, "success")
	}
}

func TestHandleCreate_NoStream(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/api/create", strings.NewReader(`{"model":"mario","from":"llama3.2","stream":false}`))
	w := httptest.NewRecorder()
	server.handleCreate(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if got := w.Result().Header.Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}
	var got types.OllamaProgressResponse
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Status != "success" {
		t.Fatalf("Status = %q, want %q", got.Status, "success")
	}
}

func TestHandleCopy(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/api/copy", strings.NewReader(`{"source":"a","destination":"b"}`))
	w := httptest.NewRecorder()
	server.handleCopy(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
}

func TestHandleDelete(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodDelete, "/api/delete", strings.NewReader(`{"model":"test"}`))
	w := httptest.NewRecorder()
	server.handleDelete(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
}

func TestHandleBlobs_Head(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodHead, "/api/blobs/sha256:abc123", nil)
	w := httptest.NewRecorder()
	server.handleBlobs(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
}

func TestHandleBlobs_Post(t *testing.T) {
	server := newTestServer()
	req := httptest.NewRequest(http.MethodPost, "/api/blobs/sha256:abc123", strings.NewReader("data"))
	w := httptest.NewRecorder()
	server.handleBlobs(w, req)

	if w.Code != http.StatusCreated {
		t.Fatalf("status = %d, want 201", w.Code)
	}
}
