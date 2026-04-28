package handlers

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestIsZeroKeepAlive(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want bool
	}{
		{"empty", ``, false},
		{"null literal", `null`, false},
		{"int zero", `0`, true},
		{"int nonzero", `300`, false},
		{"float zero", `0.0`, true},
		{"float nonzero", `0.5`, false},
		{"string zero", `"0"`, true},
		{"string nonzero", `"300"`, false},
		{"duration zero seconds", `"0s"`, true},
		{"duration zero milliseconds", `"0ms"`, true},
		{"duration nonzero", `"5m"`, false},
		{"garbage string", `"abc"`, false},
		{"garbage payload", `{"x":1}`, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := isZeroKeepAlive(json.RawMessage(c.raw))
			if got != c.want {
				t.Errorf("isZeroKeepAlive(%q) = %v, want %v", c.raw, got, c.want)
			}
		})
	}
}

func TestApplyModelNameHints_Families(t *testing.T) {
	cases := map[string]string{
		"meta-llama/Llama-3.1-8B-Instruct":    "llama3",
		"meta-llama/Llama-2-7b":               "llama2",
		"mistralai/Mistral-7B-Instruct":       "mistral",
		"mistralai/Mixtral-8x7B-Instruct":     "mixtral",
		"google/gemma-2-9b":                   "gemma",
		"microsoft/Phi-3-mini":                "phi",
		"deepseek-ai/DeepSeek-V3":             "deepseek",
		"openai/gpt-oss-120b":                 "gpt-oss",
		"Qwen/Qwen2.5-72B":                    "qwen2",
		"Qwen/Qwen3.6-27B":                    "qwen3",
		"tiiuae/falcon-40b":                   "falcon",
		"ibm-granite/granite-3.0-8b-instruct": "granite",
		"CohereForAI/c4ai-command-r-plus":     "command-r",
	}
	for name, wantFamily := range cases {
		md := modelMetadata{Family: "transformer"}
		applyModelNameHints(&md, name)
		if md.Family != wantFamily {
			t.Errorf("name=%q family=%q, want %q", name, md.Family, wantFamily)
		}
	}
}

func TestApplyModelNameHints_Quantizations(t *testing.T) {
	cases := map[string]string{
		"Qwen3.6-27B-AWQ":   "AWQ-4bit",
		"Qwen3.6-27B-FP8":   "FP8",
		"Qwen3.6-27B-NVFP4": "NVFP4",
		"Qwen-7B-GPTQ":      "GPTQ",
		"some-model-INT8":   "INT8",
		"some-model-INT4":   "INT4",
		"llama-3-Q4_K_M":    "Q4_K_M",
		"llama-3-Q8_0":      "Q8_0",
		"llama-3-Q5_K_M":    "Q5_K_M",
		"llama-3-Q4_0":      "Q4_0",
		"llama-3-bf16":      "BF16",
		"llama-3-fp16":      "FP16",
	}
	for name, want := range cases {
		md := modelMetadata{Quantization: "unknown"}
		applyModelNameHints(&md, name)
		if md.Quantization != want {
			t.Errorf("name=%q quantization=%q, want %q", name, md.Quantization, want)
		}
	}
}

func TestApplyModelNameHints_Format(t *testing.T) {
	cases := map[string]string{
		"foo-gguf":        "gguf",
		"foo-bar-AWQ":     "safetensors",
		"foo.safetensors": "safetensors",
		"plain-name":      "",
	}
	for name, want := range cases {
		md := modelMetadata{}
		applyModelNameHints(&md, name)
		if md.Format != want {
			t.Errorf("name=%q format=%q, want %q", name, md.Format, want)
		}
	}
}

// TestHandleChat_StreamLineExceedsBuffer asserts that an upstream SSE line
// over 10 MiB is reported via scanner.Err() and the proxy still emits a final
// "done" sentinel rather than hanging or crashing.
func TestHandleChat_StreamLineExceedsBuffer(t *testing.T) {
	// 11 MiB of "a" then \n — over the 10 MiB scanner limit.
	huge := strings.Repeat("a", 11*1024*1024)
	ssePayload := "data: {\"x\":\"" + huge + "\"}\n\n"

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, ssePayload)
	}))
	defer upstream.Close()

	server := newTestServer()
	server.cfg.VLLMBaseURL = upstream.URL

	body := strings.NewReader(`{"model":"qwen3:latest","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
	rr := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		defer close(done)
		server.handleChat(rr, req)
	}()

	select {
	case <-done:
	case <-time.After(15 * time.Second):
		t.Fatal("handleChat hung on oversized SSE line")
	}

	out := rr.Body.String()
	if out == "" {
		t.Fatal("expected at least a final done chunk, got empty response")
	}
	// Last non-empty line must be a valid Ollama chat response with done=true.
	lines := strings.Split(strings.TrimSpace(out), "\n")
	var final map[string]any
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &final); err != nil {
		t.Fatalf("final line not JSON: %v | %q", err, lines[len(lines)-1])
	}
	if final["done"] != true {
		t.Errorf("final chunk done=%v, want true", final["done"])
	}
}

// TestHandleChat_MalformedJSONInStream verifies invalid SSE payloads are
// skipped without aborting the stream and the final done chunk is emitted.
func TestHandleChat_MalformedJSONInStream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "data: {not json}\n\ndata: [DONE]\n\n")
	}))
	defer upstream.Close()

	server := newTestServer()
	server.cfg.VLLMBaseURL = upstream.URL

	body := strings.NewReader(`{"model":"qwen3:latest","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
	rr := httptest.NewRecorder()
	server.handleChat(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rr.Code)
	}
	out := strings.TrimSpace(rr.Body.String())
	if out == "" {
		t.Fatal("expected at least one chunk (final done) on malformed stream")
	}
	lines := strings.Split(out, "\n")
	var final map[string]any
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &final); err != nil {
		t.Fatalf("final line not JSON: %v", err)
	}
	if final["done"] != true {
		t.Errorf("final.done = %v, want true", final["done"])
	}
}

// TestHandleChat_ClientDisconnectMidStream ensures cancelling the request
// context aborts the stream loop without panic and without leaking the
// upstream connection.
func TestHandleChat_ClientDisconnectMidStream(t *testing.T) {
	upstreamDone := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		w.WriteHeader(http.StatusOK)
		// Push a small valid chunk, then block until the proxy closes the conn.
		_, _ = io.WriteString(w, "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n")
		if flusher != nil {
			flusher.Flush()
		}
		<-r.Context().Done()
		close(upstreamDone)
	}))
	defer upstream.Close()

	server := newTestServer()
	server.cfg.VLLMBaseURL = upstream.URL

	ctx, cancel := context.WithCancel(context.Background())
	body := strings.NewReader(`{"model":"qwen3:latest","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", body).WithContext(ctx)
	rr := httptest.NewRecorder()

	handlerDone := make(chan struct{})
	go func() {
		defer close(handlerDone)
		server.handleChat(rr, req)
	}()

	// Give the handler a moment to receive the first chunk, then cancel.
	time.Sleep(100 * time.Millisecond)
	cancel()

	select {
	case <-handlerDone:
	case <-time.After(5 * time.Second):
		t.Fatal("handler did not return after client disconnect")
	}
	select {
	case <-upstreamDone:
	case <-time.After(5 * time.Second):
		t.Fatal("upstream connection was not closed after client disconnect")
	}
}

// TestHandleChat_UpstreamTimeout: upstream sleeps past stream timeout; the
// proxy must abort and surface a 5xx without hanging.
func TestHandleChat_UpstreamTimeout(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(2 * time.Second):
		}
	}))
	defer upstream.Close()

	server := newTestServer()
	server.cfg.VLLMBaseURL = upstream.URL
	server.client = &http.Client{Timeout: 200 * time.Millisecond}

	body := strings.NewReader(`{"model":"qwen3:latest","stream":false,"messages":[{"role":"user","content":"hi"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/api/chat", body)
	rr := httptest.NewRecorder()

	done := make(chan struct{})
	go func() { defer close(done); server.handleChat(rr, req) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("handler hung past upstream timeout")
	}
	if rr.Code < 500 {
		t.Errorf("status = %d, want 5xx on upstream timeout (body=%q)", rr.Code, rr.Body.String())
	}
}
