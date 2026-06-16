package handlers

import (
	"encoding/base64"
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

// =============================================================================
// Binary artifact tests: images, audio, video
//
// These tests verify that base64-encoded binary data embedded in Ollama,
// OpenAI, and Anthropic requests is correctly forwarded to the upstream.
// Each test captures the upstream request payload and asserts that the
// content parts reach the upstream in the expected format.
// =============================================================================

// --- 1. Ollama /api/generate with multiple images (PNG + JPEG) ---------------

func TestOllamaGenerate_MultipleImages(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req types.OpenAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decoding upstream request: %v", err)
		}
		if len(req.Messages) != 1 {
			t.Fatalf("len(messages) = %d, want 1", len(req.Messages))
		}
		var parts []types.OpenAIContentPart
		if err := json.Unmarshal(req.Messages[0].Content, &parts); err != nil {
			t.Fatalf("content should be array: %v", err)
		}
		// Expect text + 2 images
		if len(parts) != 3 {
			t.Fatalf("len(contentParts) = %d, want 3 (text + 2 images)", len(parts))
		}
		if parts[0].Type != "text" || parts[0].Text != "Compare these images" {
			t.Errorf("parts[0] = %+v, want text part", parts[0])
		}
		if parts[1].Type != "image_url" || parts[1].ImageURL == nil {
			t.Errorf("parts[1] = %+v, want image_url", parts[1])
		}
		if parts[2].Type != "image_url" || parts[2].ImageURL == nil {
			t.Errorf("parts[2] = %+v, want image_url", parts[2])
		}
		// Verify data URL format
		if !strings.HasPrefix(parts[1].ImageURL.URL, "data:image/png;base64,") {
			t.Errorf("parts[1] URL prefix = %q, want data:image/png", parts[1].ImageURL.URL[:30])
		}
		if !strings.HasPrefix(parts[2].ImageURL.URL, "data:image/jpeg;base64,") {
			t.Errorf("parts[2] URL prefix = %q, want data:image/jpeg", parts[2].ImageURL.URL[:30])
		}

		content := "Two images compared"
		stop := "stop"
		json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			Choices: []types.OpenAIChoice{{
				Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
				FinishReason: &stop,
			}},
			Usage: &types.OpenAIUsage{PromptTokens: 30, CompletionTokens: 5, TotalTokens: 35},
		})
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "qwen3:latest"}}},
	}, 65536)
	server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	// Real PNG 1x1 pixel (barely) + JPEG placeholder (we encode a minimal JPEG SOI)
	pngB64 := base64.StdEncoding.EncodeToString([]byte{
		0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
		0x00, 0x00, 0x00, 0x0D, 'I', 'H', 'D', 'R', // IHDR chunk
	})
	jpegB64 := base64.StdEncoding.EncodeToString([]byte{
		0xFF, 0xD8, 0xFF, 0xE0, // JPEG SOI + APP0 marker
	})

	body := `{"model":"qwen3:latest","prompt":"Compare these images","images":["` + pngB64 + `","` + jpegB64 + `"],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleGenerate(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}

	var resp types.OllamaGenerateResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Response != "Two images compared" {
		t.Errorf("response = %q, want %q", resp.Response, "Two images compared")
	}
}

// --- 2. Ollama /api/chat with images (via OllamaMessage.Images) --------------

func TestOllamaChat_WithImages(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req types.OpenAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decoding upstream request: %v", err)
		}
		if len(req.Messages) != 1 {
			t.Fatalf("len(messages) = %d, want 1", len(req.Messages))
		}
		var parts []types.OpenAIContentPart
		if err := json.Unmarshal(req.Messages[0].Content, &parts); err != nil {
			t.Fatalf("content should be array: %v", err)
		}
		if len(parts) != 2 {
			t.Fatalf("len(parts) = %d, want 2 (text + image)", len(parts))
		}
		if parts[0].Type != "text" || parts[0].Text != "What is in this image?" {
			t.Errorf("parts[0] = %+v, want text", parts[0])
		}
		if parts[1].Type != "image_url" || parts[1].ImageURL == nil {
			t.Errorf("parts[1] = %+v, want image_url", parts[1])
		}
		if !strings.HasPrefix(parts[1].ImageURL.URL, "data:image/png;base64,") {
			t.Errorf("image URL prefix = %q, want data:image/png", parts[1].ImageURL.URL[:30])
		}

		content := "A scenic landscape"
		stop := "stop"
		json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			Choices: []types.OpenAIChoice{{
				Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
				FinishReason: &stop,
			}},
			Usage: &types.OpenAIUsage{PromptTokens: 20, CompletionTokens: 4, TotalTokens: 24},
		})
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "qwen3:latest"}}},
	}, 65536)
	server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	pngB64 := base64.StdEncoding.EncodeToString([]byte{
		0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A,
		0x00, 0x00, 0x00, 0x0D, 'I', 'H', 'D', 'R',
	})
	body := `{"model":"qwen3:latest","messages":[{"role":"user","content":"What is in this image?","images":["` + pngB64 + `"]}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}

	var resp types.OllamaChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Message.Content != "A scenic landscape" {
		t.Errorf("content = %q, want %q", resp.Message.Content, "A scenic landscape")
	}
	if resp.PromptEvalCount != 20 {
		t.Errorf("prompt_eval_count = %d, want 20", resp.PromptEvalCount)
	}
	if resp.EvalCount != 4 {
		t.Errorf("eval_count = %d, want 4", resp.EvalCount)
	}
}

// --- 3. OpenAI passthrough /v1/chat/completions with image_url ----------------

func TestOpenAIChat_WithImageURL(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req types.OpenAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decoding upstream request: %v", err)
		}
		if len(req.Messages) != 1 {
			t.Fatalf("len(messages) = %d, want 1", len(req.Messages))
		}
		// Verify the upstream receives the multi-modal content array intact
		var parts []types.OpenAIContentPart
		if err := json.Unmarshal(req.Messages[0].Content, &parts); err != nil {
			t.Fatalf("content should be array: %v", err)
		}
		if len(parts) != 2 {
			t.Fatalf("len(parts) = %d, want 2", len(parts))
		}
		if parts[0].Type != "text" || parts[0].Text != "Describe this image" {
			t.Errorf("parts[0] = %+v", parts[0])
		}
		if parts[1].Type != "image_url" || parts[1].ImageURL == nil {
			t.Errorf("parts[1] = %+v", parts[1])
		}

		content := "An image description"
		stop := "stop"
		json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			ID:      "chatcmpl-img",
			Object:  "chat.completion",
			Model:   req.Model,
			Created: 1700000001,
			Choices: []types.OpenAIChoice{{
				Index:        0,
				Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
				FinishReason: &stop,
			}},
			Usage: &types.OpenAIUsage{PromptTokens: 15, CompletionTokens: 3, TotalTokens: 18},
		})
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "gpt-upstream", Local: "gpt-4o"}}},
	}, 65536)

	cfg := config.Config{
		ListenAddr:            ":11434",
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	server := NewWithClients(cfg, router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	pngB64 := base64.StdEncoding.EncodeToString([]byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A})
	body := `{"model":"gpt-4o","messages":[{"role":"user","content":[{"type":"text","text":"Describe this image"},{"type":"image_url","image_url":{"url":"data:image/png;base64,` + pngB64 + `"}}]}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleOpenAIChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}

	var resp types.OpenAIChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if *resp.Choices[0].Message.Content != "An image description" {
		t.Errorf("content = %q, want %q", *resp.Choices[0].Message.Content, "An image description")
	}
}

// --- 4. Anthropic /v1/messages with image content block ----------------------

func TestAnthropicMessages_WithImage(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req types.OpenAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decoding upstream request: %v", err)
		}
		if len(req.Messages) != 1 {
			t.Fatalf("len(messages) = %d, want 1", len(req.Messages))
		}
		// Anthropic images get translated to OpenAI content parts
		var parts []types.OpenAIContentPart
		if err := json.Unmarshal(req.Messages[0].Content, &parts); err != nil {
			t.Fatalf("content should be array: %v", err)
		}
		if len(parts) != 2 {
			t.Fatalf("len(parts) = %d, want 2 (text + image)", len(parts))
		}
		if parts[0].Type != "text" || parts[0].Text != "What is in this image?" {
			t.Errorf("parts[0] = %+v", parts[0])
		}
		if parts[1].Type != "image_url" || parts[1].ImageURL == nil {
			t.Errorf("parts[1] = %+v", parts[1])
		}
		// Anthropic already prefixes data:image/...;base64, so it should pass through
		if !strings.HasPrefix(parts[1].ImageURL.URL, "data:image/png;base64,") {
			t.Errorf("image URL = %q, want data:image/png prefix", parts[1].ImageURL.URL[:40])
		}

		content := "I see a mountain range"
		stop := "stop"
		json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			Choices: []types.OpenAIChoice{{
				Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
				FinishReason: &stop,
			}},
			Usage: &types.OpenAIUsage{PromptTokens: 25, CompletionTokens: 6, TotalTokens: 31},
		})
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: upstream.URL, APIKey: "ant-key", Models: []config.ModelMapping{{Upstream: "claude-upstream", Local: "claude-sonnet-4-20250514"}}},
	}, 200000)

	cfg := config.Config{
		ListenAddr:            ":11434",
		ModelContextLength:    200000,
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	server := NewWithClients(cfg, router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	pngB64 := base64.StdEncoding.EncodeToString([]byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A})
	body := `{
		"model": "claude-sonnet-4-20250514",
		"max_tokens": 1024,
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "What is in this image?"},
				{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "` + pngB64 + `"}}
			]
		}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleAnthropicMessages(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}

	var resp types.AnthropicMessageResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Content) == 0 || resp.Content[0].Text != "I see a mountain range" {
		t.Errorf("content = %+v, want 'I see a mountain range'", resp.Content)
	}
	if resp.Usage == nil || resp.Usage.InputTokens != 25 || resp.Usage.OutputTokens != 6 {
		t.Errorf("usage = %+v, want InputTokens=25, OutputTokens=6", resp.Usage)
	}
}

// --- 5. OpenAI passthrough with input_audio content part ---------------------

func TestOpenAIChat_WithInputAudio(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req types.OpenAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decoding upstream request: %v", err)
		}
		if len(req.Messages) != 1 {
			t.Fatalf("len(messages) = %d, want 1", len(req.Messages))
		}
		var parts []types.OpenAIContentPart
		if err := json.Unmarshal(req.Messages[0].Content, &parts); err != nil {
			t.Fatalf("content should be array: %v", err)
		}
		if len(parts) != 2 {
			t.Fatalf("len(parts) = %d, want 2 (text + audio)", len(parts))
		}
		if parts[0].Type != "text" || parts[0].Text != "Transcribe this" {
			t.Errorf("parts[0] = %+v", parts[0])
		}
		if parts[1].Type != "input_audio" || parts[1].InputAudio == nil {
			t.Errorf("parts[1] = %+v, want input_audio", parts[1])
		}
		if parts[1].InputAudio.Format != "wav" {
			t.Errorf("audio format = %q, want %q", parts[1].InputAudio.Format, "wav")
		}
		if parts[1].InputAudio.Data == "" {
			t.Error("audio data should not be empty")
		}

		content := "Audio transcription result"
		stop := "stop"
		json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			ID:      "chatcmpl-audio",
			Object:  "chat.completion",
			Model:   req.Model,
			Created: 1700000002,
			Choices: []types.OpenAIChoice{{
				Index:        0,
				Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
				FinishReason: &stop,
			}},
			Usage: &types.OpenAIUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		})
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "gpt-upstream", Local: "gpt-4o-audio"}}},
	}, 65536)

	cfg := config.Config{
		ListenAddr:            ":11434",
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	server := NewWithClients(cfg, router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	audioB64 := base64.StdEncoding.EncodeToString([]byte("fake-wav-data"))
	body := `{"model":"gpt-4o-audio","messages":[{"role":"user","content":[{"type":"text","text":"Transcribe this"},{"type":"input_audio","input_audio":{"data":"` + audioB64 + `","format":"wav"}}]}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleOpenAIChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}

	var resp types.OpenAIChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if *resp.Choices[0].Message.Content != "Audio transcription result" {
		t.Errorf("content = %q, want %q", *resp.Choices[0].Message.Content, "Audio transcription result")
	}
}

// --- 6. OpenAI passthrough with reasoning + array content (regression) -------
//
// This tests the fix for normalizeOpenAIMessageMap where reasoning_content
// must NOT overwrite array-typed content (image_url, input_audio).

func TestOpenAIChat_ImageContentNotOverwrittenByReasoning(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req types.OpenAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decoding upstream request: %v", err)
		}
		if len(req.Messages) != 1 {
			t.Fatalf("len(messages) = %d, want 1", len(req.Messages))
		}
		var parts []types.OpenAIContentPart
		if err := json.Unmarshal(req.Messages[0].Content, &parts); err != nil {
			t.Fatalf("content should be array: %v", err)
		}
		if len(parts) != 2 {
			t.Fatalf("len(parts) = %d, want 2", len(parts))
		}
		if parts[1].Type != "image_url" {
			t.Errorf("parts[1].type = %q, want %q (image_url not overwritten)", parts[1].Type, "image_url")
		}

		// Upstream returns reasoning in response. The proxy must handle this.
		reasoning := "Let me analyze this image..."
		content := "This is a picture of a cat"
		stop := "stop"
		json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			ID:      "chatcmpl-reasoning",
			Object:  "chat.completion",
			Model:   req.Model,
			Created: 1700000003,
			Choices: []types.OpenAIChoice{{
				Index: 0,
				Message: &types.OpenAIRespMsg{
					Role:             "assistant",
					Content:          &content,
					ReasoningContent: &reasoning,
				},
				FinishReason: &stop,
			}},
			Usage: &types.OpenAIUsage{PromptTokens: 20, CompletionTokens: 10, TotalTokens: 30},
		})
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "gpt-upstream", Local: "gpt-4o"}}},
	}, 65536)

	cfg := config.Config{
		ListenAddr:            ":11434",
		OllamaVersion:         "0.6.4",
		UpstreamStartupWait:   0,
		UpstreamRetryInterval: 10 * time.Millisecond,
	}
	server := NewWithClients(cfg, router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	pngB64 := base64.StdEncoding.EncodeToString([]byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A})
	body := `{"model":"gpt-4o","messages":[{"role":"user","content":[{"type":"text","text":"What is this?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,` + pngB64 + `"}}]}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleOpenAIChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}

	var resp types.OpenAIChatResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Choices) == 0 || resp.Choices[0].Message == nil {
		t.Fatal("no choices in response")
	}
	if resp.Choices[0].Message.Content == nil {
		t.Fatal("content is nil")
	}
	if *resp.Choices[0].Message.Content != "This is a picture of a cat" {
		t.Errorf("content = %q, want %q", *resp.Choices[0].Message.Content, "This is a picture of a cat")
	}
}

// --- 7. OllamaGenerate_WithImages: verify exact base64 data arrives upstream --

func TestOllamaGenerate_WithImages_ExactData(t *testing.T) {
	// Known base64 string for a real 1x1 red pixel PNG
	const b64Image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req types.OpenAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decoding upstream request: %v", err)
		}
		if len(req.Messages) != 1 {
			t.Fatalf("len(messages) = %d, want 1", len(req.Messages))
		}
		var parts []types.OpenAIContentPart
		if err := json.Unmarshal(req.Messages[0].Content, &parts); err != nil {
			t.Fatalf("content should be array: %v", err)
		}
		if len(parts) != 2 {
			t.Fatalf("len(parts) = %d, want 2", len(parts))
		}

		// The image data must be wrapped in a data URL with correct MIME
		dataURL := parts[1].ImageURL.URL
		expectedPrefix := "data:image/png;base64," + b64Image
		if dataURL != expectedPrefix {
			t.Errorf("image data URL mismatch\n  got:  %q\n  want: %q", dataURL, expectedPrefix)
		}

		content := "Red pixel detected"
		stop := "stop"
		json.NewEncoder(w).Encode(types.OpenAIChatResponse{
			Choices: []types.OpenAIChoice{{
				Message:      &types.OpenAIRespMsg{Role: "assistant", Content: &content},
				FinishReason: &stop,
			}},
			Usage: &types.OpenAIUsage{PromptTokens: 10, CompletionTokens: 3, TotalTokens: 13},
		})
	}))
	defer upstream.Close()

	router, _ := config.BuildRoutingTable([]config.UpstreamConfig{
		{URL: upstream.URL, Models: []config.ModelMapping{{Upstream: "m", Local: "qwen3:latest"}}},
	}, 65536)
	server := NewWithClients(config.Config{ListenAddr: ":11434", UpstreamStartupWait: 0, UpstreamRetryInterval: 10 * time.Millisecond},
		router, &http.Client{Timeout: 5 * time.Second}, &http.Client{Timeout: 5 * time.Second}, stats.New())

	body := `{"model":"qwen3:latest","prompt":"Describe this image","images":["` + b64Image + `"],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(body))
	w := httptest.NewRecorder()
	server.handleGenerate(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", w.Code, w.Body.String())
	}

	var resp types.OllamaGenerateResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Response != "Red pixel detected" {
		t.Errorf("response = %q, want %q", resp.Response, "Red pixel detected")
	}
}
