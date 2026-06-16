package translate

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"

	"github.com/k0in/openai-ollama-proxy/internal/types"
)

func TestConvertMessagesToOpenAI_MultipleImages(t *testing.T) {
	pngB64 := base64.StdEncoding.EncodeToString([]byte{
		0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A,
		0x00, 0x00, 0x00, 0x0D, 'I', 'H', 'D', 'R',
	})
	jpegB64 := base64.StdEncoding.EncodeToString([]byte{
		0xFF, 0xD8, 0xFF, 0xE0,
	})

	msgs := []types.OllamaMessage{
		{
			Role:    "user",
			Content: "Two images",
			Images:  []string{pngB64, jpegB64},
		},
	}

	got, err := ConvertMessagesToOpenAI(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("len = %d, want 1", len(got))
	}

	var parts []types.OpenAIContentPart
	if err := json.Unmarshal(got[0].Content, &parts); err != nil {
		t.Fatal(err)
	}
	if len(parts) != 3 {
		t.Fatalf("len(parts) = %d, want 3 (text + 2 images)", len(parts))
	}
	if parts[0].Type != "text" || parts[0].Text != "Two images" {
		t.Errorf("parts[0] = %+v", parts[0])
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil {
		t.Errorf("parts[1] = %+v", parts[1])
	}
	if parts[2].Type != "image_url" || parts[2].ImageURL == nil {
		t.Errorf("parts[2] = %+v", parts[2])
	}
	// PNG MIME
	if !strings.HasPrefix(parts[1].ImageURL.URL, "data:image/png;base64,") {
		t.Errorf("parts[1] URL = %q, want data:image/png", parts[1].ImageURL.URL[:30])
	}
	// JPEG MIME
	if !strings.HasPrefix(parts[2].ImageURL.URL, "data:image/jpeg;base64,") {
		t.Errorf("parts[2] URL = %q, want data:image/jpeg", parts[2].ImageURL.URL[:30])
	}
}

func TestConvertMessagesToOpenAI_ImageOnlyNoText(t *testing.T) {
	pngB64 := base64.StdEncoding.EncodeToString([]byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A})

	msgs := []types.OllamaMessage{
		{
			Role:   "user",
			Images: []string{pngB64},
		},
	}

	got, err := ConvertMessagesToOpenAI(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("len = %d, want 1", len(got))
	}

	var parts []types.OpenAIContentPart
	if err := json.Unmarshal(got[0].Content, &parts); err != nil {
		t.Fatal(err)
	}
	if len(parts) != 1 {
		t.Fatalf("len(parts) = %d, want 1 (just the image)", len(parts))
	}
	if parts[0].Type != "image_url" || parts[0].ImageURL == nil {
		t.Errorf("parts[0] = %+v, want image_url", parts[0])
	}
	if !strings.HasPrefix(parts[0].ImageURL.URL, "data:image/png;base64,") {
		t.Errorf("URL = %q, want data:image/png prefix", parts[0].ImageURL.URL[:30])
	}
}

func TestConvertMessagesToOpenAI_ImageAlreadyDataURL(t *testing.T) {
	dataURL := "data:image/webp;base64,UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoBAAEAAwA0JaQAA3AA/vuUAAA="
	msgs := []types.OllamaMessage{
		{
			Role:    "user",
			Content: "What is this?",
			Images:  []string{dataURL},
		},
	}

	got, err := ConvertMessagesToOpenAI(msgs)
	if err != nil {
		t.Fatal(err)
	}

	var parts []types.OpenAIContentPart
	if err := json.Unmarshal(got[0].Content, &parts); err != nil {
		t.Fatal(err)
	}
	if len(parts) != 2 {
		t.Fatalf("len(parts) = %d, want 2", len(parts))
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil {
		t.Errorf("parts[1] = %+v", parts[1])
	}
	// Already a data URL — must be passed through without double-wrapping
	if parts[1].ImageURL.URL != dataURL {
		t.Errorf("image URL = %q, want %q (pass-through)", parts[1].ImageURL.URL, dataURL)
	}
}

func TestDetectImageMIME_WebP_RIFF(t *testing.T) {
	// RIFF<4-byte-size>WEBP — full 12-byte header
	got := DetectImageMIME(b64('R', 'I', 'F', 'F', 0x36, 0x00, 0x00, 0x00, 'W', 'E', 'B', 'P'))
	if got != "image/webp" {
		t.Errorf("DetectImageMIME = %q, want %q", got, "image/webp")
	}
}

func TestDetectImageMIME_AVIF_CompatibleBrand(t *testing.T) {
	// ftyp box with major brand "mif1" but compatible brands include "avif"
	data := make([]byte, 32)
	copy(data[4:8], "ftyp")
	copy(data[8:12], "mif1")
	copy(data[16:20], "avif")
	got := DetectImageMIME(b64(data...))
	// Major brand is mif1 => heif (mif1 takes precedence over compatible brand)
	if got != "image/heif" {
		t.Errorf("DetectImageMIME = %q, want %q", got, "image/heif")
	}
}

func TestDetectImageMIME_HEIC_CompatibleBrand(t *testing.T) {
	// ftyp box with major brand "mif1" and compatible brand "heic"
	data := make([]byte, 28)
	copy(data[4:8], "ftyp")
	copy(data[8:12], "mif1")
	copy(data[16:20], "mif1")
	copy(data[20:24], "heic")
	got := DetectImageMIME(b64(data...))
	if got != "image/heif" {
		t.Errorf("DetectImageMIME = %q, want %q", got, "image/heif")
	}
}

func TestDetectImageMIME_ShortB64(t *testing.T) {
	// Exactly 4 bytes decoded (3 base64 chars padded)
	got := DetectImageMIME("AAA=")
	if got != "image/jpeg" {
		t.Errorf("DetectImageMIME = %q, want %q (default)", got, "image/jpeg")
	}
}

func TestDetectImageMIME_LongPadding(t *testing.T) {
	// Long input but truncated to 700 bytes
	long := strings.Repeat("AAAA", 200) // 800 chars
	got := DetectImageMIME(long)
	// Should not panic, should not error
	if got != "image/jpeg" {
		t.Errorf("DetectImageMIME = %q, want %q (default)", got, "image/jpeg")
	}
}

func TestOllamaGenerateToOpenAI_WithJPEGImage(t *testing.T) {
	stream := false
	// JPEG SOI marker
	jpegB64 := base64.StdEncoding.EncodeToString([]byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 'J', 'F', 'I', 'F'})
	req := types.OllamaGenerateRequest{
		Model:  "qwen3:latest",
		Prompt: "What is this?",
		Images: []string{jpegB64},
		Stream: &stream,
	}

	got, err := OllamaGenerateToOpenAI(req)
	if err != nil {
		t.Fatal(err)
	}

	var parts []types.OpenAIContentPart
	if err := json.Unmarshal(got.Messages[0].Content, &parts); err != nil {
		t.Fatal(err)
	}
	if len(parts) != 2 {
		t.Fatalf("len(parts) = %d, want 2", len(parts))
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil {
		t.Fatalf("parts[1] = %+v", parts[1])
	}
	// Should detect JPEG MIME
	if !strings.HasPrefix(parts[1].ImageURL.URL, "data:image/jpeg;base64,") {
		t.Errorf("URL = %q, want data:image/jpeg prefix", parts[1].ImageURL.URL[:30])
	}
}

func TestOllamaGenerateToOpenAI_WithWebPImage(t *testing.T) {
	stream := false
	webpB64 := base64.StdEncoding.EncodeToString([]byte{'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'E', 'B', 'P'})
	req := types.OllamaGenerateRequest{
		Model:  "qwen3:latest",
		Prompt: "Describe",
		Images: []string{webpB64},
		Stream: &stream,
	}

	got, err := OllamaGenerateToOpenAI(req)
	if err != nil {
		t.Fatal(err)
	}

	var parts []types.OpenAIContentPart
	if err := json.Unmarshal(got.Messages[0].Content, &parts); err != nil {
		t.Fatal(err)
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil {
		t.Fatalf("parts[1] = %+v", parts[1])
	}
	if !strings.HasPrefix(parts[1].ImageURL.URL, "data:image/webp;base64,") {
		t.Errorf("URL = %q, want data:image/webp prefix", parts[1].ImageURL.URL[:30])
	}
}

func TestOllamaChatToOpenAI_WithImages(t *testing.T) {
	pngB64 := base64.StdEncoding.EncodeToString([]byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A})
	req := types.OllamaChatRequest{
		Model: "qwen3:latest",
		Messages: []types.OllamaMessage{
			{
				Role:    "user",
				Content: "Explain",
				Images:  []string{pngB64},
			},
		},
		Stream: boolPtr(false),
	}

	got, err := OllamaChatToOpenAI(req)
	if err != nil {
		t.Fatal(err)
	}

	var parts []types.OpenAIContentPart
	if err := json.Unmarshal(got.Messages[0].Content, &parts); err != nil {
		t.Fatal(err)
	}
	if len(parts) != 2 {
		t.Fatalf("len(parts) = %d, want 2 (text + image)", len(parts))
	}
	if parts[0].Type != "text" || parts[0].Text != "Explain" {
		t.Errorf("parts[0] = %+v", parts[0])
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil {
		t.Errorf("parts[1] = %+v", parts[1])
	}
	if !strings.HasPrefix(parts[1].ImageURL.URL, "data:image/png;base64,") {
		t.Errorf("URL = %q, want data:image/png", parts[1].ImageURL.URL[:30])
	}
}

func TestOllamaChatToOpenAI_ImagesOnAssistantMessage(t *testing.T) {
	pngB64 := base64.StdEncoding.EncodeToString([]byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A})
	req := types.OllamaChatRequest{
		Model: "qwen3:latest",
		Messages: []types.OllamaMessage{
			{Role: "system", Content: "You are a vision model."},
			{Role: "user", Content: "What is this?", Images: []string{pngB64}},
		},
		Stream: boolPtr(false),
	}

	got, err := OllamaChatToOpenAI(req)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Messages) != 2 {
		t.Fatalf("len(messages) = %d, want 2", len(got.Messages))
	}
	// First message (system) should be plain string content
	var sysContent string
	if err := json.Unmarshal(got.Messages[0].Content, &sysContent); err != nil {
		t.Fatal(err)
	}
	if sysContent != "You are a vision model." {
		t.Errorf("system = %q, want %q", sysContent, "You are a vision model.")
	}
	// Second message (user) should be array with image
	var parts []types.OpenAIContentPart
	if err := json.Unmarshal(got.Messages[1].Content, &parts); err != nil {
		t.Fatal(err)
	}
	if len(parts) != 2 {
		t.Fatalf("len(parts) = %d, want 2", len(parts))
	}
	if parts[1].Type != "image_url" {
		t.Errorf("parts[1].type = %q, want %q", parts[1].Type, "image_url")
	}
}

func TestOllamaMessages_ImageStream(t *testing.T) {
	pngB64 := base64.StdEncoding.EncodeToString([]byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A})
	stream := true
	req := types.OllamaChatRequest{
		Model: "qwen3:latest",
		Messages: []types.OllamaMessage{
			{Role: "user", Content: "What is this?", Images: []string{pngB64}},
		},
		Stream: &stream,
	}

	got, err := OllamaChatToOpenAI(req)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Stream {
		t.Error("Stream should be true")
	}
	if got.StreamOptions == nil || !got.StreamOptions.IncludeUsage {
		t.Error("StreamOptions.IncludeUsage should be enabled")
	}
	var parts []types.OpenAIContentPart
	if err := json.Unmarshal(got.Messages[0].Content, &parts); err != nil {
		t.Fatal(err)
	}
	if len(parts) != 2 {
		t.Fatalf("len(parts) = %d, want 2", len(parts))
	}
}
