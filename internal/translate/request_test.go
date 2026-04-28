package translate

import (
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/k0in/openai-ollama-proxy/internal/types"
)

func floatPtr(f float64) *float64 {
	return &f
}

func b64(data ...byte) string {
	return base64.StdEncoding.EncodeToString(data)
}

func TestDetectImageMIME_PNG(t *testing.T) {
	// PNG magic bytes: 0x89 0x50 0x4E 0x47 0x0D 0x0A 0x1A 0x0A
	got := DetectImageMIME(b64(0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A))
	if got != "image/png" {
		t.Errorf("DetectImageMIME = %q, want %q", got, "image/png")
	}
}

func TestDetectImageMIME_GIF(t *testing.T) {
	// GIF89a is the standard 6-byte header recognised by http.DetectContentType.
	got := DetectImageMIME(b64('G', 'I', 'F', '8', '9', 'a'))
	if got != "image/gif" {
		t.Errorf("DetectImageMIME = %q, want %q", got, "image/gif")
	}
}

func TestDetectImageMIME_WebP(t *testing.T) {
	// RIFF<size>WEBP — 12 bytes is the minimum for RIFF detection.
	got := DetectImageMIME(b64('R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'E', 'B', 'P'))
	if got != "image/webp" {
		t.Errorf("DetectImageMIME = %q, want %q", got, "image/webp")
	}
}

func TestDetectImageMIME_AVIF(t *testing.T) {
	// ftyp box with major brand "avif".
	got := DetectImageMIME(b64(0, 0, 0, 0x20, 'f', 't', 'y', 'p', 'a', 'v', 'i', 'f', 0, 0, 0, 0))
	if got != "image/avif" {
		t.Errorf("DetectImageMIME = %q, want %q", got, "image/avif")
	}
}

func TestDetectImageMIME_HEIC(t *testing.T) {
	// ftyp box with major brand "heic".
	got := DetectImageMIME(b64(0, 0, 0, 0x20, 'f', 't', 'y', 'p', 'h', 'e', 'i', 'c', 0, 0, 0, 0))
	if got != "image/heic" {
		t.Errorf("DetectImageMIME = %q, want %q", got, "image/heic")
	}
}

func TestDetectImageMIME_HEIF_CompatibleBrand(t *testing.T) {
	// ftyp box with major brand "mif1" + compatible brand listing "mif1" again.
	got := DetectImageMIME(b64(0, 0, 0, 0x18, 'f', 't', 'y', 'p', 'm', 'i', 'f', '1', 0, 0, 0, 0, 'm', 'i', 'f', '1', 'h', 'e', 'i', 'c'))
	// Major brand match takes precedence — mif1 -> image/heif.
	if got != "image/heif" {
		t.Errorf("DetectImageMIME = %q, want %q", got, "image/heif")
	}
}

func TestDetectImageMIME_JPEG_Default(t *testing.T) {
	// JPEG SOI + APP0 marker.
	got := DetectImageMIME(b64(0xFF, 0xD8, 0xFF, 0xE0, 0, 0x10, 'J', 'F', 'I', 'F', 0, 1, 1))
	if got != "image/jpeg" {
		t.Errorf("DetectImageMIME = %q, want %q", got, "image/jpeg")
	}
}

func TestDetectImageMIME_InvalidBase64(t *testing.T) {
	got := DetectImageMIME("!!!invalid!!!")
	if got != "image/jpeg" {
		t.Errorf("DetectImageMIME = %q, want %q (default)", got, "image/jpeg")
	}
}

func TestDetectImageMIME_ShortString(t *testing.T) {
	got := DetectImageMIME("YWJj") // "abc" - too short for magic bytes
	if got != "image/jpeg" {
		t.Errorf("DetectImageMIME = %q, want %q (default)", got, "image/jpeg")
	}
}

func TestOllamaGenerateToOpenAI_Basic(t *testing.T) {
	stream := false
	req := types.OllamaGenerateRequest{
		Model:  "qwen3:latest",
		Prompt: "Hello",
		Stream: &stream,
	}

	got, err := OllamaGenerateToOpenAI(req)
	if err != nil {
		t.Fatal(err)
	}
	if got.Stream {
		t.Fatal("Stream should be false")
	}
	if len(got.Messages) != 1 {
		t.Fatalf("len(Messages) = %d, want 1", len(got.Messages))
	}
	if got.Messages[0].Role != "user" {
		t.Errorf("Messages[0].Role = %q, want %q", got.Messages[0].Role, "user")
	}
}

func TestOllamaGenerateToOpenAI_WithSystem(t *testing.T) {
	stream := false
	req := types.OllamaGenerateRequest{
		Model:  "qwen3:latest",
		Prompt: "Hello",
		System: "You are helpful.",
		Stream: &stream,
	}

	got, err := OllamaGenerateToOpenAI(req)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Messages) != 2 {
		t.Fatalf("len(Messages) = %d, want 2", len(got.Messages))
	}
	if got.Messages[0].Role != "system" {
		t.Errorf("Messages[0].Role = %q, want %q", got.Messages[0].Role, "system")
	}
}

func TestOllamaGenerateToOpenAI_WithImages(t *testing.T) {
	stream := false
	req := types.OllamaGenerateRequest{
		Model:  "qwen3:latest",
		Prompt: "What is this?",
		Images: []string{"iVBORw0KGgo="},
		Stream: &stream,
	}

	got, err := OllamaGenerateToOpenAI(req)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Messages) != 1 {
		t.Fatalf("len(Messages) = %d, want 1", len(got.Messages))
	}

	var content json.RawMessage = got.Messages[0].Content
	var parts []types.OpenAIContentPart
	if err := json.Unmarshal(content, &parts); err != nil {
		t.Fatal(err)
	}
	if len(parts) != 2 {
		t.Fatalf("len(parts) = %d, want 2", len(parts))
	}
	if parts[0].Type != "text" {
		t.Errorf("parts[0].Type = %q, want %q", parts[0].Type, "text")
	}
	if parts[1].Type != "image_url" {
		t.Errorf("parts[1].Type = %q, want %q", parts[1].Type, "image_url")
	}
}

func TestOllamaGenerateToOpenAI_WithOptions(t *testing.T) {
	stream := false
	options := types.OllamaOptions{
		Temperature: floatPtr(0.7),
		TopP:        floatPtr(0.9),
	}
	req := types.OllamaGenerateRequest{
		Model:   "qwen3:latest",
		Prompt:  "Hello",
		Stream:  &stream,
		Options: options,
	}

	got, err := OllamaGenerateToOpenAI(req)
	if err != nil {
		t.Fatal(err)
	}
	if got.Temperature == nil || *got.Temperature != 0.7 {
		t.Errorf("Temperature = %v, want 0.7", got.Temperature)
	}
	if got.TopP == nil || *got.TopP != 0.9 {
		t.Errorf("TopP = %v, want 0.9", got.TopP)
	}
}
