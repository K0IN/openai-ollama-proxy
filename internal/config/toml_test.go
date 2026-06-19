package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadFile(t *testing.T) {
	t.Run("valid multi-upstream TOML", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "config.toml")

		content := `
listen_addr = ":9999"
model_context_length = 8192
debug = true

[[upstream]]
url = "http://localhost:8000"
api_key = "sk-local"

[[upstream.models]]
upstream = "qwen2.5-coder-14b"
local = "qwen-coder"
context_length = 32768

[[upstream.models]]
upstream = "qwen3-27b-fp8"
local = "qwen3-large"

[[upstream]]
url = "https://api.openai.com"
api_key = "sk-openai"

[[upstream.models]]
upstream = "gpt-4o"
local = "gpt-4o"
context_length = 128000

[[upstream.models]]
upstream = "gpt-5.4"
local = "gpt-5.4"
supports_thinking = ["low", "medium"]
`
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatalf("write temp config: %v", err)
		}

		cfg, router, err := LoadFile(path)
		if err != nil {
			t.Fatalf("LoadFile: %v", err)
		}

		if cfg.ListenAddr != ":9999" {
			t.Fatalf("ListenAddr = %q, want :9999", cfg.ListenAddr)
		}
		if cfg.ModelContextLength != 8192 {
			t.Fatalf("ModelContextLength = %d, want 8192", cfg.ModelContextLength)
		}
		if !cfg.Debug {
			t.Fatal("Debug should be true")
		}
		if len(cfg.Upstreams) != 2 {
			t.Fatalf("Upstreams length = %d, want 2", len(cfg.Upstreams))
		}

		// Check router
		models := router.AllModels()
		if len(models) != 5 {
			t.Fatalf("models = %v, want 5", models)
		}

		// qwen-coder with explicit context_length
		entry, _ := router.Lookup("qwen-coder")
		if entry.ContextLength != 32768 {
			t.Fatalf("qwen-coder context_length = %d, want 32768", entry.ContextLength)
		}

		// qwen3-large with global default
		entry, _ = router.Lookup("qwen3-large")
		if entry.ContextLength != 8192 {
			t.Fatalf("qwen3-large context_length = %d, want 8192", entry.ContextLength)
		}

		// gpt-4o with explicit context_length and API key
		entry, _ = router.Lookup("gpt-4o")
		if entry.APIKey != "sk-openai" {
			t.Fatalf("gpt-4o APIKey = %q", entry.APIKey)
		}

		entry, _ = router.Lookup("gpt-5.4-low")
		if !entry.SupportsThinking || entry.ThinkingLevel != "low" {
			t.Fatalf("gpt-5.4-low entry = %#v", entry)
		}

		entry, _ = router.Lookup("gpt-5.4-medium")
		if !entry.SupportsThinking || entry.ThinkingLevel != "medium" {
			t.Fatalf("gpt-5.4-medium entry = %#v", entry)
		}
	})

	t.Run("missing file", func(t *testing.T) {
		_, _, err := LoadFile("/nonexistent/path/config.toml")
		if err == nil {
			t.Fatal("expected error for missing file")
		}
	})

	t.Run("invalid TOML syntax", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "invalid.toml")
		if err := os.WriteFile(path, []byte("this is not valid {{{ toml"), 0o644); err != nil {
			t.Fatalf("write temp config: %v", err)
		}

		_, _, err := LoadFile(path)
		if err == nil {
			t.Fatal("expected error for invalid TOML")
		}
	})

	t.Run("missing upstreams", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "no-upstreams.toml")
		content := "listen_addr = \":11434\"\n"
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatalf("write temp config: %v", err)
		}

		_, _, err := LoadFile(path)
		if err == nil {
			t.Fatal("expected error for config with no upstreams configured")
		}
	})
}

func TestLoadFile_DuplicateDetection(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "dup.toml")
	content := `
[[upstream]]
url = "http://a:8000"
[[upstream.models]]
upstream = "m"
local = "same-name"

[[upstream]]
url = "http://b:9000"
[[upstream.models]]
upstream = "m2"
local = "same-name"
`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write temp config: %v", err)
	}

	_, _, err := LoadFile(path)
	if err == nil {
		t.Fatal("expected error for duplicate local model name")
	}
	if !strings.Contains(err.Error(), "duplicate local model") {
		t.Fatalf("error %q does not contain 'duplicate local model'", err.Error())
	}
}
