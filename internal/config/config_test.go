package config

import (
	"testing"
	"time"
)

func TestEnvOr_WithEnvironmentSet(t *testing.T) {
	t.Setenv("TEST_KEY", "test-value")
	got := envOr("TEST_KEY", "fallback")
	if got != "test-value" {
		t.Fatalf("envOr = %q, want %q", got, "test-value")
	}
}

func TestEnvOr_WithEnvironmentUnset(t *testing.T) {
	t.Setenv("TEST_KEY_UNSET", "")
	got := envOr("TEST_KEY_UNSET", "fallback")
	if got != "fallback" {
		t.Fatalf("envOr = %q, want %q", got, "fallback")
	}
}

func TestEnvOrDuration_ValidDuration(t *testing.T) {
	t.Setenv("TEST_DURATION", "5m")
	got := envOrDuration("TEST_DURATION", 1*time.Hour)
	if got != 5*time.Minute {
		t.Fatalf("envOrDuration = %v, want %v", got, 5*time.Minute)
	}
}

func TestEnvOrDuration_InvalidDuration(t *testing.T) {
	t.Setenv("TEST_DURATION_INVALID", "not-a-duration")
	got := envOrDuration("TEST_DURATION_INVALID", 10*time.Second)
	if got != 10*time.Second {
		t.Fatalf("envOrDuration = %v, want fallback %v", got, 10*time.Second)
	}
}

func TestEnvOrDuration_Unset(t *testing.T) {
	got := envOrDuration("TEST_DURATION_NONEXISTENT", 3*time.Second)
	if got != 3*time.Second {
		t.Fatalf("envOrDuration = %v, want %v", got, 3*time.Second)
	}
}

func TestEnvOrInt_ValidInt(t *testing.T) {
	t.Setenv("TEST_INT", "42")
	got := envOrInt("TEST_INT", 0)
	if got != 42 {
		t.Fatalf("envOrInt = %d, want %d", got, 42)
	}
}

func TestEnvOrInt_InvalidInt(t *testing.T) {
	t.Setenv("TEST_INT_INVALID", "not-a-number")
	got := envOrInt("TEST_INT_INVALID", 99)
	if got != 99 {
		t.Fatalf("envOrInt = %d, want fallback %d", got, 99)
	}
}

func TestEnvOrInt_Unset(t *testing.T) {
	got := envOrInt("TEST_INT_NONEXISTENT", 7)
	if got != 7 {
		t.Fatalf("envOrInt = %d, want %d", got, 7)
	}
}

func TestLoad_DefaultValues(t *testing.T) {
	// Clear all relevant env vars
	t.Setenv("LISTEN_ADDR", "")
	t.Setenv("VLLM_BASE_URL", "")
	t.Setenv("VLLM_API_KEY", "")
	t.Setenv("VLLM_MODEL", "")
	t.Setenv("MODEL_NAME", "")
	t.Setenv("MODEL_CONTEXT_LENGTH", "")
	t.Setenv("OLLAMA_VERSION", "")
	t.Setenv("VLLM_STARTUP_WAIT", "")
	t.Setenv("VLLM_RETRY_INTERVAL", "")
	t.Setenv("DEBUG", "")

	cfg := Load()
	if cfg.ListenAddr != ":11434" {
		t.Errorf("ListenAddr = %q, want %q", cfg.ListenAddr, ":11434")
	}
	if cfg.VLLMBaseURL != "http://localhost:8000" {
		t.Errorf("VLLMBaseURL = %q, want %q", cfg.VLLMBaseURL, "http://localhost:8000")
	}
	if cfg.VLLMModel != "default" {
		t.Errorf("VLLMModel = %q, want %q", cfg.VLLMModel, "default")
	}
	if cfg.ModelName != "qwen3:latest" {
		t.Errorf("ModelName = %q, want %q", cfg.ModelName, "qwen3:latest")
	}
	if cfg.ModelContextLength != 65536 {
		t.Errorf("ModelContextLength = %d, want %d", cfg.ModelContextLength, 65536)
	}
	if cfg.OllamaVersion != "0.6.4" {
		t.Errorf("OllamaVersion = %q, want %q", cfg.OllamaVersion, "0.6.4")
	}
	if cfg.Debug {
		t.Error("Debug should be false when env is unset")
	}
}

func TestEnvOr_NonExistentKey(t *testing.T) {
	got := envOr("NONEXISTENT_KEY_12345", "default")
	if got != "default" {
		t.Fatalf("envOr = %q, want %q", got, "default")
	}
}

func TestEnvOrDuration_ZeroFallback(t *testing.T) {
	got := envOrDuration("NONEXISTENT_DURATION", 0)
	if got != 0 {
		t.Fatalf("envOrDuration = %v, want 0", got)
	}
}

func TestEnvOrInt_NegativeFallback(t *testing.T) {
	got := envOrInt("NONEXISTENT_INT", -1)
	if got != -1 {
		t.Fatalf("envOrInt = %d, want %d", got, -1)
	}
}
