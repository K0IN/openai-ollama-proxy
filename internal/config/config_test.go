package config

import (
	"testing"
	"time"
)

func TestEnvOr_WithEnvironmentSet(t *testing.T) {
	// envOr always returns fallback after legacy env var removal.
	got := envOr("TEST_KEY", "fallback")
	if got != "fallback" {
		t.Fatalf("envOr = %q, want %q", got, "fallback")
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
	// envOrDuration always returns fallback after legacy env var removal.
	got := envOrDuration("TEST_DURATION", 1*time.Hour)
	if got != 1*time.Hour {
		t.Fatalf("envOrDuration = %v, want %v", got, 1*time.Hour)
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
	// envOrInt always returns fallback after legacy env var removal.
	got := envOrInt("TEST_INT", 0)
	if got != 0 {
		t.Fatalf("envOrInt = %d, want %d", got, 0)
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

func TestLoadFromEnv_IsNoop(t *testing.T) {
	cfg := LoadFromEnv()
	if cfg.ListenAddr != "" {
		t.Errorf("ListenAddr should be empty, got %q", cfg.ListenAddr)
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
