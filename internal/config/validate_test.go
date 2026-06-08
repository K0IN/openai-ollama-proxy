package config

import (
	"strings"
	"testing"
	"time"
)

func validConfig() Config {
	return Config{
		ListenAddr:            ":11434",
		UpstreamBaseURL:       "http://upstream:8000",
		UpstreamModel:         "default",
		ModelName:             "generic:latest",
		ModelContextLength:    65536,
		UpstreamStartupWait:   30 * time.Minute,
		UpstreamRetryInterval: 2 * time.Second,
		MaxRequestBytes:       32 << 20,
		ShutdownTimeout:       30 * time.Second,
		HTTPRequestTimeout:    30 * time.Second,
		HTTPStreamTimeout:     5 * time.Minute,
	}
}

func TestValidate_OK(t *testing.T) {
	if err := validConfig().Validate(); err != nil {
		t.Fatalf("validConfig should pass: %v", err)
	}
}

func TestValidate_ListenAddr(t *testing.T) {
	cases := []struct{ addr, want string }{
		{"", "LISTEN_ADDR must not be empty"},
		{"not-a-host-port", "is not a valid host:port"},
		{":notaport", "is not a valid host:port"},
	}
	for _, c := range cases {
		cfg := validConfig()
		cfg.ListenAddr = c.addr
		err := cfg.Validate()
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("ListenAddr=%q: got %v, want substring %q", c.addr, err, c.want)
		}
	}
}

func TestValidate_UpstreamBaseURL(t *testing.T) {
	cases := []struct{ url, want string }{
		{"", "UPSTREAM_BASE_URL must not be empty"},
		{"://broken", "is not a valid URL"},
		{"ftp://host", "must use http or https scheme"},
		{"http://", "is missing a host"},
	}
	for _, c := range cases {
		cfg := validConfig()
		cfg.UpstreamBaseURL = c.url
		err := cfg.Validate()
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("UpstreamBaseURL=%q: got %v, want substring %q", c.url, err, c.want)
		}
	}
}

func TestValidate_HTTPSBaseURL_Allowed(t *testing.T) {
	cfg := validConfig()
	cfg.UpstreamBaseURL = "https://upstream.example.com"
	if err := cfg.Validate(); err != nil {
		t.Fatalf("https URL should be valid: %v", err)
	}
}

func TestValidate_PositiveInts(t *testing.T) {
	cfg := validConfig()
	cfg.ModelContextLength = 0
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "MODEL_CONTEXT_LENGTH") {
		t.Errorf("ModelContextLength=0: got %v, want MODEL_CONTEXT_LENGTH error", err)
	}

	cfg = validConfig()
	cfg.UpstreamRetryInterval = 0
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "UPSTREAM_RETRY_INTERVAL") {
		t.Errorf("UpstreamRetryInterval=0: got %v, want UPSTREAM_RETRY_INTERVAL error", err)
	}
}

func TestValidate_NonNegativeDurations(t *testing.T) {
	cases := []func(*Config){
		func(c *Config) { c.UpstreamStartupWait = -1 },
		func(c *Config) { c.MaxRequestBytes = -1 },
		func(c *Config) { c.ShutdownTimeout = -1 },
		func(c *Config) { c.HTTPRequestTimeout = -1 },
		func(c *Config) { c.HTTPStreamTimeout = -1 },
	}
	for i, mut := range cases {
		cfg := validConfig()
		mut(&cfg)
		if err := cfg.Validate(); err == nil {
			t.Errorf("case %d: negative value should fail validation", i)
		}
	}
}

func TestValidate_RequiredStrings(t *testing.T) {
	cfg := validConfig()
	cfg.UpstreamModel = "   "
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "UPSTREAM_MODEL") {
		t.Errorf("blank UPSTREAM_MODEL should fail, got %v", err)
	}

	cfg = validConfig()
	cfg.ModelName = ""
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "MODEL_NAME") {
		t.Errorf("empty MODEL_NAME should fail, got %v", err)
	}
}

func TestValidate_AggregatesErrors(t *testing.T) {
	cfg := Config{} // entirely zero — many things wrong at once
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected error for empty Config")
	}
	for _, want := range []string{"LISTEN_ADDR", "UPSTREAM_BASE_URL", "UPSTREAM_MODEL", "MODEL_NAME", "MODEL_CONTEXT_LENGTH"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("aggregated error missing %q in %q", want, err.Error())
		}
	}
}

func TestNewHTTPClient_AppliesStreamTimeout(t *testing.T) {
	cfg := validConfig()
	cfg.HTTPStreamTimeout = 7 * time.Minute
	c := NewHTTPClient(cfg)
	if c.Timeout != 7*time.Minute {
		t.Errorf("stream timeout = %v, want 7m", c.Timeout)
	}
}

func TestNewRequestHTTPClient_AppliesRequestTimeout(t *testing.T) {
	cfg := validConfig()
	cfg.HTTPRequestTimeout = 17 * time.Second
	c := NewRequestHTTPClient(cfg)
	if c.Timeout != 17*time.Second {
		t.Errorf("request timeout = %v, want 17s", c.Timeout)
	}
}

func TestNewClients_FallbackOnZero(t *testing.T) {
	cfg := validConfig()
	cfg.HTTPStreamTimeout = 0
	cfg.HTTPRequestTimeout = 0
	if NewHTTPClient(cfg).Timeout == 0 {
		t.Errorf("zero stream timeout should fall back to non-zero default")
	}
	if NewRequestHTTPClient(cfg).Timeout == 0 {
		t.Errorf("zero request timeout should fall back to non-zero default")
	}
}
