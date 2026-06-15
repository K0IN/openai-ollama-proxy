package config

import (
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	ListenAddr            string           `toml:"listen_addr"`
	UpstreamBaseURL       string           `toml:"upstream_base_url"`
	UpstreamAPIKey        string           `toml:"upstream_api_key"`
	UpstreamModel         string           `toml:"upstream_model"`
	ModelName             string           `toml:"model_name"`
	ModelContextLength    int              `toml:"model_context_length"`
	OllamaVersion         string           `toml:"ollama_version"`
	UpstreamStartupWait   time.Duration    `toml:"upstream_startup_wait"`
	UpstreamRetryInterval time.Duration    `toml:"upstream_retry_interval"`
	MaxRequestBytes       int64            `toml:"max_request_bytes"`
	ShutdownTimeout       time.Duration    `toml:"shutdown_timeout"`
	HTTPRequestTimeout    time.Duration    `toml:"http_request_timeout"`
	HTTPStreamTimeout     time.Duration    `toml:"http_stream_timeout"`
	Debug                 bool             `toml:"debug"`
	Upstreams             []UpstreamConfig `toml:"upstream"`
}

func (c *Config) applyDefaults() {
	if c.ListenAddr == "" {
		c.ListenAddr = ":11434"
	}
	if c.ModelContextLength <= 0 {
		c.ModelContextLength = 65536
	}
	if c.OllamaVersion == "" {
		c.OllamaVersion = "0.6.4"
	}
	if c.UpstreamStartupWait <= 0 {
		c.UpstreamStartupWait = 30 * time.Minute
	}
	if c.UpstreamRetryInterval <= 0 {
		c.UpstreamRetryInterval = 2 * time.Second
	}
	if c.MaxRequestBytes <= 0 {
		c.MaxRequestBytes = 32 << 20 // 32 MiB
	}
	if c.ShutdownTimeout <= 0 {
		c.ShutdownTimeout = 30 * time.Second
	}
	if c.HTTPRequestTimeout <= 0 {
		c.HTTPRequestTimeout = 30 * time.Second
	}
	if c.HTTPStreamTimeout <= 0 {
		c.HTTPStreamTimeout = 5 * time.Minute
	}
}

func LoadFromEnv() Config {
	cfg := Config{
		ListenAddr:            envOr("LISTEN_ADDR", ":11434"),
		UpstreamBaseURL:       envOr("UPSTREAM_BASE_URL", "http://localhost:8000"),
		UpstreamAPIKey:        envOr("UPSTREAM_API_KEY", ""),
		UpstreamModel:         envOr("UPSTREAM_MODEL", "default"),
		ModelName:             envOr("MODEL_NAME", "generic:latest"),
		ModelContextLength:    envOrInt("MODEL_CONTEXT_LENGTH", 65536),
		OllamaVersion:         envOr("OLLAMA_VERSION", "0.6.4"),
		UpstreamStartupWait:   envOrDuration("UPSTREAM_STARTUP_WAIT", 30*time.Minute),
		UpstreamRetryInterval: envOrDuration("UPSTREAM_RETRY_INTERVAL", 2*time.Second),
		MaxRequestBytes:       envOrInt64("MAX_REQUEST_BYTES", 32<<20), // 32 MiB
		ShutdownTimeout:       envOrDuration("SHUTDOWN_TIMEOUT", 30*time.Second),
		HTTPRequestTimeout:    envOrDuration("HTTP_REQUEST_TIMEOUT", 30*time.Second),
		HTTPStreamTimeout:     envOrDuration("HTTP_STREAM_TIMEOUT", 5*time.Minute),
		Debug:                 os.Getenv("DEBUG") == "true" || os.Getenv("DEBUG") == "1",
	}

	if err := cfg.Validate(); err != nil {
		log.Fatalf("invalid configuration: %v", err)
	}

	return cfg
}

func (c Config) Validate() error {
	var errs []string

	if strings.TrimSpace(c.ListenAddr) == "" {
		errs = append(errs, "LISTEN_ADDR must not be empty")
	} else if _, err := net.ResolveTCPAddr("tcp", c.ListenAddr); err != nil {
		errs = append(errs, fmt.Sprintf("LISTEN_ADDR %q is not a valid host:port: %v", c.ListenAddr, err))
	}

	if strings.TrimSpace(c.UpstreamBaseURL) == "" {
		// Allow empty UPSTREAM_BASE_URL when using TOML upstream-based config
		if len(c.Upstreams) == 0 {
			errs = append(errs, "UPSTREAM_BASE_URL must not be empty")
		}
	} else {
		parsed, err := url.Parse(c.UpstreamBaseURL)
		switch {
		case err != nil:
			errs = append(errs, fmt.Sprintf("UPSTREAM_BASE_URL %q is not a valid URL: %v", c.UpstreamBaseURL, err))
		case parsed.Scheme != "http" && parsed.Scheme != "https":
			errs = append(errs, fmt.Sprintf("UPSTREAM_BASE_URL %q must use http or https scheme", c.UpstreamBaseURL))
		case parsed.Host == "":
			errs = append(errs, fmt.Sprintf("UPSTREAM_BASE_URL %q is missing a host", c.UpstreamBaseURL))
		}
	}

	if strings.TrimSpace(c.UpstreamModel) == "" {
		// Allow empty UPSTREAM_MODEL when using TOML upstream-based config
		if len(c.Upstreams) == 0 {
			errs = append(errs, "UPSTREAM_MODEL must not be empty")
		}
	}

	if strings.TrimSpace(c.ModelName) == "" {
		// Allow empty MODEL_NAME when using TOML upstream-based config
		if len(c.Upstreams) == 0 {
			errs = append(errs, "MODEL_NAME must not be empty")
		}
	}

	if c.ModelContextLength <= 0 {
		errs = append(errs, fmt.Sprintf("MODEL_CONTEXT_LENGTH must be > 0 (got %d)", c.ModelContextLength))
	}

	if c.UpstreamStartupWait < 0 {
		errs = append(errs, fmt.Sprintf("UPSTREAM_STARTUP_WAIT must be >= 0 (got %s)", c.UpstreamStartupWait))
	}
	if c.UpstreamRetryInterval <= 0 {
		errs = append(errs, fmt.Sprintf("UPSTREAM_RETRY_INTERVAL must be > 0 (got %s)", c.UpstreamRetryInterval))
	}

	if c.MaxRequestBytes < 0 {
		errs = append(errs, fmt.Sprintf("MAX_REQUEST_BYTES must be >= 0 (got %d)", c.MaxRequestBytes))
	}
	if c.ShutdownTimeout < 0 {
		errs = append(errs, fmt.Sprintf("SHUTDOWN_TIMEOUT must be >= 0 (got %s)", c.ShutdownTimeout))
	}
	if c.HTTPRequestTimeout < 0 {
		errs = append(errs, fmt.Sprintf("HTTP_REQUEST_TIMEOUT must be >= 0 (got %s)", c.HTTPRequestTimeout))
	}
	if c.HTTPStreamTimeout < 0 {
		errs = append(errs, fmt.Sprintf("HTTP_STREAM_TIMEOUT must be >= 0 (got %s)", c.HTTPStreamTimeout))
	}

	if len(errs) > 0 {
		return errors.New(strings.Join(errs, "; "))
	}
	return nil
}

func NewHTTPClient(cfg Config) *http.Client {
	timeout := cfg.HTTPStreamTimeout
	if timeout <= 0 {
		timeout = 5 * time.Minute
	}
	return &http.Client{Timeout: timeout}
}

func NewRequestHTTPClient(cfg Config) *http.Client {
	timeout := cfg.HTTPRequestTimeout
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	return &http.Client{Timeout: timeout}
}

func envOr(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func envOrDuration(key string, fallback time.Duration) time.Duration {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}

	duration, err := time.ParseDuration(value)
	if err != nil {
		log.Printf("invalid %s=%q, using default %s", key, sanitizeForLog(value), fallback) // #nosec G706 -- value sanitized via sanitizeForLog
		return fallback
	}

	return duration
}

func envOrInt(key string, fallback int) int {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}

	parsed, err := strconv.Atoi(value)
	if err != nil {
		log.Printf("invalid %s=%q, using default %d", key, sanitizeForLog(value), fallback) // #nosec G706 -- value sanitized via sanitizeForLog
		return fallback
	}

	return parsed
}

func envOrInt64(key string, fallback int64) int64 {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}

	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		log.Printf("invalid %s=%q, using default %d", key, sanitizeForLog(value), fallback) // #nosec G706 -- value sanitized via sanitizeForLog
		return fallback
	}

	return parsed
}

// sanitizeForLog replaces ASCII control characters in s with spaces so
// untrusted environment variable values cannot forge log lines (CWE-117).
func sanitizeForLog(s string) string {
	if s == "" {
		return s
	}
	b := []byte(s)
	for i, c := range b {
		if (c < 0x20 && c != '\t') || c == 0x7f {
			b[i] = ' '
		}
	}
	return string(b)
}
