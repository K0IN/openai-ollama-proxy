package config

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"strings"
	"time"
)

type Config struct {
	ListenAddr            string           `toml:"listen_addr"`
	ModelContextLength    int              `toml:"model_context_length"`
	ProxyAPIKey           string           `toml:"proxy_api_key"`
	OllamaVersion         string           `toml:"ollama_version"`
	UpstreamStartupWait   time.Duration    `toml:"upstream_startup_wait"`
	UpstreamRetryInterval time.Duration    `toml:"upstream_retry_interval"`
	LogMaxBodyBytes       int              `toml:"log_max_body_bytes"`
	MaxRequestBytes       int64            `toml:"max_request_bytes"`
	ShutdownTimeout       time.Duration    `toml:"shutdown_timeout"`
	HTTPRequestTimeout    time.Duration    `toml:"http_request_timeout"`
	HTTPStreamTimeout     time.Duration    `toml:"http_stream_timeout"`
	Debug                 bool             `toml:"debug"`
	StatsStorePath        string           `toml:"stats_store_path"`
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
	if c.LogMaxBodyBytes <= 0 {
		c.LogMaxBodyBytes = 4096
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

// LoadFromEnv is a no-op stub kept for backward compatibility.
// All configuration should be done via TOML files (see config.toml).
//
// Deprecated: will be removed in a future release.
func LoadFromEnv() Config {
	return Config{}
}

func (c Config) Validate() error {
	var errs []string

	if strings.TrimSpace(c.ListenAddr) == "" {
		errs = append(errs, "listen_addr must not be empty")
	} else if _, err := net.ResolveTCPAddr("tcp", c.ListenAddr); err != nil {
		errs = append(errs, fmt.Sprintf("listen_addr %q is not a valid host:port: %v", c.ListenAddr, err))
	}

	if c.ModelContextLength <= 0 {
		errs = append(errs, fmt.Sprintf("model_context_length must be > 0 (got %d)", c.ModelContextLength))
	}

	if c.UpstreamStartupWait < 0 {
		errs = append(errs, fmt.Sprintf("upstream_startup_wait must be >= 0 (got %s)", c.UpstreamStartupWait))
	}
	if c.UpstreamRetryInterval <= 0 {
		errs = append(errs, fmt.Sprintf("upstream_retry_interval must be > 0 (got %s)", c.UpstreamRetryInterval))
	}

	if c.MaxRequestBytes < 0 {
		errs = append(errs, fmt.Sprintf("max_request_bytes must be >= 0 (got %d)", c.MaxRequestBytes))
	}
	if c.ShutdownTimeout < 0 {
		errs = append(errs, fmt.Sprintf("shutdown_timeout must be >= 0 (got %s)", c.ShutdownTimeout))
	}
	if c.HTTPRequestTimeout < 0 {
		errs = append(errs, fmt.Sprintf("http_request_timeout must be >= 0 (got %s)", c.HTTPRequestTimeout))
	}
	if c.HTTPStreamTimeout < 0 {
		errs = append(errs, fmt.Sprintf("http_stream_timeout must be >= 0 (got %s)", c.HTTPStreamTimeout))
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
	return fallback
}

func envOrDuration(key string, fallback time.Duration) time.Duration {
	return fallback
}

func envOrInt(key string, fallback int) int {
	return fallback
}
