package config

import (
	"log"
	"net/http"
	"os"
	"strconv"
	"time"
)

type Config struct {
	ListenAddr            string
	VLLMBaseURL           string
	VLLMAPIKey            string
	VLLMModel             string
	ModelName             string
	ModelContextLength    int
	OllamaVersion         string
	UpstreamStartupWait   time.Duration
	UpstreamRetryInterval time.Duration
	Debug                 bool
}

func Load() Config {
	return Config{
		ListenAddr:            envOr("LISTEN_ADDR", ":11434"),
		VLLMBaseURL:           envOr("VLLM_BASE_URL", "http://localhost:8000"),
		VLLMAPIKey:            envOr("VLLM_API_KEY", ""),
		VLLMModel:             envOr("VLLM_MODEL", "default"),
		ModelName:             envOr("MODEL_NAME", "qwen3:latest"),
		ModelContextLength:    envOrInt("MODEL_CONTEXT_LENGTH", 65536),
		OllamaVersion:         envOr("OLLAMA_VERSION", "0.7.0"),
		UpstreamStartupWait:   envOrDuration("VLLM_STARTUP_WAIT", 30*time.Minute),
		UpstreamRetryInterval: envOrDuration("VLLM_RETRY_INTERVAL", 2*time.Second),
		Debug:                 os.Getenv("DEBUG") == "true" || os.Getenv("DEBUG") == "1",
	}
}

func NewHTTPClient() *http.Client {
	return &http.Client{Timeout: 10 * time.Minute}
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
		log.Printf("invalid %s=%q, using default %s", key, value, fallback)
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
		log.Printf("invalid %s=%q, using default %d", key, value, fallback)
		return fallback
	}

	return parsed
}
