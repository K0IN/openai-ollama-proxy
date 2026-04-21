package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	ListenAddr            string
	VLLMBaseURL           string
	VLLMAPIKey            string
	VLLMModel             string
	ModelName             string // Ollama model name exposed to clients
	ModelContextLength    int
	OllamaVersion         string
	UpstreamStartupWait   time.Duration
	UpstreamRetryInterval time.Duration
	Debug                 bool // log request/response bodies
}

var cfg Config

var httpClient = &http.Client{
	Timeout: 10 * time.Minute, // generous for long generations
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envOrDuration(key string, fallback time.Duration) time.Duration {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		log.Printf("invalid %s=%q, using default %s", key, v, fallback)
		return fallback
	}
	return d
}

func envOrInt(key string, fallback int) int {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	value, err := strconv.Atoi(v)
	if err != nil {
		log.Printf("invalid %s=%q, using default %d", key, v, fallback)
		return fallback
	}
	return value
}

// statusRecorder wraps ResponseWriter to capture the status code for logging.
type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

// Flush forwards to the underlying ResponseWriter if it supports http.Flusher.
// Without this, all w.(http.Flusher) assertions in handlers fail and streaming breaks.
func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// loggingMiddleware logs every request with method, path, status, and duration.
// When cfg.Debug is true it also logs full request bodies and all headers.
func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}

		ua := r.Header.Get("User-Agent")
		if ua == "" {
			ua = "-"
		}

		if cfg.Debug {
			// Log all request headers
			var hdrs strings.Builder
			for k, vals := range r.Header {
				for _, v := range vals {
					hdrs.WriteString(fmt.Sprintf("  %s: %s\n", k, v))
				}
			}
			log.Printf(">>> %s %s | ua=%q\n%s", r.Method, r.URL.String(), ua, hdrs.String())

			// Log full request body
			if r.Body != nil && r.Method == http.MethodPost {
				body, _ := io.ReadAll(r.Body)
				r.Body = io.NopCloser(bytes.NewReader(body))
				if len(body) > 0 {
					var buf bytes.Buffer
					if err := json.Indent(&buf, body, "  ", "  "); err == nil {
						log.Printf(">>> REQUEST BODY (%d bytes):\n  %s", len(body), buf.String())
					} else {
						log.Printf(">>> REQUEST BODY (%d bytes): %s", len(body), string(body))
					}
				}
			}
		}

		next.ServeHTTP(rec, r)

		dur := time.Since(start).Round(time.Millisecond)
		log.Printf("<<< %s %s %d %s | ua=%q", r.Method, r.URL.Path, rec.status, dur, ua)
	})
}

func main() {
	cfg = Config{
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

	mux := http.NewServeMux()

	// Ollama API endpoints
	mux.HandleFunc("/api/generate", handleGenerate)
	mux.HandleFunc("/api/chat", handleChat)
	mux.HandleFunc("/api/embed", handleEmbed)
	mux.HandleFunc("/api/embeddings", handleEmbeddings)
	mux.HandleFunc("/api/pull", handlePull)
	mux.HandleFunc("/api/create", handleCreate)
	mux.HandleFunc("/api/copy", handleCopy)
	mux.HandleFunc("/api/delete", handleDelete)
	mux.HandleFunc("/api/blobs/", handleBlobs)
	mux.HandleFunc("/api/tags", handleTags)
	mux.HandleFunc("/api/show", handleShow)
	mux.HandleFunc("/api/version", handleVersion)
	mux.HandleFunc("/api/ps", handlePs)

	// OpenAI-compatible endpoints for VS Code Copilot Custom OpenAI and similar clients.
	mux.HandleFunc("/models", handleOpenAIModels)
	mux.HandleFunc("/v1/models", handleOpenAIModels)
	mux.HandleFunc("/embeddings", handleOpenAIEmbeddings)
	mux.HandleFunc("/v1/embeddings", handleOpenAIEmbeddings)
	mux.HandleFunc("/chat/completions", handleOpenAIChat)
	mux.HandleFunc("/v1/chat/completions", handleOpenAIChat)

	// Health / root
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			handleHead(w, r)
			return
		}
		handleRoot(w, r)
	})

	log.Printf("openai-ollama-proxy listening on %s", cfg.ListenAddr)
	log.Printf("  vLLM backend: %s", cfg.VLLMBaseURL)
	log.Printf("  vLLM model:   %s", cfg.VLLMModel)
	log.Printf("  Ollama model: %s", cfg.ModelName)
	if cfg.Debug {
		log.Printf("  debug logging: enabled")
	}

	if err := http.ListenAndServe(cfg.ListenAddr, loggingMiddleware(mux)); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
