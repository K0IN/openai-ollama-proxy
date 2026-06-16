package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/k0in/openai-ollama-proxy/internal/config"
	"github.com/k0in/openai-ollama-proxy/internal/stats"
)

type Server struct {
	cfg           config.Config
	router        *config.RoutingTable
	client        *http.Client
	requestClient *http.Client
	stats         *stats.Stats
}

type modelMetadata struct {
	ContextLength  int
	Family         string
	ParentModel    string
	Format         string
	ParameterSize  string
	ParameterCount int64
	Quantization   string
}

var modelSizePattern = regexp.MustCompile(`(?i)(?:^|[^0-9a-z])(\d+(?:\.\d+)?)\s*([kmbt])(?:[^0-9a-z]|$)`)

// New constructs a Server that uses a single HTTP client for both streaming
// and short upstream calls. Prefer NewWithClients in production so that
// streaming completions cannot starve embeddings / health calls.
func New(cfg config.Config, router *config.RoutingTable, client *http.Client, st *stats.Stats) *Server {
	if client == nil {
		client = config.NewHTTPClient(cfg)
	}
	if st == nil {
		st = stats.New()
	}

	return &Server{cfg: cfg, router: router, client: client, requestClient: client, stats: st}
}

// NewWithClients constructs a Server backed by separate HTTP clients for
// streaming completions (streamClient) and short upstream calls
// (requestClient). Either argument may be nil to fall back to the package
// defaults.
func NewWithClients(cfg config.Config, router *config.RoutingTable, streamClient, requestClient *http.Client, st *stats.Stats) *Server {
	if streamClient == nil {
		streamClient = config.NewHTTPClient(cfg)
	}
	if requestClient == nil {
		requestClient = config.NewRequestHTTPClient(cfg)
	}
	if st == nil {
		st = stats.New()
	}

	return &Server{cfg: cfg, router: router, client: streamClient, requestClient: requestClient, stats: st}
}

// Stats returns the server's stats collector, for use by callers that need
// to persist stats (e.g. periodic save to disk).
func (server *Server) Stats() *stats.Stats {
	return server.stats
}

func (server *Server) Routes() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/api/generate", server.handleGenerate)
	mux.HandleFunc("/api/chat", server.handleChat)
	mux.HandleFunc("/api/embed", server.handleEmbed)
	mux.HandleFunc("/api/embeddings", server.handleEmbeddings)
	mux.HandleFunc("/api/pull", server.handlePull)
	mux.HandleFunc("/api/push", server.handlePush)
	mux.HandleFunc("/api/create", server.handleCreate)
	mux.HandleFunc("/api/copy", server.handleCopy)
	mux.HandleFunc("/api/delete", server.handleDelete)
	mux.HandleFunc("/api/blobs/", server.handleBlobs)
	mux.HandleFunc("/api/tags", server.handleTags)
	mux.HandleFunc("/api/show", server.handleShow)
	mux.HandleFunc("/api/version", server.handleVersion)
	mux.HandleFunc("/api/ps", server.handlePs)

	mux.HandleFunc("/models", server.handleOpenAIModels)
	mux.HandleFunc("/v1/models", server.handleOpenAIModels)
	mux.HandleFunc("/embeddings", server.handleOpenAIEmbeddings)
	mux.HandleFunc("/v1/embeddings", server.handleOpenAIEmbeddings)
	mux.HandleFunc("/chat/completions", server.handleOpenAIChat)
	mux.HandleFunc("/v1/chat/completions", server.handleOpenAIChat)

	mux.HandleFunc("/messages", server.handleAnthropicMessages)
	mux.HandleFunc("/v1/messages", server.handleAnthropicMessages)

	mux.HandleFunc("/stats", server.handleStats)

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			server.handleHead(w, r)
			return
		}
		server.handleRoot(w, r)
	})

	return mux
}

// resolveRouteForModel maps a user-facing model name to the upstream
// connection details via the RoutingTable.
func (server *Server) resolveRouteForModel(model string) (baseURL, apiKey, upstreamModel string, ctxLen int) {
	if server.router != nil {
		if entry, ok := server.router.Lookup(model); ok {
			return entry.URL, entry.APIKey, entry.UpstreamModel, entry.ContextLength
		}
	}
	// If model is not in the routing table, return defaults from the first upstream.
	if upstreams := server.router.AllUpstreams(); len(upstreams) > 0 {
		u := upstreams[0]
		if len(u.Models) > 0 {
			return u.URL, u.APIKey, u.Models[0].Upstream, server.cfg.ModelContextLength
		}
		return u.URL, u.APIKey, "", server.cfg.ModelContextLength
	}
	return "", "", "", server.cfg.ModelContextLength
}

// resolveEffectiveAPIKey returns the API key to use for the upstream request.
// If the upstream has passthrough enabled, the caller's API key (extracted from
// the incoming request) is used instead of the configured upstream api_key.
func (server *Server) resolveEffectiveAPIKey(upstreamAPIKey string, passthrough bool, incomingAPIKey string) string {
	if passthrough {
		return incomingAPIKey
	}
	return upstreamAPIKey
}

// resolveRouteForModelPassthrough extends resolveRouteForModel to also return
// the passthrough flag.
func (server *Server) resolveRouteForModelPassthrough(model string) (baseURL, apiKey, upstreamModel string, ctxLen int, passthrough bool) {
	if server.router != nil {
		if entry, ok := server.router.Lookup(model); ok {
			return entry.URL, entry.APIKey, entry.UpstreamModel, entry.ContextLength, entry.Passthrough
		}
	}
	if upstreams := server.router.AllUpstreams(); len(upstreams) > 0 {
		u := upstreams[0]
		if len(u.Models) > 0 {
			return u.URL, u.APIKey, u.Models[0].Upstream, server.cfg.ModelContextLength, u.Passthrough
		}
		return u.URL, u.APIKey, "", server.cfg.ModelContextLength, u.Passthrough
	}
	return "", "", "", server.cfg.ModelContextLength, false
}

func (server *Server) firstUpstreamModel() string {
	if server.router != nil {
		for _, m := range server.router.AllModels() {
			return m
		}
	}
	return ""
}

func applyModelNameHints(metadata *modelMetadata, name string) {
	lower := strings.ToLower(name)

	switch {
	case strings.Contains(lower, "qwen3"):
		metadata.Family = "qwen3"
	case strings.Contains(lower, "qwen2"):
		metadata.Family = "qwen2"
	case strings.Contains(lower, "qwen"):
		metadata.Family = "qwen"
	case strings.Contains(lower, "llama-3"), strings.Contains(lower, "llama3"):
		metadata.Family = "llama3"
	case strings.Contains(lower, "llama-2"), strings.Contains(lower, "llama2"):
		metadata.Family = "llama2"
	case strings.Contains(lower, "llama"):
		metadata.Family = "llama"
	case strings.Contains(lower, "mixtral"):
		metadata.Family = "mixtral"
	case strings.Contains(lower, "mistral"):
		metadata.Family = "mistral"
	case strings.Contains(lower, "gemma"):
		metadata.Family = "gemma"
	case strings.Contains(lower, "phi"):
		metadata.Family = "phi"
	case strings.Contains(lower, "deepseek"):
		metadata.Family = "deepseek"
	case strings.Contains(lower, "gpt-oss"), strings.Contains(lower, "gptoss"):
		metadata.Family = "gpt-oss"
	case strings.Contains(lower, "yi-") || strings.HasPrefix(lower, "yi/"):
		metadata.Family = "yi"
	case strings.Contains(lower, "command-r"), strings.Contains(lower, "commandr"):
		metadata.Family = "command-r"
	case strings.Contains(lower, "falcon"):
		metadata.Family = "falcon"
	case strings.Contains(lower, "granite"):
		metadata.Family = "granite"
	}

	switch {
	case strings.Contains(lower, "awq"):
		metadata.Quantization = "AWQ-4bit"
	case strings.Contains(lower, "nvfp4"):
		metadata.Quantization = "NVFP4"
	case strings.Contains(lower, "fp8"):
		metadata.Quantization = "FP8"
	case strings.Contains(lower, "gptq"):
		metadata.Quantization = "GPTQ"
	case strings.Contains(lower, "int8"):
		metadata.Quantization = "INT8"
	case strings.Contains(lower, "int4"):
		metadata.Quantization = "INT4"
	case strings.Contains(lower, "q8_0"):
		metadata.Quantization = "Q8_0"
	case strings.Contains(lower, "q6_k"):
		metadata.Quantization = "Q6_K"
	case strings.Contains(lower, "q5_k_m"), strings.Contains(lower, "q5_k"):
		metadata.Quantization = "Q5_K_M"
	case strings.Contains(lower, "q4_k_m"), strings.Contains(lower, "q4_k"):
		metadata.Quantization = "Q4_K_M"
	case strings.Contains(lower, "q4_0"):
		metadata.Quantization = "Q4_0"
	case strings.Contains(lower, "bf16"), strings.Contains(lower, "bfloat16"):
		metadata.Quantization = "BF16"
	case strings.Contains(lower, "fp16"), strings.Contains(lower, "float16"):
		metadata.Quantization = "FP16"
	}

	if size, count := modelParameterHint(name); size != "" {
		metadata.ParameterSize = size
		metadata.ParameterCount = count
	}

	switch {
	case strings.Contains(lower, "gguf"):
		metadata.Format = "gguf"
	case strings.Contains(lower, "safetensors"),
		strings.Contains(lower, "awq"),
		strings.Contains(lower, "fp8"),
		strings.Contains(lower, "nvfp4"),
		strings.Contains(lower, "gptq"):
		metadata.Format = "safetensors"
	}
}

func modelParameterHint(name string) (string, int64) {
	match := modelSizePattern.FindStringSubmatch(name)
	if len(match) != 3 {
		return "", 0
	}

	value, err := strconv.ParseFloat(match[1], 64)
	if err != nil || value <= 0 {
		return "", 0
	}

	unit := strings.ToUpper(match[2])
	scale := map[string]float64{
		"K": 1e3,
		"M": 1e6,
		"B": 1e9,
		"T": 1e12,
	}[unit]
	if scale == 0 {
		return "", 0
	}

	return match[1] + unit, int64(math.Round(value * scale))
}

func ollamaModelInfo(metadata modelMetadata) map[string]any {
	info := map[string]any{
		"general.architecture":         metadata.Family,
		"general.quantization_version": metadata.Quantization,
	}
	if metadata.ParameterCount > 0 {
		info["general.parameter_count"] = metadata.ParameterCount
	}
	if metadata.ContextLength > 0 {
		info[metadata.Family+".context_length"] = metadata.ContextLength
		info["general.context_length"] = metadata.ContextLength
	}
	return info
}

func (server *Server) probeUpstreamHealth(ctx context.Context) (bool, error) {
	upstreams := server.router.AllUpstreams()
	// No upstream configured — the proxy is self-sufficient (provides
	// synthetic /api/tags, /api/version, etc.) and is always healthy.
	if len(upstreams) == 0 {
		return true, nil
	}

	// Probe all upstreams; return healthy if at least one is reachable.
	var lastErr error
	for _, u := range upstreams {
		baseURL := strings.TrimRight(u.URL, "/")

		healthCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
		defer cancel()

		req, err := http.NewRequestWithContext(healthCtx, http.MethodGet, baseURL+"/v1/models", nil)
		if err != nil {
			lastErr = err
			continue
		}
		if u.APIKey != "" {
			req.Header.Set("Authorization", "Bearer "+u.APIKey)
		}

		resp, err := server.requestClient.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()

		if resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
			return true, nil
		}
		lastErr = fmt.Errorf("upstream %s returned status %d", u.URL, resp.StatusCode)
	}

	return false, lastErr
}

func (server *Server) doUpstreamChatWithRetryForRoute(ctx context.Context, payload []byte, baseURL, apiKey string) (*http.Response, error) {
	deadline := time.Now().Add(server.cfg.UpstreamStartupWait)
	if server.cfg.UpstreamRetryInterval <= 0 {
		server.cfg.UpstreamRetryInterval = 2 * time.Second
	}

	var lastErr error

	for {
		upstream, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/chat/completions", bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		upstream.Header.Set("Content-Type", "application/json")
		if apiKey != "" {
			upstream.Header.Set("Authorization", "Bearer "+apiKey)
		}

		resp, err := server.client.Do(upstream)
		if err == nil {
			if resp.StatusCode != http.StatusServiceUnavailable {
				return resp, nil
			}

			errBody, _ := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			lastErr = fmt.Errorf("upstream returned 503: %s", strings.TrimSpace(string(errBody)))
		} else {
			lastErr = err
		}

		if time.Now().After(deadline) {
			break
		}

		select {
		case <-ctx.Done():
			return nil, errors.New("request canceled")
		case <-time.After(server.cfg.UpstreamRetryInterval):
		}
	}

	if lastErr == nil {
		lastErr = errors.New("upstream not ready")
	}

	return nil, fmt.Errorf("upstream unavailable after %s: %w", server.cfg.UpstreamStartupWait, lastErr)
}

func copyResponseHeaders(dst http.ResponseWriter, src *http.Response) {
	for key, values := range src.Header {
		for _, value := range values {
			dst.Header().Add(key, value)
		}
	}
}

func isZeroKeepAlive(raw json.RawMessage) bool {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return false
	}

	var num json.Number
	if err := json.Unmarshal(raw, &num); err == nil {
		if intVal, err := num.Int64(); err == nil {
			return intVal == 0
		}
		if floatVal, err := num.Float64(); err == nil {
			return floatVal == 0
		}
	}

	var str string
	if err := json.Unmarshal(raw, &str); err == nil {
		str = strings.TrimSpace(str)
		if str == "" {
			return false
		}
		if intVal, err := strconv.ParseInt(str, 10, 64); err == nil {
			return intVal == 0
		}
		if floatVal, err := strconv.ParseFloat(str, 64); err == nil {
			return floatVal == 0
		}
		if d, err := time.ParseDuration(str); err == nil {
			return d == 0
		}
	}

	return false
}
