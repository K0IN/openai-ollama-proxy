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
	"github.com/k0in/openai-ollama-proxy/internal/types"
)

type Server struct {
	cfg    config.Config
	client *http.Client
	// requestClient is used for short upstream calls (embeddings, models list,
	// health probes). client is used for streaming chat/generate completions.
	requestClient *http.Client
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
func New(cfg config.Config, client *http.Client) *Server {
	if client == nil {
		client = config.NewHTTPClient(cfg)
	}

	return &Server{cfg: cfg, client: client, requestClient: client}
}

// NewWithClients constructs a Server backed by separate HTTP clients for
// streaming completions (streamClient) and short upstream calls
// (requestClient). Either argument may be nil to fall back to the package
// defaults.
func NewWithClients(cfg config.Config, streamClient, requestClient *http.Client) *Server {
	if streamClient == nil {
		streamClient = config.NewHTTPClient(cfg)
	}
	if requestClient == nil {
		requestClient = config.NewRequestHTTPClient(cfg)
	}

	return &Server{cfg: cfg, client: streamClient, requestClient: requestClient}
}

func (server *Server) Routes() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/api/generate", server.handleGenerate)
	mux.HandleFunc("/api/chat", server.handleChat)
	mux.HandleFunc("/api/embed", server.handleEmbed)
	mux.HandleFunc("/api/embeddings", server.handleEmbeddings)
	mux.HandleFunc("/api/pull", server.handlePull)
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

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodHead {
			server.handleHead(w, r)
			return
		}
		server.handleRoot(w, r)
	})

	return mux
}

func (server *Server) currentModelMetadata(ctx context.Context) modelMetadata {
	metadata := server.fallbackModelMetadata()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, server.cfg.VLLMBaseURL+"/v1/models", nil)
	if err != nil {
		return metadata
	}
	if server.cfg.VLLMAPIKey != "" {
		req.Header.Set("Authorization", "Bearer "+server.cfg.VLLMAPIKey)
	}

	resp, err := server.requestClient.Do(req)
	if err != nil {
		return metadata
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return metadata
	}

	var list types.OpenAIModelListResponse
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		return metadata
	}

	for _, model := range list.Data {
		if model.ID != server.cfg.VLLMModel && model.Root != server.cfg.VLLMModel {
			continue
		}
		if model.MaxModelLen > 0 {
			metadata.ContextLength = model.MaxModelLen
		}
		if model.Root != "" {
			metadata.ParentModel = model.Root
			applyModelNameHints(&metadata, model.Root)
		} else {
			applyModelNameHints(&metadata, model.ID)
		}
		return metadata
	}

	return metadata
}

func (server *Server) fallbackModelMetadata() modelMetadata {
	metadata := modelMetadata{
		ContextLength: server.cfg.ModelContextLength,
		Family:        "transformer",
		ParentModel:   server.cfg.VLLMModel,
		Format:        "unknown",
		ParameterSize: "unknown",
		Quantization:  "unknown",
	}
	applyModelNameHints(&metadata, server.cfg.VLLMModel)
	return metadata
}

func applyModelNameHints(metadata *modelMetadata, name string) {
	lower := strings.ToLower(name)

	// Family detection ordered by specificity (longest/most specific first).
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

	// Quantization hints.
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

	// Format hints.
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

func (server *Server) probeVLLMHealth(ctx context.Context) (bool, error) {
	healthCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()

	upstream, err := http.NewRequestWithContext(healthCtx, http.MethodGet, strings.TrimRight(server.cfg.VLLMBaseURL, "/")+"/health", nil)
	if err != nil {
		return false, err
	}
	if server.cfg.VLLMAPIKey != "" {
		upstream.Header.Set("Authorization", "Bearer "+server.cfg.VLLMAPIKey)
	}

	resp, err := server.requestClient.Do(upstream)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)

	return resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices, nil
}

func (server *Server) doUpstreamChatWithRetry(ctx context.Context, payload []byte) (*http.Response, error) {
	deadline := time.Now().Add(server.cfg.UpstreamStartupWait)
	if server.cfg.UpstreamRetryInterval <= 0 {
		server.cfg.UpstreamRetryInterval = 2 * time.Second
	}

	var lastErr error

	for {
		upstream, err := http.NewRequestWithContext(ctx, http.MethodPost, server.cfg.VLLMBaseURL+"/v1/chat/completions", bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		upstream.Header.Set("Content-Type", "application/json")
		if server.cfg.VLLMAPIKey != "" {
			upstream.Header.Set("Authorization", "Bearer "+server.cfg.VLLMAPIKey)
		}

		resp, err := server.client.Do(upstream)
		if err == nil {
			if resp.StatusCode != http.StatusServiceUnavailable {
				return resp, nil
			}

			errBody, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
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

	return nil, fmt.Errorf("vLLM unavailable after %s: %w", server.cfg.UpstreamStartupWait, lastErr)
}

func copyResponseHeaders(dst http.ResponseWriter, src *http.Response) {
	for key, values := range src.Header {
		for _, value := range values {
			dst.Header().Add(key, value)
		}
	}
}

// isZeroKeepAlive reports whether the Ollama keep_alive value explicitly
// requests an immediate unload. The value can be a JSON number (seconds),
// or a JSON string in either bare-integer ("0") or duration ("5m") form.
// Returns false for missing / unparseable values to avoid spurious unloads.
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
