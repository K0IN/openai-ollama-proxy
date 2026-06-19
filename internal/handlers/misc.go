package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/k0in/openai-ollama-proxy/internal/types"
)

func (server *Server) handlePull(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	defer func() { _ = r.Body.Close() }()
	var req types.OllamaPullRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && !errors.Is(err, io.EOF) {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	writeProgressObjects(w, req.Stream, buildPullProgressResponses(req))
}

func (server *Server) handleCreate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	defer func() { _ = r.Body.Close() }()
	var req struct {
		Model    string            `json:"model"`
		From     string            `json:"from,omitempty"`
		Stream   *bool             `json:"stream,omitempty"`
		System   string            `json:"system,omitempty"`
		Template string            `json:"template,omitempty"`
		Quantize string            `json:"quantize,omitempty"`
		Files    map[string]string `json:"files,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && !errors.Is(err, io.EOF) {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	writeProgressObjects(w, req.Stream, buildCreateProgressResponses(req.System, req.Template, req.Quantize, req.Files))
}

func (server *Server) handleCopy(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer func() { _ = r.Body.Close() }()
	var req struct {
		Source      string `json:"source"`
		Destination string `json:"destination"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && !errors.Is(err, io.EOF) {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.Source == "" || req.Destination == "" {
		http.Error(w, "source and destination are required", http.StatusBadRequest)
		return
	}
	w.WriteHeader(http.StatusOK)
}

func (server *Server) handlePush(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	defer func() { _ = r.Body.Close() }()
	var req types.OllamaPullRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && !errors.Is(err, io.EOF) {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	writeProgressObjects(w, req.Stream, buildPushProgressResponses(req))
}

func (server *Server) handleDelete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer func() { _ = r.Body.Close() }()
	_, _ = io.Copy(io.Discard, r.Body)
	w.WriteHeader(http.StatusOK)
}

func (server *Server) handleBlobs(w http.ResponseWriter, r *http.Request) {
	defer func() { _ = r.Body.Close() }()
	_, _ = io.Copy(io.Discard, r.Body)

	switch r.Method {
	case http.MethodHead:
		w.WriteHeader(http.StatusOK)
	case http.MethodPost:
		w.WriteHeader(http.StatusCreated)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func (server *Server) handleTags(w http.ResponseWriter, r *http.Request) {
	models := server.router.AllModels()
	resp := types.OllamaTagsResponse{
		Models: make([]types.OllamaModelInfo, 0, len(models)),
	}

	now := time.Now().UTC().Format(time.RFC3339)

	for _, m := range models {
		entry, _ := server.router.Lookup(m)
		meta := modelMetadata{
			ContextLength: entry.ContextLength,
			Family:        "transformer",
			Format:        "unknown",
			ParameterSize: "unknown",
			Quantization:  "unknown",
		}
		applyModelNameHints(&meta, m)
		resp.Models = append(resp.Models, types.OllamaModelInfo{
			Name:       m,
			Model:      m,
			ModifiedAt: now,
			Size:       0,
			Digest:     "sha256:proxy",
			Details:    toModelDetails(meta),
		})
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (server *Server) handleShow(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req types.OllamaShowRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && err.Error() != "EOF" {
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}

	entry, ok := server.router.Lookup(req.Model)
	meta := modelMetadata{
		ContextLength: server.cfg.ModelContextLength,
		Family:        "transformer",
		Format:        "unknown",
		ParameterSize: "unknown",
		Quantization:  "unknown",
	}
	if ok {
		meta.ContextLength = entry.ContextLength
		meta.ParentModel = entry.UpstreamModel
		applyModelNameHints(&meta, req.Model)
		applyModelNameHints(&meta, entry.UpstreamModel)
	} else {
		applyModelNameHints(&meta, req.Model)
	}

	capabilities := []string{"completion", "tools"}
	if entry.SupportsVision {
		capabilities = append(capabilities, "vision")
	}
	if entry.SupportsThinking {
		capabilities = append(capabilities, "thinking")
	}

	resp := types.OllamaShowResponse{
		Modelfile:    "# proxied model",
		Parameters:   fmt.Sprintf("num_ctx %d", meta.ContextLength),
		Template:     "",
		Details:      toModelDetails(meta),
		ModelInfo:    ollamaModelInfo(meta),
		Capabilities: capabilities,
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (server *Server) handleVersion(w http.ResponseWriter, r *http.Request) {
	resp := types.OllamaVersionResponse{Version: server.cfg.OllamaVersion}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (server *Server) handlePs(w http.ResponseWriter, r *http.Request) {
	models := server.router.AllModels()
	resp := types.OllamaPsResponse{
		Models: make([]types.OllamaPsModel, 0, len(models)),
	}

	now := time.Now().UTC()
	expiresAt := now.Add(24 * time.Hour).UTC().Format(time.RFC3339)

	for _, m := range models {
		entry, _ := server.router.Lookup(m)
		meta := modelMetadata{
			ContextLength: entry.ContextLength,
			Family:        "transformer",
			Format:        "unknown",
			ParameterSize: "unknown",
			Quantization:  "unknown",
		}
		applyModelNameHints(&meta, m)
		resp.Models = append(resp.Models, types.OllamaPsModel{
			Name:      m,
			Model:     m,
			Size:      0,
			Digest:    "sha256:proxy",
			Details:   toModelDetails(meta),
			ExpiresAt: expiresAt,
			SizeVRAM:  0,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (server *Server) handleHead(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	healthy, err := server.probeUpstreamHealth(r.Context())
	if err != nil || !healthy {
		w.WriteHeader(http.StatusServiceUnavailable)
		return
	}

	w.WriteHeader(http.StatusOK)
}

func (server *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	w.Header().Set("Content-Type", "text/plain")
	healthy, err := server.probeUpstreamHealth(r.Context())
	if err != nil || !healthy {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = fmt.Fprint(w, "Ollama is down")
		return
	}

	_, _ = fmt.Fprint(w, "Ollama is running")
}

func (server *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	snapshot := server.stats.Snapshot()

	perModelStats := make(map[string]interface{}, len(snapshot.PerModel))
	for model, ms := range snapshot.PerModel {
		perModelStats[model] = map[string]interface{}{
			"total_input_tokens":    ms.TotalInput,
			"total_output_tokens":   ms.TotalOutput,
			"total_tokens":          ms.TotalTokens,
			"total_requests":        ms.Requests,
			"output_tokens_per_sec": ms.OutputTokensPerSec,
		}
	}

	resp := map[string]interface{}{
		"model": snapshot.Model,
		"stats": map[string]interface{}{
			"total_input_tokens":        snapshot.TotalInput,
			"total_output_tokens":       snapshot.TotalOutput,
			"total_tokens":              snapshot.TotalInput + snapshot.TotalOutput,
			"total_requests":            snapshot.Requests,
			"uptime_seconds":            snapshot.Uptime.Seconds(),
			"current_input_tokens":      snapshot.CurrentInput,
			"current_output_tokens":     snapshot.CurrentOutput,
			"input_tokens_per_sec":      snapshot.InputPerSecond,
			"output_tokens_per_sec":     snapshot.OutputPerSecond,
			"tokens_per_sec":            snapshot.InputPerSecond + snapshot.OutputPerSecond,
			"avg_input_tokens_per_sec":  snapshot.AvgInputTokensPerSec,
			"avg_output_tokens_per_sec": snapshot.AvgOutputTokensPerSec,
			"avg_tokens_per_sec":        snapshot.AvgTokensPerSec,
			"per_model":                 perModelStats,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func toModelDetails(metadata modelMetadata) types.OllamaModelDetails {
	return types.OllamaModelDetails{
		ParentModel:       metadata.ParentModel,
		Format:            metadata.Format,
		Family:            metadata.Family,
		Families:          []string{metadata.Family},
		ParameterSize:     metadata.ParameterSize,
		QuantizationLevel: metadata.Quantization,
	}
}

func writeProgressObjects(w http.ResponseWriter, stream *bool, responses []types.OllamaProgressResponse) {
	if len(responses) == 0 {
		w.WriteHeader(http.StatusOK)
		return
	}

	if stream == nil || *stream {
		w.Header().Set("Content-Type", "application/x-ndjson")
		encoder := json.NewEncoder(w)
		for _, response := range responses {
			_ = encoder.Encode(response)
		}
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(responses[len(responses)-1])
}

func buildPullProgressResponses(req types.OllamaPullRequest) []types.OllamaProgressResponse {
	name := req.Model
	if name == "" {
		name = "digestname"
	}

	return []types.OllamaProgressResponse{
		{Status: "pulling manifest"},
		{
			Status:    fmt.Sprintf("pulling %s", name),
			Digest:    fmt.Sprintf("sha256:%x", name),
			Total:     int64(max(1, len(name))) * 1024,
			Completed: int64(max(1, len(name))) * 1024,
		},
		{Status: "verifying sha256 digest"},
		{Status: "writing manifest"},
		{Status: "removing any unused layers"},
		{Status: "success"},
	}
}

func buildPushProgressResponses(req types.OllamaPullRequest) []types.OllamaProgressResponse {
	name := req.Model
	if name == "" {
		name = "proxied-model"
	}

	return []types.OllamaProgressResponse{
		{Status: "retrieving manifest"},
		{
			Status:    fmt.Sprintf("starting upload %s", name),
			Digest:    fmt.Sprintf("sha256:%x", name),
			Total:     int64(max(1, len(name))) * 1024,
			Completed: int64(max(1, len(name))) * 1024,
		},
		{Status: "pushing manifest"},
		{Status: "success"},
	}
}

func buildCreateProgressResponses(system string, template string, quantize string, files map[string]string) []types.OllamaProgressResponse {
	if quantize != "" {
		label := strings.ToUpper(quantize)
		return []types.OllamaProgressResponse{
			{Status: fmt.Sprintf("quantizing F16 model to %s", label), Digest: "0", Total: 1, Completed: 1},
			{Status: "verifying conversion"},
			{Status: "creating new layer sha256:proxy"},
			{Status: "writing manifest"},
			{Status: "success"},
		}
	}

	if len(files) > 0 {
		fileNames := make([]string, 0, len(files))
		for name := range files {
			fileNames = append(fileNames, name)
		}
		sort.Strings(fileNames)

		for _, name := range fileNames {
			if strings.HasSuffix(strings.ToLower(name), ".gguf") {
				return []types.OllamaProgressResponse{
					{Status: "parsing GGUF"},
					{Status: fmt.Sprintf("using existing layer %s", files[name])},
					{Status: "writing manifest"},
					{Status: "success"},
				}
			}
		}

		return []types.OllamaProgressResponse{
			{Status: "converting model"},
			{Status: "creating new layer sha256:proxy"},
			{Status: "using autodetected template llama3-instruct"},
			{Status: "writing manifest"},
			{Status: "success"},
		}
	}

	responses := []types.OllamaProgressResponse{{Status: "reading model metadata"}}
	if system != "" {
		responses = append(responses, types.OllamaProgressResponse{Status: "creating system layer"})
	}
	if template != "" {
		responses = append(responses, types.OllamaProgressResponse{Status: "creating template layer"})
	}
	responses = append(
		responses,
		types.OllamaProgressResponse{Status: "writing layer sha256:proxy"},
		types.OllamaProgressResponse{Status: "writing manifest"},
		types.OllamaProgressResponse{Status: "success"},
	)

	return responses
}
