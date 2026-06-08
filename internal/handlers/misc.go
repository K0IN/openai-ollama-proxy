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
	_, _ = io.Copy(io.Discard, r.Body)
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
	metadata := server.currentModelMetadata(r.Context())
	resp := types.OllamaTagsResponse{
		Models: []types.OllamaModelInfo{{
			Name:       server.cfg.ModelName,
			Model:      server.cfg.ModelName,
			ModifiedAt: time.Now().UTC().Format(time.RFC3339),
			Size:       0,
			Digest:     "proxy",
			Details:    toModelDetails(metadata),
		}},
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (server *Server) handleShow(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	metadata := server.currentModelMetadata(r.Context())
	resp := types.OllamaShowResponse{
		Modelfile:    "# proxied model",
		Parameters:   fmt.Sprintf("num_ctx %d", metadata.ContextLength),
		Template:     "",
		Details:      toModelDetails(metadata),
		ModelInfo:    ollamaModelInfo(metadata),
		Capabilities: []string{"completion", "tools"},
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
	metadata := server.currentModelMetadata(r.Context())
	resp := types.OllamaPsResponse{
		Models: []types.OllamaPsModel{{
			Name:      server.cfg.ModelName,
			Model:     server.cfg.ModelName,
			Size:      0,
			Digest:    "proxy",
			Details:   toModelDetails(metadata),
			ExpiresAt: time.Now().Add(24 * time.Hour).UTC().Format(time.RFC3339),
			SizeVRAM:  0,
		}},
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
