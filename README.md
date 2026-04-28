# openai-ollama-proxy

Proxy that translates Ollama API requests to OpenAI-compatible API requests. Lets Ollama clients talk to vLLM.

Works with VS Code (Ollama integration, Copilot Custom OpenAI) and the Ollama CLI — no client changes needed.

> **Tested on RTX 5090** — All benchmarks and examples in this repository were validated on an NVIDIA RTX 5090 GPU.

## Architecture

```
VS Code / Ollama client
        │
        ▼
┌─────────────────┐
│  ollama-proxy    │  :11434
│  (this project)  │
└────────┬────────┘
         │  OpenAI API
         ▼
┌─────────────────┐
│     vLLM         │  :8000
│  (GPU backend)   │
└─────────────────┘
```

## Run

### Locally

```bash
# Build
go build -o openai-ollama-proxy ./cmd/proxy

# Run
VLLM_BASE_URL=http://localhost:8000 VLLM_MODEL=your-model ./openai-ollama-proxy
```

### Install globally

```bash
go install github.com/k0in/openai-ollama-proxy@latest
openai-ollama-proxy
```

### Docker

```bash
# Build
docker build -t openai-ollama-proxy .

# Run
docker run -p 11434:11434 \
  -e VLLM_BASE_URL=http://host.docker.internal:8000 \
  -e VLLM_MODEL=your-model \
  openai-ollama-proxy
```

### Docker Compose Examples

Pre-configured examples for different models are available in the `examples/` directory:

| Example | Model | Description |
|---------|-------|-------------|
| [examples/docker-compose-qwen3-27b.yml](examples/docker-compose-qwen3-27b.yml) | Qwen3.6-27B (NVFP4) | Smaller, faster, ~27B params |
| [examples/docker-compose-qwen3-35b.yml](examples/docker-compose-qwen3-35b.yml) | Qwen3.6-35B (AWQ) | Mixture of Experts, ~35B params |

```bash
# Run with Qwen3.6-27B
docker compose -f examples/docker-compose-qwen3-27b.yml up -d

# Run with Qwen3.6-35B
docker compose -f examples/docker-compose-qwen3-35b.yml up -d
```

### Environment variables

| Variable | Default | Notes |
|---|---|---|
| `LISTEN_ADDR` | `:11434` | host:port the proxy binds to |
| `VLLM_BASE_URL` | `http://localhost:8000` | upstream vLLM, must be `http(s)://host[:port]` |
| `VLLM_API_KEY` | *(empty)* | sent as `Authorization: Bearer …`; required when vLLM enforces it |
| `VLLM_MODEL` | `default` | model id presented to vLLM |
| `MODEL_NAME` | `qwen3:latest` | model name presented to Ollama clients |
| `MODEL_CONTEXT_LENGTH` | `65536` | reported via `/api/show` and `/v1/models` |
| `OLLAMA_VERSION` | `0.6.4` | reported by `/api/version` |
| `VLLM_STARTUP_WAIT` | `30m` | retry budget while vLLM is loading the model |
| `VLLM_RETRY_INTERVAL` | `2s` | delay between startup retries |
| `HTTP_REQUEST_TIMEOUT` | `30s` | cap for short upstream calls (embeddings, models, health) |
| `HTTP_STREAM_TIMEOUT` | `5m` | cap for streaming chat / generate requests |
| `SHUTDOWN_TIMEOUT` | `30s` | drain budget for in-flight requests on SIGTERM/SIGINT |
| `MAX_REQUEST_BYTES` | `33554432` | reject inbound JSON bodies larger than this (32 MiB) |
| `DEBUG` | *(empty)* | `true`/`1` enables request/response dumps with secrets redacted |

A copy-paste template lives in [`.env.example`](.env.example).

## Quickstart

```bash
cp .env.example .env          # then edit VLLM_API_KEY (and HF_TOKEN if needed)
docker compose up -d
```

vLLM + proxy running with Qwen 3.6. Done.

## Test with Ollama CLI

Using the proxy as ollama cli endpoint (make sure ollama is not running on the same port):

```bash
OLLAMA_HOST=http://localhost:11434 ollama run qwen3:latest "hello"

# or with docker
docker run -it --entrypoint ollama -e OLLAMA_HOST="http://host.docker.internal:11434" docker.io/ollama/ollama run qwen:latest
```

## Production deployment

The proxy is a single statically-linked binary (~10 MB) that can run from
`scratch`. Recommended deployment shape:

- **Run behind a reverse proxy** (nginx / Caddy / Envoy) that terminates TLS
  and supplies CORS, rate limiting, and per-client authentication. The proxy
  itself only speaks plaintext HTTP and trusts the network it sits on.
- **Container is non-root** (UID 65532). Mount nothing writable.
- **Set explicit timeouts**: shorten `HTTP_STREAM_TIMEOUT` to your worst-case
  completion latency, lower `HTTP_REQUEST_TIMEOUT` if you are not using long
  embeddings, and tune `MAX_REQUEST_BYTES` to your real client payloads.
- **SIGTERM is honoured**: orchestrators get a clean drain up to
  `SHUTDOWN_TIMEOUT`. Set your readiness probe on `GET /` (returns 503 while
  vLLM is unhealthy).
- **Resource sizing**: the proxy is I/O-bound and effectively memory-free
  per request once `MAX_REQUEST_BYTES` is in place. 64 MiB RAM and 0.1 vCPU
  is plenty for a single-tenant deployment.

Example minimal Kubernetes deployment fragment:

```yaml
spec:
  containers:
    - name: proxy
      image: ghcr.io/k0in/openai-ollama-proxy:latest
      ports: [{containerPort: 11434}]
      env:
        - name: VLLM_BASE_URL
          value: http://vllm:8000
        - name: VLLM_API_KEY
          valueFrom: {secretKeyRef: {name: vllm, key: api-key}}
        - name: HTTP_STREAM_TIMEOUT
          value: 3m
      readinessProbe:
        httpGet: {path: /, port: 11434}
        periodSeconds: 5
      resources:
        requests: {cpu: "50m", memory: "64Mi"}
        limits:   {cpu: "500m", memory: "256Mi"}
      securityContext:
        runAsNonRoot: true
        readOnlyRootFilesystem: true
        allowPrivilegeEscalation: false
        capabilities: {drop: ["ALL"]}
```

## Security considerations

This proxy is designed for **trusted networks** (loopback, VPN, in-cluster
service mesh). The following are intentionally **out of scope** and must be
provided by something in front of it:

- **TLS / HTTPS termination** — the proxy listens on plain HTTP.
- **Per-client authentication / authorization** — there is one shared
  `VLLM_API_KEY`; treat anyone who can reach the listen address as fully
  authorized to use the upstream model.
- **Rate limiting / quotas** — none. Front it with nginx `limit_req`,
  Envoy rate limiter, or your gateway of choice.
- **CORS** — no CORS headers are emitted. Set them on your reverse proxy if
  browser clients need them.
- **WAF / abuse protection** — none.

What the proxy *does* enforce on its own:

- Inbound bodies are capped at `MAX_REQUEST_BYTES` (default 32 MiB) via
  `http.MaxBytesReader`. Larger requests get HTTP 413.
- Per-route HTTP client timeouts (`HTTP_REQUEST_TIMEOUT`,
  `HTTP_STREAM_TIMEOUT`) bound the time a single upstream call can hold a
  goroutine.
- Graceful shutdown drains in-flight requests for up to `SHUTDOWN_TIMEOUT`.
- Debug logging redacts `Authorization`, `Cookie`, `X-Api-Key`,
  `X-Auth-Token`, and any JSON body field whose name contains `api_key`,
  `apikey`, `authorization`, `password`, `secret`, or `token`.
- The container image runs as UID 65532 (non-root) on `scratch`.

If you intend to expose this directly to the public internet, **don't** —
add a real gateway first.
