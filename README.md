# openai-ollama-proxy

Proxy that translates Ollama API requests to OpenAI-compatible API requests. Lets Ollama clients talk to vLLM.

Works with Github copilot (Ollama integration) and the Ollama CLI.

> **Tested on RTX 5090** — All benchmarks and examples in this repository were validated on an NVIDIA RTX 5090 GPU.

## Why i released this

This was a quick hack to get GitHub Copilot working with vLLM (using vibecoding). I just needed it for myself and used it happily for a while. 

But lately github has announced their copilot plans will go up in price and use token based pricing, so to give users CHOICE and CONTROL over their data and costs, I decided to clean it up a bit and release it as an open source project.

So you can use, YOUR models with YOUR development tools as you want, and not be forced into a specific ecosystem or pricing model or using any official open-ai api (this project will be archived, when github copilot support openai api with configurable endpoints).

## Architecture

```
GitHub Copilot (vscode) / Ollama client
        │
        ▼
┌─────────────────┐
│  ollama-proxy   │  :11434
│  (this project) │
└────────┬────────┘
         │  OpenAI API
         ▼
┌─────────────────┐
│     vLLM        │  :8000
│  (GPU backend)  │
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
| [examples/docker-compose-qwen2.5-coder-14b.yml](examples/docker-compose-qwen2.5-coder-14b.yml) | Qwen2.5-Coder-14B (GPTQ Int4) | Coding model, ~14B params |

```bash
# Run with Qwen3.6-27B
docker compose -f examples/docker-compose-qwen3-27b.yml up -d

# Run with Qwen3.6-35B
docker compose -f examples/docker-compose-qwen3-35b.yml up -d

# Run with Qwen2.5-Coder-14B
docker compose -f examples/docker-compose-qwen2.5-coder-14b.yml up -d
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

## Quickstart

```bash
cd examples
docker compose -f docker-compose-qwen3-27b.yml up -d
```

## Test with Ollama CLI

Using the proxy as ollama cli endpoint (make sure ollama is not running on the same port):

```bash
OLLAMA_HOST=http://localhost:11434 ollama run qwen3:latest "hello"

# or run ollama in docker
docker run -it --entrypoint ollama -e OLLAMA_HOST="http://host.docker.internal:11434" docker.io/ollama/ollama run qwen3:latest "hello"
```
