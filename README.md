# openai-ollama-proxy

Proxy that translates Ollama API requests to OpenAI-compatible API requests. Lets Ollama clients talk to **any** OpenAI-compatible API (vLLM, Ollama, LM Studio, localai, etc.).

Also includes **Anthropic Messages API support** — meaning tools like **[Claude Code](https://github.com/anthropics/claude-code)** can use the proxy to talk to any OpenAI-compatible backend.

Works with Github copilot (Ollama integration), Ollama CLI, and Claude Code CLI.

> **Tested on RTX 5090** — All benchmarks and examples in this repository were validated on an NVIDIA RTX 5090 GPU.

## Why i released this

This was a quick hack to get GitHub Copilot working with local LLMs via OpenAI-compatible APIs (using vibecoding). I just needed it for myself and used it happily for a while. 

But lately github has announced their copilot plans will go up in price and use token based pricing, so to give users CHOICE and CONTROL over their data and costs, I decided to clean it up a bit and release it as an open source project.

So you can use YOUR models with YOUR development tools as you want, and not be forced into a specific ecosystem or pricing model or using any official open-ai api (this project will be archived, when github copilot support openai api with configurable endpoints).

## Architecture

```
GitHub Copilot (vscode) / Ollama client / Claude Code CLI
        │
        ▼
┌─────────────────┐
│  ollama-proxy   │  :11434
│  (this project) │
└────────┬────────┘
         │  OpenAI API
         ▼
┌──────────────────────┐
│  OpenAI-compatible   │  :8000
│  API (vLLM/Ollama/…) │
└──────────────────────┘
```

### Anthropic API route

The proxy exposes Anthropic Messages API-compatible endpoints at `/v1/messages` and `/messages`. This allows Anthropic SDK clients and tools like Claude Code to connect to the proxy and use any OpenAI-compatible upstream backend. The proxy performs bidirectional translation:

- Anthropic **request** → OpenAI chat completion request (upstream)
- OpenAI chat completion **response** → Anthropic Messages API response (to client)
- Streaming events from OpenAI SSE are translated to Anthropic SSE events (`message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`)
```

## Run

### Locally

```bash
# Build
go build -o openai-ollama-proxy ./cmd/proxy

# Run (edit proxy.toml first with your upstream settings)
cp proxy.toml my-config.toml
# ... edit my-config.toml ...
CONFIG_FILE=./my-config.toml ./openai-ollama-proxy
```

### Docker

(note: im using podman for development, docker commands should work but are not tested)

```bash
# Pull
docker pull ghcr.io/k0in/openai-ollama-proxy:latest

# Run (mount your own TOML config)
docker run -p 11434:11434 \
  -v ./my-config.toml:/proxy.toml:ro \
  -e CONFIG_FILE=/proxy.toml \
  ghcr.io/k0in/openai-ollama-proxy:latest
```

### Docker Compose

The [docker-compose.yml](docker-compose.yml) already configures the proxy with `CONFIG_FILE` and mounts `proxy.toml`. Edit `proxy.toml` with your upstream settings, then:

```bash
docker compose up -d
```

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

## Configuration

The proxy is configured via a **TOML file** (see [proxy.toml](proxy.toml) for a complete example).

Set the `CONFIG_FILE` environment variable to point to your TOML file:

```bash
export CONFIG_FILE=/path/to/proxy.toml
./openai-ollama-proxy
```

### TOML reference

```toml
# Core settings
listen_addr = ":11434"          # host:port the proxy binds to
ollama_version = "0.6.4"       # reported by /api/version
model_context_length = 65536   # reported via /api/show and /v1/models

# Optional settings
proxy_api_key = "sk-..."           # require clients to send this via Authorization: Bearer
debug = false                       # enable request/response dumps (secrets redacted)
log_max_body_bytes = 4096           # truncate debug-logged bodies
stats_store_path = "/tmp/stats.json" # persist stats across restarts
upstream_startup_wait = "30m"       # retry budget while upstream loads the model
upstream_retry_interval = "2s"      # delay between startup retries
http_request_timeout = "30s"        # cap for short upstream calls (embeddings, models, health)
http_stream_timeout = "5m"          # cap for streaming chat / generate requests
shutdown_timeout = "30s"            # drain budget for in-flight requests on SIGTERM/SIGINT
max_request_bytes = 33554432        # reject inbound JSON bodies larger than this (32 MiB)

# Upstream definitions
[[upstream]]
url = "http://vllm:8000"           # OpenAI-compatible API endpoint

# Each upstream uses either a static api_key or passthrough (not both).
# Option A: pre-configured api_key (sent as Authorization: Bearer for all requests)
api_key = "change-me"
# Option B: passthrough — forward the client's API key as-is to the upstream
# passthrough = true

[[upstream.models]]
upstream = "Qwen3.6"               # model name sent to upstream
local = "qwen3:latest"             # model name presented to Ollama/OpenAI clients
context_length = 96000              # optional, overrides global model_context_length
```

### Multiple upstreams

You can define multiple `[[upstream]]` blocks, each with its own models. Requests are routed based on the local model name:

```toml
[[upstream]]
url = "http://localhost:8000"
api_key = "sk-local"

[[upstream.models]]
upstream = "qwen2.5-coder-14b"
local = "qwen-coder"
context_length = 32768

[[upstream.models]]
upstream = "qwen3-27b-fp8"
local = "qwen3-large"

[[upstream]]
url = "https://api.openai.com"
api_key = "sk-..."

[[upstream.models]]
upstream = "gpt-4o"
local = "gpt-4o"
context_length = 128000
```

### Environment variable placeholders

Variables in the TOML file are expanded from the environment using `${VAR}` or `${VAR:-default}` syntax. For example:

```toml
api_key = "${UPSTREAM_API_KEY:-change-me}"
```

This is useful for keeping secrets out of config files in Docker/CI environments.

### Configuration reference

| TOML key | Default | Equivalent env var | Description |
|---|---|---|---|
| `listen_addr` | `:11434` | — | host:port the proxy binds to |
| `ollama_version` | `0.6.4` | — | reported by `/api/version` |
| `model_context_length` | `65536` | — | reported via `/api/show` and `/v1/models` |
| `proxy_api_key` | *(empty)* | — | require clients to authenticate with this key |
| `debug` | `false` | — | enable request/response debug logging |
| `log_max_body_bytes` | `4096` | — | truncate debug-logged request bodies to this many bytes |
| `stats_store_path` | *(empty)* | — | persist stats JSON so they survive restarts |
| `upstream_startup_wait` | `30m` | — | retry budget while upstream is loading the model |
| `upstream_retry_interval` | `2s` | — | delay between startup retries |
| `http_request_timeout` | `30s` | — | cap for short upstream calls (embeddings, models, health) |
| `http_stream_timeout` | `5m` | — | cap for streaming chat / generate requests |
| `shutdown_timeout` | `30s` | — | drain budget for in-flight requests on SIGTERM/SIGINT |
| `max_request_bytes` | `33554432` | — | reject inbound JSON bodies larger than this (32 MiB) |
| `upstream[].url` | *(required)* | — | upstream OpenAI-compatible API endpoint URL |
| `upstream[].api_key` | *(empty)* | — | sent as `Authorization: Bearer …` |
| `upstream[].models[].upstream` | *(required)* | — | model name presented to the upstream API |
| `upstream[].models[].local` | *(required)* | — | model name presented to Ollama/OpenAI clients |
| `upstream[].models[].context_length` | *global* | — | overrides `model_context_length` for this model |

## Quickstart

```bash
# Edit proxy.toml with your upstream settings, then:
docker compose up -d
```

Or use one of the pre-configured examples:

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

## Use with Claude Code CLI

[Claude Code](https://github.com/anthropics/claude-code) is Anthropic's agentic coding tool. You can point it at this proxy to use any upstream backend instead of Anthropic's API:

```bash
ANTHROPIC_BASE_URL=http://localhost:11434 ANTHROPIC_MODEL=my-model ANTHROPIC_AUTH_TOKEN=ollama claude
```

The proxy translates the Anthropic Messages API to OpenAI-compatible requests, so Claude Code's agentic loop, tool use, and streaming all work through any upstream that supports OpenAI chat completions with tools.

**Note:** Claude Code requires tool support. Make sure your upstream backend supports OpenAI-format tool calls.

## Multi-modal support (images & audio)

Images and audio are supported through all three API paths:

| Path | Input format |
|---|---|
| `/api/chat` (Ollama) | `images` field on Ollama messages — base64-encoded, auto-detected MIME type |
| `/api/generate` (Ollama) | `images` field on the generate request |
| `/v1/chat/completions` (OpenAI) | `content` array with `image_url` or `input_audio` parts |
| `/v1/messages` (Anthropic) | `content` array with `image` blocks (base64 source) |

The proxy:
- Detects image MIME type from magic bytes (PNG, JPEG, GIF, WebP, AVIF, HEIC, HEIF)
- Wraps bare base64 data into proper `data:` URLs for OpenAI upstreams
- Passes `input_audio` parts through unchanged to upstream
- Preserves multi-modal content arrays through the response normalization pipeline

## Testing

The project has two test suites:

### Unit tests

Fast, lightweight tests that verify translation logic, edge cases, and handler behaviour. No external dependencies required.

```bash
go test -v -race -count=1 ./...
```

### Conformance tests (`tests/conformance/`)

Integration tests that drive the proxy with the **official Go SDKs** of all three supported providers (OpenAI, Anthropic, Ollama). They verify that client requests are correctly translated and forwarded to the upstream by capturing the actual HTTP payload the proxy sends.

Because the SDKs live in a separate Go module, the core proxy module stays dependency-free.

```bash
cd tests/conformance && go test -v -count=1 ./...
```

**Prerequisite:** the first run downloads ~16 MiB of SDK dependencies (`go mod download`). After that, tests run offline.

**Coverage:**

| SDK | Tests | What's verified |
|---|---|---|
| OpenAI | 6 | Sampling params, function tools, vision (image_url), streaming content, embeddings, model list |
| Anthropic | 5 | System prompt + params, tool schema translation (`input_schema` → OpenAI `function`), image blocks, tool_result round-trip, stream accumulation |
| Ollama | 5 | Options → OpenAI sampling params, images → multimodal, tools → OpenAI schema, streaming, model list |

## Missing features

* Other upstream APIs (files, etc)
* Other upstream services
