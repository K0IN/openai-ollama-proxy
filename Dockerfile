FROM --platform=${TARGETPLATFORM} golang:1.25.11-alpine AS builder
WORKDIR /app
COPY go.mod ./
COPY go.sum ./
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /proxy ./cmd/proxy \
    && mkdir -p /var/lib/openai-ollama-proxy

FROM --platform=${TARGETPLATFORM} scratch
LABEL maintainer="k0in" \
	repo="https://github.com/k0in/openai-ollama-proxy" \
	email="thisk0in@gmail.com"
COPY proxy.toml /proxy.toml
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=builder --chown=65532:65532 /proxy /proxy
COPY --from=builder --chown=65532:65532 /var/lib/openai-ollama-proxy /var/lib/openai-ollama-proxy
USER 65532:65532
EXPOSE 11434
ENTRYPOINT ["/proxy"]
