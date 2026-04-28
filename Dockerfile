FROM --platform=${TARGETPLATFORM} golang:1.24-alpine AS builder
WORKDIR /app
COPY go.mod ./
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /proxy ./cmd/proxy

FROM --platform=${TARGETPLATFORM} scratch
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=builder --chown=65532:65532 /proxy /proxy
USER 65532:65532
EXPOSE 11434
ENTRYPOINT ["/proxy"]
