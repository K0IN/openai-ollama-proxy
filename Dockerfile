FROM golang:1.24-alpine AS builder
WORKDIR /app
COPY go.mod ./
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /proxy ./cmd/proxy

FROM scratch
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=builder /proxy /proxy
EXPOSE 11434
ENTRYPOINT ["/proxy"]
