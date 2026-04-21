FROM golang:1.24-alpine AS builder
WORKDIR /app
COPY go.mod ./
COPY *.go ./
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /proxy .

FROM alpine:3.21
RUN apk add --no-cache ca-certificates
COPY --from=builder /proxy /proxy
EXPOSE 11434
ENTRYPOINT ["/proxy"]
