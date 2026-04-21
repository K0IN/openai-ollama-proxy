package main

import (
	"log"
	"net/http"

	"github.com/k0in/openai-ollama-proxy/internal/config"
	"github.com/k0in/openai-ollama-proxy/internal/handlers"
	applogging "github.com/k0in/openai-ollama-proxy/internal/logging"
)

func main() {
	cfg := config.Load()
	server := handlers.New(cfg, config.NewHTTPClient())

	log.Printf("openai-ollama-proxy listening on %s", cfg.ListenAddr)
	log.Printf("  vLLM backend: %s", cfg.VLLMBaseURL)
	log.Printf("  vLLM model:   %s", cfg.VLLMModel)
	log.Printf("  Ollama model: %s", cfg.ModelName)
	if cfg.Debug {
		log.Printf("  debug logging: enabled")
	}

	if err := http.ListenAndServe(cfg.ListenAddr, applogging.Middleware(cfg.Debug, server.Routes())); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
