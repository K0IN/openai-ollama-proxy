package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/k0in/openai-ollama-proxy/internal/config"
	"github.com/k0in/openai-ollama-proxy/internal/handlers"
	applogging "github.com/k0in/openai-ollama-proxy/internal/logging"
)

func main() {
	cfg := config.Load()
	server := handlers.NewWithClients(cfg, config.NewHTTPClient(cfg), config.NewRequestHTTPClient(cfg))

	log.Printf("openai-ollama-proxy listening on %s", cfg.ListenAddr)
	log.Printf("  vLLM backend:        %s", cfg.VLLMBaseURL)
	log.Printf("  vLLM model:          %s", cfg.VLLMModel)
	log.Printf("  Ollama model:        %s", cfg.ModelName)
	log.Printf("  max request bytes:   %d", cfg.MaxRequestBytes)
	log.Printf("  request timeout:     %s", cfg.HTTPRequestTimeout)
	log.Printf("  stream timeout:      %s", cfg.HTTPStreamTimeout)
	log.Printf("  shutdown timeout:    %s", cfg.ShutdownTimeout)
	if cfg.Debug {
		log.Printf("  debug logging:       enabled")
	}

	handler := applogging.MaxBytes(cfg.MaxRequestBytes, applogging.Middleware(cfg.Debug, server.Routes()))

	httpServer := &http.Server{
		Addr:              cfg.ListenAddr,
		Handler:           handler,
		ReadHeaderTimeout: 30 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	serverErr := make(chan error, 1)
	go func() {
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serverErr <- err
			return
		}
		serverErr <- nil
	}()

	select {
	case err := <-serverErr:
		if err != nil {
			stop()
			log.Printf("server error: %v", err)
			os.Exit(1) //nolint:gocritic // stop() above releases signal handler before exit
		}
	case <-ctx.Done():
		log.Printf("shutdown signal received, draining for up to %s", cfg.ShutdownTimeout)
		shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
		defer cancel()
		if err := httpServer.Shutdown(shutdownCtx); err != nil {
			log.Printf("graceful shutdown failed: %v", err)
			if closeErr := httpServer.Close(); closeErr != nil {
				log.Printf("forced close error: %v", closeErr)
			}
		}
		log.Printf("shutdown complete")
	}
}
