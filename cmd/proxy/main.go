package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/k0in/openai-ollama-proxy/internal/config"
	"github.com/k0in/openai-ollama-proxy/internal/handlers"
	applogging "github.com/k0in/openai-ollama-proxy/internal/logging"
	"github.com/k0in/openai-ollama-proxy/internal/stats"
)

func main() {
	cfg, router := config.Load()

	// Load persisted stats if a store path is configured.
	st, err := stats.LoadFromFile(cfg.StatsStorePath)
	if err != nil {
		log.Printf("failed to load stats from %q: %v (starting fresh)", cfg.StatsStorePath, err)
		st = stats.New()
	}

	server := handlers.NewWithClients(cfg, router, config.NewHTTPClient(cfg), config.NewRequestHTTPClient(cfg), st)

	handler := applogging.MaxBytes(cfg.MaxRequestBytes, applogging.Middleware(cfg.Debug, applogging.AuthMiddleware(cfg.ProxyAPIKey, server.Routes())))

	httpServer := &http.Server{
		Addr:              cfg.ListenAddr,
		Handler:           handler,
		ReadHeaderTimeout: 30 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	statsStorePath := cfg.StatsStorePath
	if statsStorePath != "" {
		// Periodic save every 30 seconds so stats survive a crash.
		go func() {
			ticker := time.NewTicker(30 * time.Second)
			defer ticker.Stop()
			for {
				select {
				case <-ticker.C:
					if err := st.Save(statsStorePath); err != nil {
						log.Printf("error saving stats: %v", err)
					}
				case <-ctx.Done():
					return
				}
			}
		}()
	}

	log.Printf("openai-ollama-proxy listening on %s", cfg.ListenAddr)
	for _, u := range router.AllUpstreams() {
		apiKeyStatus := "not set"
		if u.APIKey != "" {
			apiKeyStatus = "set"
		}
		if u.Passthrough {
			apiKeyStatus = "passthrough"
		}
		parsedURL, _ := url.Parse(u.URL)
		displayURL := u.URL
		if parsedURL != nil {
			displayURL = parsedURL.Host
		}
		log.Printf("  provider: %s  (API key: %s)", displayURL, apiKeyStatus)
		for _, m := range u.Models {
			log.Printf("    %s -> %s", m.Local, m.Upstream)
		}
	}
	log.Printf("  max request bytes:   %d", cfg.MaxRequestBytes)
	log.Printf("  request timeout:     %s", cfg.HTTPRequestTimeout)
	log.Printf("  stream timeout:      %s", cfg.HTTPStreamTimeout)
	log.Printf("  shutdown timeout:    %s", cfg.ShutdownTimeout)
	if cfg.Debug {
		log.Printf("  debug logging:       enabled")
	}
	if cfg.ProxyAPIKey != "" {
		log.Printf("  proxy API key:       configured")
	}
	if statsStorePath != "" {
		log.Printf("  stats store:         %s", statsStorePath)
	} else {
		log.Printf("  stats store:         not stored")
	}

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
		// Save stats before shutting down
		if statsStorePath != "" {
			if err := st.Save(statsStorePath); err != nil {
				log.Printf("error saving stats on shutdown: %v", err)
			}
		}
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
