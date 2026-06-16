package config

import (
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"

	"github.com/BurntSushi/toml"
)

func expandEnvVars(s string) string {
	re := regexp.MustCompile(`\$\{([^}]+)\}`)
	return re.ReplaceAllStringFunc(s, func(match string) string {
		inner := match[2 : len(match)-1]

		if before, after, ok := strings.Cut(inner, ":-"); ok {
			varName := before
			defaultVal := after
			if val := os.Getenv(varName); val != "" {
				return val
			}
			return defaultVal
		}

		return os.Getenv(inner)
	})
}

func LoadFile(path string) (Config, *RoutingTable, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, nil, fmt.Errorf("reading config file %q: %w", path, err)
	}

	expanded := []byte(expandEnvVars(string(data)))

	var cfg Config
	if err := toml.Unmarshal(expanded, &cfg); err != nil {
		return Config{}, nil, fmt.Errorf("parsing config file %q: %w", path, err)
	}

	cfg.applyDefaults()

	if err := cfg.Validate(); err != nil {
		return Config{}, nil, fmt.Errorf("invalid configuration: %w", err)
	}

	router, err := BuildRoutingTable(cfg.Upstreams, cfg.ModelContextLength)
	if err != nil {
		return Config{}, nil, fmt.Errorf("building routing table: %w", err)
	}

	return cfg, router, nil
}

func Load() (Config, *RoutingTable) {
	path := os.Getenv("CONFIG_FILE")
	if path == "" {
		// Fall back to environment-variable-only mode so that users who set
		// UPSTREAM_BASE_URL / UPSTREAM_MODEL (e.g. docker-compose examples)
		// do not need a TOML file.
		cfg := LoadFromEnv()
		router, err := BuildRoutingTable(cfg.Upstreams, cfg.ModelContextLength)
		if err != nil {
			// No [[upstream]] entries in env-only mode — that is fine; the
			// code falls back to the flat config fields at runtime.
			return cfg, nil
		}
		return cfg, router
	}

	cfg, router, err := LoadFile(path)
	if err != nil {
		log.Fatalf("failed to load config file %q: %v", path, err)
	}

	return cfg, router
}
