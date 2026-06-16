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
	data, err := os.ReadFile(path) // #nosec G304 G703 -- path comes from user config, not arbitrary input
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
		log.Fatal("CONFIG_FILE is required. Set CONFIG_FILE=/path/to/proxy.toml or mount a config file. See proxy.toml for an example.")
	}

	cfg, router, err := LoadFile(path)
	if err != nil {
		log.Fatalf("failed to load config file %q: %v", path, err) // #nosec G706 -- path is from user's own CONFIG_FILE env var
	}

	return cfg, router
}
