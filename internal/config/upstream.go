package config

import (
	"fmt"
	"net/url"
	"sort"
	"strings"
)

type ModelMapping struct {
	Upstream          string   `toml:"upstream"`
	Local             string   `toml:"local"`
	ContextLength     int      `toml:"context_length,omitempty"`
	SupportsVision    bool     `toml:"supports_vision,omitempty"`
	SupportsThinking  []string `toml:"supports_thinking,omitempty"`
	EnableThinkingAPI bool     `toml:"-"`
	ThinkingLevel     string   `toml:"thinking_level,omitempty"`
}

type UpstreamConfig struct {
	URL          string         `toml:"url"`
	APIKey       string         `toml:"api_key"`
	Passthrough  bool           `toml:"passthrough"`
	RetryOnError bool           `toml:"retry_on_429"` // config key kept for backward compat; also retries 5xx, 403, 400, and 401
	Models       []ModelMapping `toml:"models"`
}

type UpstreamEntry struct {
	URL              string
	APIKey           string
	Passthrough      bool
	RetryOnError     bool
	UpstreamModel    string
	ContextLength    int
	SupportsVision   bool
	SupportsThinking bool
	ThinkingLevel    string
}

type RoutingTable struct {
	entries      map[string]UpstreamEntry
	allModels    []string
	allUpstreams []UpstreamConfig
}

func (rt *RoutingTable) Lookup(localModel string) (UpstreamEntry, bool) {
	if rt == nil {
		return UpstreamEntry{}, false
	}
	entry, ok := rt.entries[localModel]
	return entry, ok
}

func (rt *RoutingTable) AllModels() []string {
	if rt == nil {
		return nil
	}
	return rt.allModels
}

func (rt *RoutingTable) AllUpstreams() []UpstreamConfig {
	if rt == nil {
		return nil
	}
	return rt.allUpstreams
}

func (m ModelMapping) Validate() error {
	if strings.TrimSpace(m.Upstream) == "" {
		return fmt.Errorf("model mapping has empty upstream name")
	}
	if strings.TrimSpace(m.Local) == "" {
		return fmt.Errorf("model mapping has empty local name")
	}
	if m.ContextLength < 0 {
		return fmt.Errorf("model mapping context_length must be >= 0 (got %d)", m.ContextLength)
	}
	return nil
}

func (m ModelMapping) expandedModels() []ModelMapping {
	base := m
	base.EnableThinkingAPI = len(m.SupportsThinking) > 0 || m.ThinkingLevel != ""
	if len(m.SupportsThinking) == 0 {
		return []ModelMapping{base}
	}

	expanded := make([]ModelMapping, 0, len(m.SupportsThinking))
	for _, level := range m.SupportsThinking {
		clone := base
		clone.Local = fmt.Sprintf("%s-%s", m.Local, level)
		clone.ThinkingLevel = level
		clone.SupportsThinking = nil
		expanded = append(expanded, clone)
	}
	return expanded
}

func (u UpstreamConfig) Validate() error {
	if strings.TrimSpace(u.URL) == "" {
		return fmt.Errorf("upstream URL must not be empty")
	}
	parsed, err := url.Parse(u.URL)
	if err != nil {
		return fmt.Errorf("upstream URL %q is not a valid URL: %w", u.URL, err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return fmt.Errorf("upstream URL %q must use http or https scheme", u.URL)
	}
	if parsed.Host == "" {
		return fmt.Errorf("upstream URL %q is missing a host", u.URL)
	}
	if len(u.Models) == 0 {
		return fmt.Errorf("upstream %q must define at least one model", u.URL)
	}

	// api_key and passthrough are mutually exclusive.
	if u.Passthrough && u.APIKey != "" {
		return fmt.Errorf("upstream %q: api_key and passthrough cannot both be set", u.URL)
	}

	for i, m := range u.Models {
		if err := m.Validate(); err != nil {
			return fmt.Errorf("upstream %q model[%d]: %w", u.URL, i, err)
		}
	}
	return nil
}

func BuildRoutingTable(upstreams []UpstreamConfig, globalCtxLen int) (*RoutingTable, error) {
	if len(upstreams) == 0 {
		return nil, fmt.Errorf("at least one [[upstream]] must be configured")
	}
	entries := make(map[string]UpstreamEntry)
	var allModels []string

	for i, u := range upstreams {
		if err := u.Validate(); err != nil {
			return nil, fmt.Errorf("upstream[%d] (%s): %w", i, u.URL, err)
		}

		for _, m := range u.Models {
			for _, expanded := range m.expandedModels() {
				local := expanded.Local
				if _, exists := entries[local]; exists {
					return nil, fmt.Errorf("duplicate local model %q (upstream[%d] %s)", local, i, u.URL)
				}

				ctxLen := globalCtxLen
				if expanded.ContextLength > 0 {
					ctxLen = expanded.ContextLength
				}

				entries[local] = UpstreamEntry{
					URL:              u.URL,
					APIKey:           u.APIKey,
					Passthrough:      u.Passthrough,
					RetryOnError:     u.RetryOnError,
					UpstreamModel:    expanded.Upstream,
					ContextLength:    ctxLen,
					SupportsVision:   expanded.SupportsVision,
					SupportsThinking: expanded.EnableThinkingAPI,
					ThinkingLevel:    expanded.ThinkingLevel,
				}
				allModels = append(allModels, local)
			}
		}
	}

	sort.Strings(allModels)

	return &RoutingTable{
		entries:      entries,
		allModels:    allModels,
		allUpstreams: upstreams,
	}, nil
}
