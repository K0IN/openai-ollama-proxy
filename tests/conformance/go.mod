module github.com/k0in/openai-ollama-proxy/tests/conformance

go 1.26.0

// The conformance suite imports the proxy's internal packages directly so it
// can spin the real handlers up in-process. Because the packages live under
// internal/, the module path must be rooted at the proxy's module path, and we
// point that dependency at the repo root via a replace directive.
require (
	github.com/anthropics/anthropic-sdk-go v1.46.0
	github.com/openai/openai-go/v3 v3.37.0
)

require (
	github.com/k0in/openai-ollama-proxy v0.0.0
	github.com/ollama/ollama v0.30.8
)

replace github.com/k0in/openai-ollama-proxy => ../../

require (
	github.com/BurntSushi/toml v1.6.0 // indirect
	github.com/bahlo/generic-list-go v0.2.0 // indirect
	github.com/buger/jsonparser v1.1.2 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/invopop/jsonschema v0.13.0 // indirect
	github.com/mailru/easyjson v0.7.7 // indirect
	github.com/standard-webhooks/standard-webhooks/libraries v0.0.1 // indirect
	github.com/tidwall/gjson v1.18.0 // indirect
	github.com/tidwall/match v1.1.1 // indirect
	github.com/tidwall/pretty v1.2.1 // indirect
	github.com/tidwall/sjson v1.2.5 // indirect
	github.com/wk8/go-ordered-map/v2 v2.1.8 // indirect
	golang.org/x/crypto v0.45.0 // indirect
	golang.org/x/sync v0.17.0 // indirect
	golang.org/x/sys v0.38.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
