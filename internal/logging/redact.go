package logging

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"
)

var redactedSensitiveHeaders = map[string]struct{}{
	http.CanonicalHeaderKey("Authorization"):       {},
	http.CanonicalHeaderKey("Proxy-Authorization"): {},
	http.CanonicalHeaderKey("Cookie"):              {},
	http.CanonicalHeaderKey("Set-Cookie"):          {},
	http.CanonicalHeaderKey("X-Api-Key"):           {},
	http.CanonicalHeaderKey("Api-Key"):             {},
	http.CanonicalHeaderKey("X-Auth-Token"):        {},
}

var redactedJSONFieldSubstrings = []string{
	"api_key",
	"api-key",
	"apikey",
	"authorization",
	"password",
	"secret",
	"token",
}

func RedactHeaderValue(name, value string) string {
	if _, sensitive := redactedSensitiveHeaders[http.CanonicalHeaderKey(name)]; sensitive {
		if value == "" {
			return ""
		}
		return "[REDACTED]"
	}
	return SanitizeForLog(value)
}

// SanitizeForLog replaces ASCII control characters (including CR/LF) in s
// with a single space so attacker-controlled input cannot forge or split log
// lines (CWE-117). Tab is preserved.
func SanitizeForLog(s string) string {
	if s == "" {
		return s
	}
	needs := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if (c < 0x20 && c != '\t') || c == 0x7f {
			needs = true
			break
		}
	}
	if !needs {
		return s
	}
	b := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if (c < 0x20 && c != '\t') || c == 0x7f {
			b[i] = ' '
			continue
		}
		b[i] = c
	}
	return string(b)
}

func RedactJSONForLog(payload []byte) []byte {
	if len(payload) == 0 {
		return payload
	}
	var value any
	dec := json.NewDecoder(bytes.NewReader(payload))
	dec.UseNumber()
	if err := dec.Decode(&value); err != nil {
		return payload
	}
	redacted := redactJSONValue(value)
	out, err := json.Marshal(redacted)
	if err != nil {
		return payload
	}
	return out
}

func redactJSONValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		for key, child := range typed {
			if isRedactedJSONField(key) {
				typed[key] = "[REDACTED]"
				continue
			}
			typed[key] = redactJSONValue(child)
		}
		return typed
	case []any:
		for i, child := range typed {
			typed[i] = redactJSONValue(child)
		}
		return typed
	default:
		return typed
	}
}

func isRedactedJSONField(name string) bool {
	lower := strings.ToLower(name)
	for _, needle := range redactedJSONFieldSubstrings {
		if strings.Contains(lower, needle) {
			return true
		}
	}
	return false
}
