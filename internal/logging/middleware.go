package logging

import (
	"bytes"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"
)

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (recorder *statusRecorder) WriteHeader(code int) {
	recorder.status = code
	recorder.ResponseWriter.WriteHeader(code)
}

func (recorder *statusRecorder) Flush() {
	if flusher, ok := recorder.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func Middleware(debug bool, maxBodyBytes int, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Never log /stats requests.
		if r.URL.Path == "/stats" {
			next.ServeHTTP(w, r)
			return
		}

		start := time.Now()
		recorder := &statusRecorder{ResponseWriter: w, status: http.StatusOK}

		userAgent := r.Header.Get("User-Agent")
		if userAgent == "" {
			userAgent = "-"
		}

		if debug {
			var headers strings.Builder
			for key, values := range r.Header {
				for _, value := range values {
					fmt.Fprintf(&headers, "  %s: %s\n", key, RedactHeaderValue(key, value))
				}
			}
			log.Printf(">>> %s %s | ua=%q\n%s", SanitizeForLog(r.Method), SanitizeForLog(r.URL.String()), SanitizeForLog(userAgent), headers.String()) // #nosec G706 -- inputs sanitized via SanitizeForLog

			if r.Body != nil && r.Method == http.MethodPost {
				body, _ := io.ReadAll(r.Body)
				r.Body = io.NopCloser(bytes.NewReader(body))
				if len(body) > 0 {
					redacted := RedactJSONForLog(body)
					var truncated []byte
					if maxBodyBytes > 0 && len(redacted) > maxBodyBytes {
						truncated = redacted[:maxBodyBytes]
						truncated = append(truncated, []byte("\n... (truncated ", strconv.Itoa(len(redacted)-maxBodyBytes), " bytes)")...)
					} else {
						truncated = redacted
					}
					var indented bytes.Buffer
					if err := json.Indent(&indented, truncated, "  ", "  "); err == nil {
						log.Printf(">>> REQUEST BODY (%d bytes):\n  %s", len(body), indented.String())
					} else {
						log.Printf(">>> REQUEST BODY (%d bytes): %s", len(body), string(truncated))
					}
				}
			}
		}

		next.ServeHTTP(recorder, r)

		duration := time.Since(start).Round(time.Millisecond)
		log.Printf("<<< %s %s %d %s | ua=%q", SanitizeForLog(r.Method), SanitizeForLog(r.URL.Path), recorder.status, duration, SanitizeForLog(userAgent)) // #nosec G706 -- inputs sanitized via SanitizeForLog
	})
}

// ExtractAPIKey attempts to read an API key from the request, checking the
// Authorization: Bearer header first, then falling back to X-API-Key header.
func ExtractAPIKey(r *http.Request) string {
	if auth := r.Header.Get("Authorization"); strings.HasPrefix(auth, "Bearer ") {
		return auth[len("Bearer "):]
	}
	return r.Header.Get("X-API-Key")
}

// AuthMiddleware validates the proxy API key using a constant-time comparison.
//
// If the configured proxy_api_key is empty (not set or empty string), ALL
// requests are accepted regardless of what key (if any) is supplied.
//
// When proxy_api_key is set, exactly one must match:
//   - Authorization: Bearer <key>
//   - X-API-Key: <key>
//
// Responses are returned in a format compatible with OpenAI, Anthropic, and
// Ollama client expectations.
func AuthMiddleware(proxyAPIKey string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// If no API key is configured, accept all requests.
		if proxyAPIKey == "" {
			next.ServeHTTP(w, r)
			return
		}

		supplied := ExtractAPIKey(r)
		if supplied == "" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":{"message":"Missing API key — provide via Authorization: Bearer or X-API-Key header","type":"unauthorized_error"}}`))
			return
		}

		// Constant-time comparison to prevent timing side-channel attacks.
		if subtle.ConstantTimeCompare([]byte(supplied), []byte(proxyAPIKey)) != 1 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":{"message":"Invalid API key","type":"unauthorized_error"}}`))
			return
		}

		next.ServeHTTP(w, r)
	})
}
