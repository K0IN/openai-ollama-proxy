package logging

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
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

func Middleware(debug bool, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
					headers.WriteString(fmt.Sprintf("  %s: %s\n", key, RedactHeaderValue(key, value)))
				}
			}
			log.Printf(">>> %s %s | ua=%q\n%s", r.Method, r.URL.String(), userAgent, headers.String())

			if r.Body != nil && r.Method == http.MethodPost {
				body, _ := io.ReadAll(r.Body)
				r.Body = io.NopCloser(bytes.NewReader(body))
				if len(body) > 0 {
					redacted := RedactJSONForLog(body)
					var indented bytes.Buffer
					if err := json.Indent(&indented, redacted, "  ", "  "); err == nil {
						log.Printf(">>> REQUEST BODY (%d bytes):\n  %s", len(body), indented.String())
					} else {
						log.Printf(">>> REQUEST BODY (%d bytes): %s", len(body), string(redacted))
					}
				}
			}
		}

		next.ServeHTTP(recorder, r)

		duration := time.Since(start).Round(time.Millisecond)
		log.Printf("<<< %s %s %d %s | ua=%q", r.Method, r.URL.Path, recorder.status, duration, userAgent)
	})
}
