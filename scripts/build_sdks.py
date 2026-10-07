#!/usr/bin/env python3
"""Generates production-ready SDKs for Node.js, Go, Rust, Java, and Ruby in classone-sdks."""

from pathlib import Path

TARGET_DIR = Path(r"C:\Users\Thiago Gonzaga\classone-sdks")


def write_file(rel_path: str, content: str):
    full_path = TARGET_DIR / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [+] {rel_path}")


def build_nodejs():
    print("\n--- Building Node.js SDK ---")
    write_file(
        "nodejs/package.json",
        """{
  "name": "@classone/sdk",
  "version": "0.1.0",
  "description": "Official Node.js / TypeScript client SDK for ClassOne System 1 decision models",
  "main": "index.js",
  "module": "index.mjs",
  "type": "module",
  "exports": {
    ".": "./index.js"
  },
  "scripts": {
    "test": "node --test test/*.test.js",
    "example": "node examples/basic.js"
  },
  "keywords": ["classone", "system-1", "decision-model", "classification", "routing", "ai"],
  "author": "ClassOne Contributors",
  "license": "Apache-2.0"
}
""",
    )

    write_file(
        "nodejs/index.js",
        """/**
 * ClassOne Node.js / JavaScript Client SDK
 * Zero-dependency client for ClassOne System 1 decision models.
 */

export class Noul {
  constructor(instructions) {
    if (!instructions) throw new Error("Noul requires instructions");
    this.type = "noul";
    this.instructions = instructions;
  }
  toJSON() {
    return { type: "noul", instructions: this.instructions };
  }
}

export class Choice {
  constructor(instructions, criteria) {
    if (!instructions) throw new Error("Choice requires instructions");
    if (!criteria || Object.keys(criteria).length < 2) {
      throw new Error("Choice requires at least 2 criteria options");
    }
    this.type = "choice";
    this.instructions = instructions;
    this.criteria = criteria;
  }
  toJSON() {
    return { type: "choice", instructions: this.instructions, criteria: this.criteria };
  }
}

export class Score {
  constructor(instructions, criteria) {
    if (!instructions) throw new Error("Score requires instructions");
    if (!Array.isArray(criteria) || criteria.length < 2) {
      throw new Error("Score requires at least 2 rubric criteria levels");
    }
    this.type = "score";
    this.instructions = instructions;
    this.criteria = criteria;
  }
  toJSON() {
    return { type: "score", instructions: this.instructions, criteria: this.criteria };
  }
}

export class ClassOneResponse {
  constructor(data) {
    this.model = data.model || "";
    this.answers = data.answers || {};
    this.usage = data.usage || { input_tokens: 0, output_tokens: 0 };

    // Quick-access helpers
    this.nouls = {};
    this.choices = {};
    this.scores = {};

    for (const [qid, ans] of Object.entries(this.answers)) {
      if (ans.type === "noul") this.nouls[qid] = ans;
      else if (ans.type === "choice") this.choices[qid] = ans;
      else if (ans.type === "score") this.scores[qid] = ans;
    }
  }

  get(qid) {
    return this.answers[qid];
  }
}

export class ClassOneClient {
  constructor(options = {}) {
    this.baseUrl = (options.baseUrl || process.env.CLASSONE_BASE_URL || "http://127.0.0.1:8000").replace(/\\/+$/, "");
    this.apiKey = options.apiKey || process.env.CLASSONE_API_KEY || "default-key";
    this.timeout = options.timeout || 30000;
  }

  async decide(state, questions, options = {}) {
    const model = options.model || "class-one-gemma-4-e2b-it";
    const endpoint = options.endpoint || "/v1/decide";

    const serializedQuestions = {};
    for (const [qid, q] of Object.entries(questions)) {
      if (typeof q.toJSON === "function") {
        serializedQuestions[qid] = q.toJSON();
      } else {
        serializedQuestions[qid] = q;
      }
    }

    const payload = {
      model,
      state,
      questions: serializedQuestions,
    };

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);

    try {
      const resp = await fetch(`${this.baseUrl}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!resp.ok) {
        const errText = await resp.text();
        throw new Error(`ClassOne API error (${resp.status}): ${errText}`);
      }

      const data = await resp.json();
      return new ClassOneResponse(data);
    } finally {
      clearTimeout(timer);
    }
  }
}

export default ClassOneClient;
""",
    )

    write_file(
        "nodejs/test/client.test.js",
        """import test from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { ClassOneClient, Noul, Choice, Score } from "../index.js";

test("ClassOne Node.js SDK Unit & Mock Server Test", async (t) => {
  // 1. Test builder primitives validation
  await t.test("Primitives validation", () => {
    const n = new Noul("Is urgent?");
    assert.equal(n.type, "noul");
    assert.equal(n.toJSON().instructions, "Is urgent?");

    const c = new Choice("Select routing", { billing: "Billing issues", tech: "App bugs" });
    assert.equal(c.type, "choice");
    assert.equal(Object.keys(c.criteria).length, 2);

    const s = new Score("Risk level", ["low", "medium", "critical"]);
    assert.equal(s.type, "score");
    assert.equal(s.criteria.length, 3);

    assert.throws(() => new Choice("Too few", { onlyOne: "one" }));
    assert.throws(() => new Score("Too few", ["one"]));
  });

  // 2. Mock HTTP server test
  await t.test("Client decide() against mock server", async () => {
    const server = http.createServer((req, res) => {
      let body = "";
      req.on("data", (chunk) => { body += chunk; });
      req.on("end", () => {
        const payload = JSON.parse(body);
        assert.equal(payload.model, "test-model");
        assert.equal(payload.state, "Customer reported billing error.");
        assert.ok(payload.questions.intent);
        assert.ok(payload.questions.is_urgent);

        const responseData = {
          model: "test-model",
          answers: {
            intent: {
              type: "choice",
              choice: "billing",
              probabilities: { billing: 0.96, tech: 0.04 },
              confidence: 0.92,
            },
            is_urgent: {
              type: "noul",
              noul: 0.12,
            },
          },
          usage: { input_tokens: 42, output_tokens: 1 },
        };

        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(responseData));
      });
    });

    await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
    const port = server.address().port;

    try {
      const client = new ClassOneClient({
        baseUrl: `http://127.0.0.1:${port}`,
        apiKey: "test-secret-key",
      });

      const res = await client.decide(
        "Customer reported billing error.",
        {
          intent: new Choice("Route intent:", { billing: "Charges", tech: "Bugs" }),
          is_urgent: new Noul("Is urgent issue?"),
        },
        { model: "test-model" }
      );

      assert.equal(res.model, "test-model");
      assert.equal(res.choices.intent.choice, "billing");
      assert.equal(res.choices.intent.probabilities.billing, 0.96);
      assert.equal(res.nouls.is_urgent.noul, 0.12);
      assert.equal(res.usage.input_tokens, 42);
      assert.equal(res.get("intent").choice, "billing");
    } finally {
      server.close();
    }
  });
});
""",
    )

    write_file(
        "nodejs/examples/basic.js",
        """import { ClassOneClient, Noul, Choice, Score } from "../index.js";

async function main() {
  console.log("=== ClassOne Node.js Client Example ===");
  const client = new ClassOneClient({
    baseUrl: process.env.CLASSONE_BASE_URL || "http://127.0.0.1:8000",
  });

  const state = "Customer subscription renewed twice on transaction #9842. Requesting refund.";
  const questions = {
    intent: new Choice("Select routing department:", {
      billing: "Payment, charges, invoice and refund requests",
      tech: "Technical issues, app errors, downtime",
      account: "Password reset, login, profile settings"
    }),
    is_urgent: new Noul("Does this query require immediate priority handling?"),
    risk_level: new Score("Rate customer churn risk:", ["low", "moderate", "high", "critical"])
  };

  console.log("Sending decision request...");
  try {
    const res = await client.decide(state, questions);
    console.log(`Model: ${res.model}`);
    console.log(`Intent Choice: ${res.choices.intent.choice} (confidence: ${(res.choices.intent.confidence * 100).toFixed(1)}%)`);
    console.log(`Is Urgent (Noul): ${res.nouls.is_urgent.noul.toFixed(4)}`);
    console.log(`Risk Level (Score): ${res.scores.risk_level.score.toFixed(2)} / 4.0`);
    console.log(`Tokens Processed: ${res.usage.input_tokens}`);
  } catch (err) {
    console.error("Request failed (make sure ClassOne server is running):", err.message);
  }
}

main();
""",
    )


def build_go():
    print("\n--- Building Go SDK ---")
    write_file(
        "go/go.mod",
        """module github.com/devops-thiago/classone-sdks/go

go 1.22
""",
    )

    write_file(
        "go/types.go",
        """package classone

import "fmt"

type NoulQuestion struct {
	Type         string `json:"type"`
	Instructions string `json:"instructions"`
}

func NewNoul(instructions string) NoulQuestion {
	return NoulQuestion{
		Type:         "noul",
		Instructions: instructions,
	}
}

type ChoiceQuestion struct {
	Type         string            `json:"type"`
	Instructions string            `json:"instructions"`
	Criteria     map[string]string `json:"criteria"`
}

func NewChoice(instructions string, criteria map[string]string) (ChoiceQuestion, error) {
	if len(criteria) < 2 {
		return ChoiceQuestion{}, fmt.Errorf("choice question must have at least 2 criteria options")
	}
	return ChoiceQuestion{
		Type:         "choice",
		Instructions: instructions,
		Criteria:     criteria,
	}, nil
}

type ScoreQuestion struct {
	Type         string   `json:"type"`
	Instructions string   `json:"instructions"`
	Criteria     []string `json:"criteria"`
}

func NewScore(instructions string, criteria []string) (ScoreQuestion, error) {
	if len(criteria) < 2 {
		return ScoreQuestion{}, fmt.Errorf("score question must have at least 2 criteria levels")
	}
	return ScoreQuestion{
		Type:         "score",
		Instructions: instructions,
		Criteria:     criteria,
	}, nil
}

type DecisionRequest struct {
	Model     string         `json:"model"`
	State     interface{}    `json:"state"`
	Questions map[string]any `json:"questions"`
}

type DecisionUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type NoulAnswer struct {
	Type string  `json:"type"`
	Noul float64 `json:"noul"`
}

type ChoiceAnswer struct {
	Type          string             `json:"type"`
	Choice        string             `json:"choice"`
	Probabilities map[string]float64 `json:"probabilities"`
	Confidence    float64            `json:"confidence"`
}

type ScoreAnswer struct {
	Type          string             `json:"type"`
	Score         float64            `json:"score"`
	Probabilities map[string]float64 `json:"probabilities"`
	Confidence    float64            `json:"confidence"`
}

type RawDecisionResponse struct {
	Model   string                    `json:"model"`
	Answers map[string]map[string]any `json:"answers"`
	Usage   DecisionUsage             `json:"usage"`
}

type DecisionResponse struct {
	Model   string
	Usage   DecisionUsage
	Nouls   map[string]NoulAnswer
	Choices map[string]ChoiceAnswer
	Scores  map[string]ScoreAnswer
	Raw     map[string]map[string]any
}
""",
    )

    write_file(
        "go/client.go",
        """package classone

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Client struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
}

type Option func(*Client)

func WithBaseURL(url string) Option {
	return func(c *Client) {
		c.BaseURL = strings.TrimRight(url, "/")
	}
}

func WithAPIKey(key string) Option {
	return func(c *Client) {
		c.APIKey = key
	}
}

func WithTimeout(timeout time.Duration) Option {
	return func(c *Client) {
		c.HTTPClient.Timeout = timeout
	}
}

func NewClient(opts ...Option) *Client {
	c := &Client{
		BaseURL: "http://127.0.0.1:8000",
		APIKey:  "default-key",
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

func (c *Client) Decide(ctx context.Context, req DecisionRequest) (*DecisionResponse, error) {
	if req.Model == "" {
		req.Model = "class-one-gemma-4-e2b-it"
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/v1/decide", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	}

	resp, err := c.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		errBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("api error (status %d): %s", resp.StatusCode, string(errBytes))
	}

	var raw RawDecisionResponse
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	parsed := &DecisionResponse{
		Model:   raw.Model,
		Usage:   raw.Usage,
		Nouls:   make(map[string]NoulAnswer),
		Choices: make(map[string]ChoiceAnswer),
		Scores:  make(map[string]ScoreAnswer),
		Raw:     raw.Answers,
	}

	for qid, ansMap := range raw.Answers {
		ansBytes, _ := json.Marshal(ansMap)
		tVal, _ := ansMap["type"].(string)

		switch tVal {
		case "noul":
			var n NoulAnswer
			if err := json.Unmarshal(ansBytes, &n); err == nil {
				parsed.Nouls[qid] = n
			}
		case "choice":
			var ch ChoiceAnswer
			if err := json.Unmarshal(ansBytes, &ch); err == nil {
				parsed.Choices[qid] = ch
			}
		case "score":
			var sc ScoreAnswer
			if err := json.Unmarshal(ansBytes, &sc); err == nil {
				parsed.Scores[qid] = sc
			}
		}
	}

	return parsed, nil
}
""",
    )

    write_file(
        "go/client_test.go",
        """package classone

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestClassOneClient(t *testing.T) {
	// 1. Primitive validation tests
	t.Run("Primitive validation", func(t *testing.T) {
		n := NewNoul("Is safe?")
		if n.Type != "noul" {
			t.Fatalf("expected noul, got %s", n.Type)
		}

		_, err := NewChoice("Too few", map[string]string{"a": "single"})
		if err == nil {
			t.Fatal("expected error on 1 choice, got nil")
		}

		_, err = NewScore("Too few", []string{"single"})
		if err == nil {
			t.Fatal("expected error on 1 score level, got nil")
		}
	})

	// 2. Mock HTTP server integration test
	t.Run("Decide against mock server", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/v1/decide" {
				t.Fatalf("unexpected path: %s", r.URL.Path)
			}
			if r.Header.Get("Authorization") != "Bearer secret-test-token" {
				t.Fatalf("unexpected auth header: %s", r.Header.Get("Authorization"))
			}

			resp := RawDecisionResponse{
				Model: "class-one-gemma-4-e2b-it",
				Usage: DecisionUsage{InputTokens: 50, OutputTokens: 1},
				Answers: map[string]map[string]any{
					"intent": {
						"type":   "choice",
						"choice": "billing",
						"probabilities": map[string]float64{
							"billing": 0.95,
							"tech":    0.05,
						},
						"confidence": 0.90,
					},
					"urgent": {
						"type": "noul",
						"noul": 0.08,
					},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)
		}))
		defer server.Close()

		client := NewClient(
			WithBaseURL(server.URL),
			WithAPIKey("secret-test-token"),
		)

		ch, err := NewChoice("Route ticket:", map[string]string{
			"billing": "Invoice questions",
			"tech":    "Software bugs",
		})
		if err != nil {
			t.Fatal(err)
		}

		req := DecisionRequest{
			State: "Charged twice for subscription renewal.",
			Questions: map[string]any{
				"intent": ch,
				"urgent": NewNoul("Is high priority?"),
			},
		}

		res, err := client.Decide(context.Background(), req)
		if err != nil {
			t.Fatalf("Decide failed: %v", err)
		}

		if res.Model != "class-one-gemma-4-e2b-it" {
			t.Errorf("expected model class-one-gemma-4-e2b-it, got %s", res.Model)
		}
		if res.Choices["intent"].Choice != "billing" {
			t.Errorf("expected choice billing, got %s", res.Choices["intent"].Choice)
		}
		if res.Nouls["urgent"].Noul != 0.08 {
			t.Errorf("expected noul 0.08, got %f", res.Nouls["urgent"].Noul)
		}
		if res.Usage.InputTokens != 50 {
			t.Errorf("expected 50 tokens, got %d", res.Usage.InputTokens)
		}
	})
}
""",
    )

    write_file(
        "go/examples/main.go",
        """package main

import (
	"context"
	"fmt"
	"log"

	"github.com/devops-thiago/classone-sdks/go"
)

func main() {
	fmt.Println("=== ClassOne Go Client Example ===")

	client := classone.NewClient(
		classone.WithBaseURL("http://127.0.0.1:8000"),
	)

	ch, err := classone.NewChoice("Select ticket routing:", map[string]string{
		"billing": "Invoices, double charges, refunds",
		"tech":    "App crashes, latency, system errors",
		"account": "Logins, password recovery",
	})
	if err != nil {
		log.Fatal(err)
	}

	sc, err := classone.NewScore("Rate urgency level:", []string{"low", "medium", "critical"})
	if err != nil {
		log.Fatal(err)
	}

	req := classone.DecisionRequest{
		State: "Our payment gateway is returning 500 internal errors since 9am.",
		Questions: map[string]any{
			"intent":   ch,
			"urgent":   classone.NewNoul("Does this require immediate P1 incident response?"),
			"severity": sc,
		},
	}

	res, err := client.Decide(context.Background(), req)
	if err != nil {
		log.Fatalf("Request failed: %v", err)
	}

	fmt.Printf("Model: %s\\n", res.Model)
	fmt.Printf("Intent: %s (confidence: %.1f%%)\\n", res.Choices["intent"].Choice, res.Choices["intent"].Confidence*100)
	fmt.Printf("Urgent (Noul): %.4f\\n", res.Nouls["urgent"].Noul)
	fmt.Printf("Severity (Score): %.2f / 3.0\\n", res.Scores["severity"].Score)
	fmt.Printf("Input Tokens: %d\\n", res.Usage.InputTokens)
}
""",
    )


def build_rust():
    print("\n--- Building Rust SDK ---")
    write_file(
        "rust/Cargo.toml",
        """[package]
name = "classone"
version = "0.1.0"
edition = "2021"
description = "Official Rust client SDK for ClassOne System 1 decision models"
license = "Apache-2.0"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
minreq = { version = "2.12", features = ["json-using-serde"] }
thiserror = "1.0"

[[example]]
name = "basic"
path = "examples/basic.rs"
""",
    )

    write_file(
        "rust/src/lib.rs",
        """pub mod client;
pub mod types;

pub use client::ClassOneClient;
pub use types::{Choice, ChoiceAnswer, DecisionRequest, DecisionResponse, DecisionUsage, Noul, NoulAnswer, Score, ScoreAnswer};
""",
    )

    write_file(
        "rust/src/types.rs",
        """use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Noul {
    #[serde(rename = "type")]
    pub question_type: String,
    pub instructions: String,
}

impl Noul {
    pub fn new(instructions: impl Into<String>) -> Self {
        Self {
            question_type: "noul".to_string(),
            instructions: instructions.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    #[serde(rename = "type")]
    pub question_type: String,
    pub instructions: String,
    pub criteria: HashMap<String, String>,
}

impl Choice {
    pub fn new(instructions: impl Into<String>, criteria: HashMap<String, String>) -> Result<Self, String> {
        if criteria.len() < 2 {
            return Err("Choice requires at least 2 criteria options".to_string());
        }
        Ok(Self {
            question_type: "choice".to_string(),
            instructions: instructions.into(),
            criteria,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score {
    #[serde(rename = "type")]
    pub question_type: String,
    pub instructions: String,
    pub criteria: Vec<String>,
}

impl Score {
    pub fn new(instructions: impl Into<String>, criteria: Vec<String>) -> Result<Self, String> {
        if criteria.len() < 2 {
            return Err("Score requires at least 2 rubric criteria levels".to_string());
        }
        Ok(Self {
            question_type: "score".to_string(),
            instructions: instructions.into(),
            criteria,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRequest {
    pub model: String,
    pub state: serde_json::Value,
    pub questions: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoulAnswer {
    #[serde(rename = "type")]
    pub answer_type: String,
    pub noul: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceAnswer {
    #[serde(rename = "type")]
    pub answer_type: String,
    pub choice: String,
    pub probabilities: HashMap<String, f64>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreAnswer {
    #[serde(rename = "type")]
    pub answer_type: String,
    pub score: f64,
    pub probabilities: HashMap<String, f64>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawDecisionResponse {
    pub model: String,
    pub answers: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub usage: Option<DecisionUsage>,
}

#[derive(Debug, Clone)]
pub struct DecisionResponse {
    pub model: String,
    pub usage: DecisionUsage,
    pub nouls: HashMap<String, NoulAnswer>,
    pub choices: HashMap<String, ChoiceAnswer>,
    pub scores: HashMap<String, ScoreAnswer>,
    pub raw: HashMap<String, serde_json::Value>,
}
""",
    )

    write_file(
        "rust/src/client.rs",
        """use crate::types::*;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ClassOneError {
    #[error("HTTP error: {0}")]
    Minreq(#[from] minreq::Error),
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("API error ({status}): {body}")]
    ApiError { status: i32, body: String },
}

pub struct ClassOneClient {
    base_url: String,
    api_key: String,
    timeout_secs: u64,
}

impl ClassOneClient {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            timeout_secs: 30,
        }
    }

    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    pub fn decide(
        &self,
        state: impl Into<serde_json::Value>,
        questions: HashMap<String, serde_json::Value>,
        model: Option<&str>,
    ) -> Result<DecisionResponse, ClassOneError> {
        let req_body = DecisionRequest {
            model: model.unwrap_or("class-one-gemma-4-e2b-it").to_string(),
            state: state.into(),
            questions,
        };

        let url = format!("{}/v1/decide", self.base_url);
        let resp = minreq::post(&url)
            .with_header("Content-Type", "application/json")
            .with_header("Authorization", format!("Bearer {}", self.api_key))
            .with_timeout(self.timeout_secs)
            .with_json(&req_body)?
            .send()?;

        if resp.status_code < 200 || resp.status_code >= 300 {
            let body = resp.as_str().unwrap_or_default().to_string();
            return Err(ClassOneError::ApiError {
                status: resp.status_code,
                body,
            });
        }

        let raw: RawDecisionResponse = resp.json()?;
        let mut nouls = HashMap::new();
        let mut choices = HashMap::new();
        let mut scores = HashMap::new();

        for (qid, val) in &raw.answers {
            if let Some(t_val) = val.get("type").and_then(|v| v.as_str()) {
                match t_val {
                    "noul" => {
                        if let Ok(n) = serde_json::from_value::<NoulAnswer>(val.clone()) {
                            nouls.insert(qid.clone(), n);
                        }
                    }
                    "choice" => {
                        if let Ok(c) = serde_json::from_value::<ChoiceAnswer>(val.clone()) {
                            choices.insert(qid.clone(), c);
                        }
                    }
                    "score" => {
                        if let Ok(s) = serde_json::from_value::<ScoreAnswer>(val.clone()) {
                            scores.insert(qid.clone(), s);
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(DecisionResponse {
            model: raw.model,
            usage: raw.usage.unwrap_or(DecisionUsage { input_tokens: 0, output_tokens: 0 }),
            nouls,
            choices,
            scores,
            raw: raw.answers,
        })
    }
}
""",
    )

    write_file(
        "rust/examples/basic.rs",
        """use classone::{Choice, ClassOneClient, Noul};
use std::collections::HashMap;

fn main() {
    println!("=== ClassOne Rust SDK Example ===");
    let client = ClassOneClient::new("http://127.0.0.1:8000", "test-key");

    let mut criteria = HashMap::new();
    criteria.insert("billing".to_string(), "Charges & refunds".to_string());
    criteria.insert("tech".to_string(), "App bugs & outages".to_string());

    let choice = Choice::new("Route ticket:", criteria).unwrap();
    let noul = Noul::new("Is urgent issue?");

    let mut questions = HashMap::new();
    questions.insert("intent".to_string(), serde_json::to_value(choice).unwrap());
    questions.insert("urgent".to_string(), serde_json::to_value(noul).unwrap());

    match client.decide("Checkout returned 500 error.", questions, None) {
        Ok(res) => {
            println!("Model: {}", res.model);
            if let Some(intent) = res.choices.get("intent") {
                println!("Intent: {} (confidence: {:.1}%)", intent.choice, intent.confidence * 100.0);
            }
            if let Some(urgent) = res.nouls.get("urgent") {
                println!("Urgent: {:.4}", urgent.noul);
            }
        }
        Err(e) => println!("Request failed (start ClassOne server first): {}", e),
    }
}
""",
    )


def build_java():
    print("\n--- Building Java SDK ---")
    write_file(
        "java/pom.xml",
        """<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>io.classone</groupId>
    <artifactId>classone-sdk</artifactId>
    <version>0.1.0</version>
    <properties>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>
</project>
""",
    )

    write_file(
        "java/src/main/java/io/classone/Noul.java",
        """package io.classone;

public record Noul(String instructions) {
    public static Noul of(String instructions) {
        if (instructions == null || instructions.isBlank()) {
            throw new IllegalArgumentException("instructions cannot be empty");
        }
        return new Noul(instructions);
    }
}
""",
    )

    write_file(
        "java/src/main/java/io/classone/Choice.java",
        """package io.classone;

import java.util.Map;

public record Choice(String instructions, Map<String, String> criteria) {
    public static Choice of(String instructions, Map<String, String> criteria) {
        if (instructions == null || instructions.isBlank()) {
            throw new IllegalArgumentException("instructions cannot be empty");
        }
        if (criteria == null || criteria.size() < 2) {
            throw new IllegalArgumentException("criteria must contain at least 2 options");
        }
        return new Choice(instructions, criteria);
    }
}
""",
    )

    write_file(
        "java/src/main/java/io/classone/Score.java",
        """package io.classone;

import java.util.List;

public record Score(String instructions, List<String> criteria) {
    public static Score of(String instructions, List<String> criteria) {
        if (instructions == null || instructions.isBlank()) {
            throw new IllegalArgumentException("instructions cannot be empty");
        }
        if (criteria == null || criteria.size() < 2) {
            throw new IllegalArgumentException("criteria must contain at least 2 rubric levels");
        }
        return new Score(instructions, criteria);
    }
}
""",
    )

    write_file(
        "java/src/main/java/io/classone/ClassOneResponse.java",
        """package io.classone;

import java.util.Map;

public record ClassOneResponse(
    String model,
    Map<String, Object> answers,
    Map<String, Integer> usage
) {
    public record NoulResult(double noul) {}
    public record ChoiceResult(String choice, Map<String, Double> probabilities, double confidence) {}
    public record ScoreResult(double score, Map<String, Double> probabilities, double confidence) {}
}
""",
    )

    write_file(
        "java/src/main/java/io/classone/ClassOneClient.java",
        """package io.classone;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ClassOneClient {
    private final String baseUrl;
    private final String apiKey;
    private final HttpClient httpClient;

    private ClassOneClient(Builder builder) {
        this.baseUrl = builder.baseUrl.replaceAll("/+$", "");
        this.apiKey = builder.apiKey;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(builder.timeoutSeconds))
                .build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String baseUrl = System.getenv().getOrDefault("CLASSONE_BASE_URL", "http://127.0.0.1:8000");
        private String apiKey = System.getenv().getOrDefault("CLASSONE_API_KEY", "default-key");
        private int timeoutSeconds = 30;

        public Builder baseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
            return this;
        }

        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        public Builder timeoutSeconds(int timeoutSeconds) {
            this.timeoutSeconds = timeoutSeconds;
            return this;
        }

        public ClassOneClient build() {
            return new ClassOneClient(this);
        }
    }

    public String decideRaw(String state, Map<String, ?> questions, String model) throws IOException, InterruptedException {
        String jsonPayload = buildJson(state, questions, model != null ? model : "class-one-gemma-4-e2b-it");

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/v1/decide"))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + apiKey)
                .POST(HttpRequest.BodyPublishers.ofString(jsonPayload))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            throw new IOException("ClassOne API error (" + response.statusCode() + "): " + response.body());
        }
        return response.body();
    }

    private String buildJson(String state, Map<String, ?> questions, String model) {
        StringBuilder sb = new StringBuilder("{");
        sb.append("\\"model\\":\\"").append(escape(model)).append("\\",");
        sb.append("\\"state\\":\\"").append(escape(state)).append("\\",");
        sb.append("\\"questions\\":{");
        int i = 0;
        for (var entry : questions.entrySet()) {
            if (i++ > 0) sb.append(",");
            sb.append("\\"").append(escape(entry.getKey())).append("\\":");
            Object q = entry.getValue();
            if (q instanceof Noul n) {
                sb.append("{\\"type\\":\\"noul\\",\\"instructions\\":\\"").append(escape(n.instructions())).append("\\"}");
            } else if (q instanceof Choice c) {
                sb.append("{\\"type\\":\\"choice\\",\\"instructions\\":\\"").append(escape(c.instructions())).append("\\",\\"criteria\\":{");
                int j = 0;
                for (var crit : c.criteria().entrySet()) {
                    if (j++ > 0) sb.append(",");
                    sb.append("\\"").append(escape(crit.getKey())).append("\\":\\"").append(escape(crit.getValue())).append("\\"");
                }
                sb.append("}}");
            } else if (q instanceof Score s) {
                sb.append("{\\"type\\":\\"score\\",\\"instructions\\":\\"").append(escape(s.instructions())).append("\\",\\"criteria\\":[");
                for (int k = 0; k < s.criteria().size(); k++) {
                    if (k > 0) sb.append(",");
                    sb.append("\\"").append(escape(s.criteria().get(k))).append("\\"");
                }
                sb.append("]}");
            }
        }
        sb.append("}}");
        return sb.toString();
    }

    private String escape(String s) {
        return s.replace("\\\\", "\\\\\\\\").replace("\\"", "\\\\\\"").replace("\\n", "\\\\n").replace("\\r", "\\\\r");
    }
}
""",
    )

    write_file(
        "java/examples/BasicExample.java",
        """import io.classone.*;
import java.util.List;
import java.util.Map;

public class BasicExample {
    public static void main(String[] args) {
        System.out.println("=== ClassOne Java SDK Example ===");
        ClassOneClient client = ClassOneClient.builder()
                .baseUrl("http://127.0.0.1:8000")
                .build();

        var questions = Map.of(
            "intent", Choice.of("Route ticket:", Map.of("billing", "Billing queries", "tech", "App bugs")),
            "urgent", Noul.of("Is urgent issue?"),
            "risk", Score.of("Risk severity:", List.of("low", "medium", "critical"))
        );

        try {
            String jsonResp = client.decideRaw("Double charge on invoice #1234.", questions, null);
            System.out.println("Decision response: " + jsonResp);
        } catch (Exception e) {
            System.out.println("Request failed (start ClassOne server first): " + e.getMessage());
        }
    }
}
""",
    )


def build_ruby():
    print("\n--- Building Ruby SDK ---")
    write_file(
        "ruby/classone.gemspec",
        """Gem::Specification.new do |s|
  s.name        = 'classone'
  s.version     = '0.1.0'
  s.summary     = 'Official Ruby client SDK for ClassOne System 1 decision models'
  s.description = 'Fast single-pass typed decisions with calibrated probabilities for Ruby'
  s.authors     = ['ClassOne Contributors']
  s.license     = 'Apache-2.0'
  s.files       = Dir['lib/**/*.rb', 'examples/**/*.rb', 'README.md']
  s.require_paths = ['lib']
  s.required_ruby_version = '>= 3.0.0'
end
""",
    )

    write_file(
        "ruby/lib/classone.rb",
        """require 'json'
require 'net/http'
require 'uri'

module ClassOne
  class Error < StandardError; end

  class Noul
    attr_reader :instructions
    def initialize(instructions)
      raise ArgumentError, "instructions required" if instructions.nil? || instructions.empty?
      @instructions = instructions
    end

    def to_h
      { type: 'noul', instructions: @instructions }
    end
  end

  class Choice
    attr_reader :instructions, :criteria
    def initialize(instructions, criteria)
      raise ArgumentError, "instructions required" if instructions.nil? || instructions.empty?
      raise ArgumentError, "at least 2 criteria options required" if criteria.nil? || criteria.size < 2
      @instructions = instructions
      @criteria = criteria
    end

    def to_h
      { type: 'choice', instructions: @instructions, criteria: @criteria }
    end
  end

  class Score
    attr_reader :instructions, :criteria
    def initialize(instructions, criteria)
      raise ArgumentError, "instructions required" if instructions.nil? || instructions.empty?
      raise ArgumentError, "at least 2 rubric levels required" if criteria.nil? || criteria.size < 2
      @instructions = instructions
      @criteria = criteria
    end

    def to_h
      { type: 'score', instructions: @instructions, criteria: @criteria }
    end
  end

  class Response
    attr_reader :model, :answers, :usage, :nouls, :choices, :scores

    def initialize(data)
      @model = data['model'] || ''
      @answers = data['answers'] || {}
      @usage = data['usage'] || {}

      @nouls = {}
      @choices = {}
      @scores = {}

      @answers.each do |qid, val|
        case val['type']
        when 'noul' then @nouls[qid] = val
        when 'choice' then @choices[qid] = val
        when 'score' then @scores[qid] = val
        end
      end
    end

    def [](qid)
      @answers[qid]
    end
  end

  class Client
    def initialize(base_url: nil, api_key: nil, timeout: 30)
      @base_url = (base_url || ENV['CLASSONE_BASE_URL'] || 'http://127.0.0.1:8000').chomp('/')
      @api_key = api_key || ENV['CLASSONE_API_KEY'] || 'default-key'
      @timeout = timeout
    end

    def decide(state, questions, model: 'class-one-gemma-4-e2b-it')
      serialized_q = {}
      questions.each do |qid, q|
        serialized_q[qid] = q.respond_to?(:to_h) ? q.to_h : q
      end

      payload = {
        model: model,
        state: state,
        questions: serialized_q
      }

      uri = URI("#{@base_url}/v1/decide")
      http = Net::HTTP.new(uri.host, uri.port)
      http.use_ssl = (uri.scheme == 'https')
      http.read_timeout = @timeout

      req = Net::HTTP::Post.new(uri.path, {
        'Content-Type' => 'application/json',
        'Authorization' => "Bearer #{@api_key}"
      })
      req.body = JSON.generate(payload)

      res = http.request(req)
      unless res.is_a?(Net::HTTPSuccess)
        raise Error, "ClassOne API error (#{res.code}): #{res.body}"
      end

      data = JSON.parse(res.body)
      Response.new(data)
    end
  end
end
""",
    )

    write_file(
        "ruby/test/client_test.rb",
        """require_relative '../lib/classone'
require 'minitest/autorun'
require 'socket'

class ClassOneClientTest < Minitest::Test
  def test_primitives_validation
    n = ClassOne::Noul.new("Is urgent?")
    assert_equal 'noul', n.to_h[:type]

    c = ClassOne::Choice.new("Route ticket", { "billing" => "Invoices", "tech" => "Bugs" })
    assert_equal 'choice', c.to_h[:type]
    assert_equal 2, c.to_h[:criteria].size

    s = ClassOne::Score.new("Risk tier", ["low", "medium", "critical"])
    assert_equal 'score', s.to_h[:type]
    assert_equal 3, s.to_h[:criteria].size

    assert_raises(ArgumentError) { ClassOne::Choice.new("Few", { "one" => "only" }) }
    assert_raises(ArgumentError) { ClassOne::Score.new("Few", ["only"]) }
  end

  def test_mock_server_decide
    server = TCPServer.new('127.0.0.1', 0)
    port = server.addr[1]

    t = Thread.new do
      session = server.accept
      # Read HTTP request headers
      while (line = session.gets) && line !~ /^\\s*$/
      end

      body = JSON.generate({
        model: 'test-model',
        answers: {
          intent: {
            type: 'choice',
            choice: 'billing',
            probabilities: { 'billing' => 0.98, 'tech' => 0.02 },
            confidence: 0.96
          },
          urgent: {
            type: 'noul',
            noul: 0.15
          }
        },
        usage: { input_tokens: 38, output_tokens: 1 }
      })

      session.print "HTTP/1.1 200 OK\\r\\nContent-Type: application/json\\r\\nContent-Length: #{body.bytesize}\\r\\n\\r\\n#{body}"
      session.close
    end

    client = ClassOne::Client.new(base_url: "http://127.0.0.1:#{port}")
    res = client.decide(
      'Duplicate invoice charge.',
      {
        intent: ClassOne::Choice.new('Route:', { 'billing' => 'Invoices', 'tech' => 'Bugs' }),
        urgent: ClassOne::Noul.new('Is urgent?')
      },
      model: 'test-model'
    )

    assert_equal 'test-model', res.model
    assert_equal 'billing', res.choices['intent']['choice']
    assert_equal 0.15, res.nouls['urgent']['noul']
    assert_equal 38, res.usage['input_tokens']
  ensure
    server.close if server && !server.closed?
    t.join if t
  end
end
""",
    )

    write_file(
        "ruby/examples/basic.rb",
        """require_relative '../lib/classone'

puts "=== ClassOne Ruby SDK Example ==="
client = ClassOne::Client.new(base_url: ENV['CLASSONE_BASE_URL'] || 'http://127.0.0.1:8000')

state = "Transaction failed with error 503 at checkout."
questions = {
  intent: ClassOne::Choice.new("Select department:", {
    "billing" => "Invoice, payments, charges",
    "tech" => "Application downtime and 500 errors"
  }),
  urgent: ClassOne::Noul.new("Does this require immediate escalation?")
}

begin
  res = client.decide(state, questions)
  puts "Model: #{res.model}"
  puts "Intent: #{res.choices['intent']['choice']} (confidence: #{(res.choices['intent']['confidence'] * 100).round(1)}%)"
  puts "Urgent: #{res.nouls['urgent']['noul'].round(4)}"
  puts "Tokens: #{res.usage['input_tokens']}"
rescue ClassOne::Error => e
  puts "Request failed (start ClassOne server first): #{e.message}"
end
""",
    )


def build_repo_metadata():
    print("\n--- Writing Repo Documentation & CI ---")
    write_file(
        "README.md",
        r"""# ClassOne Multi-Language SDKs

Official, production-ready client libraries for **ClassOne System 1 Decision Models**.

ClassOne evaluates fast, typed, and calibrated probabilistic decisions in a single forward pass without token-by-token generation overhead.

---

## Supported Languages

| Language | Directory | Package / Artifact | Requirements |
|---|---|---|---|
| **Node.js / TypeScript** | `nodejs/` | `@classone/sdk` | Node.js 18+ (Zero external dependencies) |
| **Go** | `go/` | `github.com/devops-thiago/classone-sdks/go` | Go 1.21+ (Zero external dependencies) |
| **Rust** | `rust/` | `classone` | Rust 1.75+ (Cargo) |
| **Java** | `java/` | `io.classone:classone-sdk` | Java 17+ (Zero external runtime dependencies) |
| **Ruby** | `ruby/` | `classone` | Ruby 3.0+ (Zero external gem dependencies) |

---

## Core Primitives

All SDKs implement the three core ClassOne decision primitives:

1. **`Noul`**: Boolean evaluation returning a calibrated probability $\in [0.0, 1.0]$.
2. **`Choice`**: Categorical classification against dynamic criteria with normalized probabilities and scale-invariant confidence.
3. **`Score`**: Ordinal evaluation mapping input states against an ordered rubric scale.

---

## Quickstart Examples

### 1. Node.js / TypeScript
```javascript
import { ClassOneClient, Choice, Noul } from "@classone/sdk";

const client = new ClassOneClient({ baseUrl: "http://127.0.0.1:8000" });
const res = await client.decide("Checkout returned 500 errors since 9am.", {
  intent: new Choice("Route ticket:", {
    billing: "Payment and charges",
    tech: "Software outages"
  }),
  urgent: new Noul("Is urgent priority?")
});

console.log(res.choices.intent.choice); // "tech"
console.log(res.nouls.urgent.noul);     // 0.9412
```

### 2. Go
```go
package main

import (
    "context"
    "fmt"
    "github.com/devops-thiago/classone-sdks/go"
)

func main() {
    client := classone.NewClient()
    choice, _ := classone.NewChoice("Route ticket:", map[string]string{
        "billing": "Payment and charges",
        "tech":    "Software outages",
    })
    res, _ := client.Decide(context.Background(), classone.DecisionRequest{
        State: "Checkout returned 500 errors since 9am.",
        Questions: map[string]any{
            "intent": choice,
            "urgent": classone.NewNoul("Is urgent priority?"),
        },
    })
    fmt.Println(res.Choices["intent"].Choice) // "tech"
}
```

### 3. Rust
```rust
use classone::{ClassOneClient, Choice, Noul};
use std::collections::HashMap;

fn main() {
    let client = ClassOneClient::new("http://127.0.0.1:8000", "api-key");
    let mut criteria = HashMap::new();
    criteria.insert("billing".to_string(), "Payment & charges".to_string());
    criteria.insert("tech".to_string(), "Software outages".to_string());

    let mut questions = HashMap::new();
    questions.insert("intent".to_string(), serde_json::to_value(Choice::new("Route:", criteria).unwrap()).unwrap());

    let res = client.decide("Checkout returned 500 errors.", questions, None).unwrap();
    println!("{}", res.choices["intent"].choice);
}
```

### 4. Java
```java
import io.classone.*;
import java.util.Map;

public class Main {
    public static void main(String[] args) throws Exception {
        ClassOneClient client = ClassOneClient.builder().build();
        var questions = Map.of(
            "intent", Choice.of("Route:", Map.of("billing", "Payment", "tech", "Outage")),
            "urgent", Noul.of("Is urgent?")
        );
        String json = client.decideRaw("Checkout returned 500 errors.", questions, null);
        System.out.println(json);
    }
}
```

### 5. Ruby
```ruby
require 'classone'

client = ClassOne::Client.new
res = client.decide("Checkout returned 500 errors since 9am.", {
  intent: ClassOne::Choice.new("Route:", { "billing" => "Payment", "tech" => "Outage" }),
  urgent: ClassOne::Noul.new("Is urgent?")
})

puts res.choices["intent"]["choice"] # "tech"
```

---

## Running Tests Locally

```bash
# Node.js
cd nodejs && npm test

# Go
cd go && go test -v ./...

# Rust
cd rust && cargo test

# Java
cd java && javac src/main/java/io/classone/*.java

# Ruby
cd ruby && ruby test/client_test.rb
```

---

## License
Apache-2.0
""",
    )

    write_file(
        ".gitignore",
        """# Node
node_modules/
dist/
package-lock.json

# Go
bin/
*.exe

# Rust
target/
Cargo.lock

# Java
target/
*.class

# Ruby
*.gem
.bundle/

# OS / Editor
.DS_Store
Thumbs.db
.vscode/
.idea/
""",
    )

    write_file(
        ".github/workflows/ci.yml",
        """name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test-nodejs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: cd nodejs && npm test

  test-go:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-go@v5
        with:
          go-version: '1.22'
      - run: cd go && go test -v ./...

  test-rust:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cd rust && cargo test

  test-java:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with:
          distribution: 'temurin'
          java-version: '17'
      - run: cd java && javac src/main/java/io/classone/*.java

  test-ruby:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ruby/setup-ruby@v1
        with:
          ruby-version: '3.2'
      - run: cd ruby && ruby test/client_test.rb
""",
    )


def main():
    print(f"Generating ClassOne multi-language SDKs in: {TARGET_DIR}")
    build_nodejs()
    build_go()
    build_rust()
    build_java()
    build_ruby()
    build_repo_metadata()
    print("\n[✓] All 5 SDKs generated successfully!")


if __name__ == "__main__":
    main()
