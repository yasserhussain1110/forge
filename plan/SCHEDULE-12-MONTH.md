# 12-Month Plan: Round-Robin AI Platform + Audience + Sales

**Goal (from `~/workspace/ideas/12_month_plan_ai_platform_audience_sales.md`):** after 12 months, be the person people point to for AI platform engineering, with an audience that trusts you and the ability to turn that trust into money (contracts and cohorts).

**Method:** one platform, six parts, built in **passes**. Each pass goes round all six parts, **one part at a time**. Every time you come back to a part, you take it one level deeper.

**Start:** Monday 16 Nov 2026 (W1) · **End:** Sunday 14 Nov 2027 (W52) · **Time:** 5 hours a day, Monday to Saturday

---

## 1. The one system you are building

All six topics are parts of the same platform, so work on one part makes the others better.

```
                    ┌──────────────────────────────┐
  user ───────────► │  ROUTING  (gateway)          │ ──► SERVING: Forge (your engine),
  or AGENT          │  routing, caching, fallbacks │      a local model, or a hosted API
                    └──────────────┬───────────────┘
                                   │
                         RAG (find the right documents,
                              add them to the prompt)

  EVALS          score the output of every part, so a change is measured, not guessed
  OBSERVABILITY  trace every request: time per step, tokens, cost
```

| Part | What it means |
|---|---|
| **Serving** | Running the model and generating tokens. This is Forge, the engine you are writing in this repo. |
| **RAG** | Retrieval-augmented generation: find relevant documents and give them to the model before it answers. |
| **Evals** | Automated tests that score model output, so you know whether a change helped. |
| **Observability** | Recording what happened in each request (steps, time, tokens, cost) so you can see where time and money go. |
| **Routing** | A gateway in front of the models that decides which model gets each request, and handles caching, retries and fallbacks. |
| **Agents** | A loop where the model calls tools (search, calculator, code) step by step to finish a task. |

**Repos:**
- `forge`, this repo: serving only.
- One second repo for everything else, with one folder per part: `rag/`, `evals/`, `observability/`, `gateway/` and `agents/`. Tag a release each time a part ships.

**Architecture rules** (these keep the parts connectable and the signature tool easy to split out):

```
  agents/ ──► gateway/ ──HTTP (OpenAI-compatible API, + traceparent header)──► Forge
                 │                                                       └──► hosted API, other local models
                 ▼
               rag/

  evals/          calls the gateway or Forge over HTTP, never imports their code
  observability/  receives traces and metrics from all of them
```

1. **HTTP is the only connection to Forge.** The second repo never does `import forge`. It talks to Forge only through `/v1/chat/completions` and `/v1/models`, so Forge and vLLM can be swapped for each other.
2. **Forge reads the `traceparent` header.** `traceparent` is the W3C standard HTTP header that carries the ID of the trace a request belongs to; OpenTelemetry uses it to join spans from different services into one trace. Forge week 8 (day 51) accepts it and logs its trace ID; W43 adds Forge's own spans to that trace.
3. **One package per folder in the second repo.** Each of `rag/`, `evals/`, `observability/`, `gateway/`, `agents/` has its own `pyproject.toml` (its own dependencies, version and release tag), set up as a **uv workspace**: one repo holding several Python packages that share one lockfile. Shared helpers (tracing setup, a model client) go in a small `common/` package. Parts use each other only through public interfaces, mostly HTTP, never through internal modules.
4. **Correctness tests live in Forge; quality evals live in `evals/`.** "Does my kernel / cache / CUDA graph give the same numbers as Hugging Face?" is a Forge unit test run in Forge's CI. "Are the answers still good?" is an eval in `evals/` that runs over HTTP against any backend.

**Hardware:** your RTX 4060 (8 GB) runs Forge, the embedding models and small local models. When RAG answers or agents need a stronger model, use a hosted API or a 4-bit 8B model run locally. Record any API spend in `RESULTS.md`.

---

## 2. The year at a glance

```
PASS 1 · breadth · 2 weeks per part · W1–W13 (16 Nov – 14 Feb)
 Serving │ RAG │ Evals │ ░ │ Observ. │ Routing │ Agents
  W1–2     W3–4   W5–6   W7   W8–9      W10–11    W12–13            ░ = holiday buffer

PASS 2 · depth · serving 6 weeks, others 2–3 · W14–W32 (15 Feb – 27 Jun)
 Serving ██████ │ RAG │ Evals │ Observ. │ Routing │ Agents
  W14–19          W20–22 W23–25  W26–27    W28–30    W31–32

PASS 3 · deeper · serving 6 weeks, others 2 · W33–W48 (28 Jun – 17 Oct)
 Serving ██████ │ RAG │ Evals │ Observ. │ Routing │ Agents
  W33–38          W39–40 W41–42  W43–44    W45–46    W47–48

WRAP-UP · W49–W52 (18 Oct – 14 Nov): launch the signature tool, release Forge, publish the reference architecture, review the year
```

Where Forge (serving) gets to:

| Pass | Forge weeks | What Forge can do afterwards |
|---|---|---|
| 1 | 5–6 | Runs Llama with your own code; KV cache; sampling; streaming |
| 2 | 7–12 | Profiled; HTTP server; benchmarked against vLLM; CUDA graphs; your own Triton kernels; FlashAttention |
| 3 | 13–18 | Decode attention kernel; continuous batching; PagedAttention; prefix caching; chunked prefill; load-tested |
| Year 2 | 19–24 | Speculative decoding, quantization, two GPUs, full benchmark launch |

---

## 3. Daily and weekly rhythm

**Monday to Saturday, 5 hours:**

| Block | Time | What goes in it |
|---|---|---|
| **Build** | 3 h | The current week's work for the current part (tables below). In serving weeks, open the next Forge day file and do what it says. |
| **Audience** | 1.25 h | 45 min writing one post about today's build · 20 min of useful comments on bigger accounts in the niche · 10 min replying to comments and DMs |
| **Sales** | 0.75 h | W1–W8: reading. From W9: 5 personalised outreach messages, follow-ups and calls. |

**Sunday:** the Forge "Sunday review" day file (in serving weeks), the weekly scorecard (section 9) and planning next week's posts.

### Rules
1. **One part at a time.** Don't touch other parts except to fix something that is actually broken.
2. **Slot lengths are fixed.** Don't switch early because a part got hard or boring. Getting stuck is often your best post.
3. **Every slot ends with a number and a write-up.** The part's README gets the measured result, and you publish a long post about it.
4. **Paid work comes first.** When a paid build lands, it replaces the current build slot. Skip that slot's remaining weeks, not your writing.
5. **Forge's own rules still apply:** keep `notes/` and `RESULTS.md`, and every number you measure goes into `RESULTS.md`.

---

## 4. Pass 1: breadth · W1–W13 · 16 Nov 2026 – 14 Feb 2027

Goal: a thin but working version of every part, each with one measured number. Pass 1 ends with all six parts connected end to end.

### Serving · W1–W2
| Week | Dates | Work |
|---|---|---|
| W1 | 16–22 Nov | Forge week 5 (days 30–35): RMSNorm and MLP, RoPE and attention with GQA, the full model with real weights, first generated text, first measurement |
| W2 | 23–29 Nov | Forge week 6 (days 36–42): inference arithmetic, KV cache, proving the cache is correct, sampling, stop conditions and streaming |

**Ships:** Forge generates text with a KV cache, and you have the measured speed-up from the cache. **Long write-up:** Forge blog post 1 (day 41).

### RAG · W3–W4
| Week | Dates | Work |
|---|---|---|
| W3 | 30 Nov – 6 Dec | First, create the second repo as a uv workspace (architecture rule 3): a `common/` package and a `rag/` package with its own `pyproject.toml`; add the other folders as packages when their slot starts. Then pick one public document set. Public company annual reports suit your finance and data background. Parse, chunk, embed with a local embedding model, search, and answer with a model. |
| W4 | 7–13 Dec | Write 30–50 test questions, each with the passage that answers it. Measure **recall@5**: for what fraction of questions does the right passage appear in the top 5 results? Compare 2 chunk sizes. |

**Ships:** `rag/` v0 with recall@5 in its README.

### Evals · W5–W6
| Week | Dates | Work |
|---|---|---|
| W5 | 14–20 Dec | Harness v0: a dataset file (question plus expected answer), a runner that calls any OpenAI-compatible endpoint, and simple scorers (exact match, contains, rubric). Run it on the RAG answers. |
| W6 | 21–27 Dec | **LLM-as-judge**, meaning a model scores answers. Hand-label 50 answers yourself, then measure how often the judge agrees with you. |

**Ships:** `evals/` v0 with the judge's agreement rate.

### Buffer · W7 · 28 Dec – 3 Jan
Holidays. Catch up on anything that slipped, or rest.

### Observability · W8–W9
| Week | Dates | Work |
|---|---|---|
| W8 | 4–10 Jan | Log every request: input tokens, output tokens, **TTFT** (time to first token), total latency and cost. Cover RAG, the hosted API and Forge. |
| W9 | 11–17 Jan | **Tracing** with OpenTelemetry, which records each step of a request as a timed span. Trace one RAG request through retrieve → generate, and view it locally in Jaeger. |

**Ships:** `observability/` v0, plus a chart of where a RAG request's time goes.

### Routing · W10–W11
| Week | Dates | Work |
|---|---|---|
| W10 | 18–24 Jan | An OpenAI-compatible proxy that forwards to a hosted API or a local model, with retries and timeouts. Backends are just URLs (HTTP only, architecture rule 1), and the proxy forwards the `traceparent` header to them. |
| W11 | 25–31 Jan | Rule-based routing: send easy requests to a small model and hard ones to a large model. Measure the cost saved and the quality lost with the evals harness. |

**Ships:** `gateway/` v0 with a cost vs quality number.

### Agents · W12–W13
| Week | Dates | Work |
|---|---|---|
| W12 | 1–7 Feb | A tool-calling loop with two tools, RAG search and a calculator. Each step is traced with your observability code. |
| W13 | 8–14 Feb | 20 test tasks; measure the task success rate. **Connect all six parts end to end:** agent → gateway → RAG → model, scored by evals and traced by observability. |

**Ships:** `agents/` v0. **Long write-up:** "I built every part of an LLM platform in 12 weeks: what each one taught me."

### Pass 1 checkpoint (14 Feb 2027)
- [ ] All six parts work, each with one number in its README
- [ ] About 60 short posts and 6 long write-ups published
- [ ] 10–20 discovery calls held (outreach starts in W9)
- Note which topics got the most saves, shares and DMs. That's your first signal for the signature tool.

---

## 5. Pass 2: depth · W14–W32 · 15 Feb – 27 Jun 2027

Goal: take each part from "works" to "measured and production-shaped". Serving gets 6 weeks because Forge has to go in order and is the biggest part.

### Serving · W14–W19
| Week | Dates | Forge week | Topic |
|---|---|---|---|
| W14 | 15–21 Feb | 7 | Understanding performance: rooflines, PyTorch profiler, Nsight, serving metrics |
| W15 | 22–28 Feb | 8 | FastAPI server, chat completions endpoint, streaming, benchmark harness, compare with vLLM. **Design doc 1** (day 55) |
| W16 | 1–7 Mar | 9 | gpt-fast, torch.compile, CUDA graphs, CI. **Blog post 2** (day 62) |
| W17 | 8–14 Mar | 10 | How GPUs work: CUDA basics, memory hierarchy, your first CUDA kernels |
| W18 | 15–21 Mar | 11 | Triton: vector add, softmax, matmul, your fused RMSNorm, SwiGLU and RoPE kernels |
| W19 | 22–28 Mar | 12 | FlashAttention: online softmax, the papers, your prefill attention kernel |

**Ships:** Forge has a server and is benchmarked against vLLM. From now on the gateway can use Forge as a backend.

### RAG · W20–W22
| Week | Dates | Work |
|---|---|---|
| W20 | 29 Mar – 4 Apr | Ingestion as a real data pipeline: incremental updates, deduplication, freshness, and lineage (which document version produced which chunk). This is your data-engineering edge. |
| W21 | 5–11 Apr | **Hybrid search** (keyword search plus vector search) and a **reranker** (a model that re-orders the top results). Measure how recall@k changes. |
| W22 | 12–18 Apr | **Groundedness:** is every claim in the answer supported by the retrieved text? Score it with the evals harness. Write up a failure analysis. |

### Evals · W23–W25
| Week | Dates | Work |
|---|---|---|
| W23 | 19–25 Apr | Regression testing in CI: every prompt or model change is compared with a saved baseline, and the build fails if scores drop. |
| W24 | 26 Apr – 2 May | RAG evals: groundedness and answer relevance, with checks on the judge. Version the eval datasets. |
| W25 | 3–9 May | Evals for Forge: do the KV cache, CUDA graphs and your kernels change the output? Numerical-equivalence checks go into Forge's own test suite and CI; answer-quality evals go into `evals/` and run over HTTP (architecture rule 4). |

### Observability · W26–W27
| Week | Dates | Work |
|---|---|---|
| W26 | 10–16 May | Prometheus metrics and a Grafana dashboard for the Forge server: tokens per second, TTFT and per-token latency percentiles, GPU utilisation. Compare cost per million tokens for Forge and a hosted API. |
| W27 | 17–23 May | Failure analysis: sort bad answers into categories using traces, and add alerting on error rate and latency. |

### Routing · W28–W30
| Week | Dates | Work |
|---|---|---|
| W28 | 24–30 May | **Exact cache** (same request returns the stored answer) and **semantic cache** (a request with the same meaning returns the stored answer). Measure the hit rate and how often a cached answer is wrong. |
| W29 | 31 May – 6 Jun | Fallbacks, rate limits and a **circuit breaker** (stop sending requests to a backend that keeps failing). Test it by killing backends on purpose. |
| W30 | 7–13 Jun | Add Forge as a backend. Route by cost using the numbers from observability. |

### Agents · W31–W32
| Week | Dates | Work |
|---|---|---|
| W31 | 14–20 Jun | Reliability: retries, timeouts, saved state, runs that can resume after a crash, and sandboxed code execution in Docker. |
| W32 | 21–27 Jun | Guardrails: redact personal data, and build a **prompt-injection** test set (inputs that try to override the instructions). Measure how often each defence blocks an attack. |

### Pass 2 checkpoint (27 Jun 2027)
- [ ] Forge benchmarked against vLLM; every part has production-shaped features and numbers
- [ ] At least 1 paid build closed and 50 discovery calls held (the target was W26)
- [ ] Newsletter live and inbound DMs started
- **Pick the signature open-source tool:** the non-serving part that got the most real use, questions or DMs. In pass 3 its slot aims at a public release.
- If there are 0 paid builds after about 50 calls, go through the call log: is the problem urgent, are you reaching the person with the budget, is the offer clear? Fix the offer before sending more messages.

---

## 6. Pass 3: deeper · W33–W48 · 28 Jun – 17 Oct 2027

Goal: serving gets the hardest Forge work, and every other part gets the step that makes it worth paying for.

### Serving · W33–W38
| Week | Dates | Forge week | Topic |
|---|---|---|---|
| W33 | 28 Jun – 4 Jul | 13 | Decode attention kernel, benchmarks, hardening. **Blog post 3** (day 90) |
| W34 | 5–11 Jul | 14 | Continuous batching: the Orca paper, scheduler loop. **Design doc 2** (day 95) |
| W35 | 12–18 Jul | 15 | PagedAttention: block allocator, block tables, paged decode kernel |
| W36 | 19–25 Jul | 16 | Finishing the paged cache: preemption, packed prefill. **Blog post 4** (day 111) |
| W37 | 26 Jul – 1 Aug | 17 | Prefix caching and chunked prefill |
| W38 | 2–8 Aug | 18 | Load tests, cancellation, stress tests, vLLM comparison. **Design doc 3** (day 125) |

### RAG · W39–W40
| Week | Dates | Work |
|---|---|---|
| W39 | 9–15 Aug | Scale: grow the document set 10×. Measure index build time, query latency (p50 and p99) and cost per 1,000 queries. |
| W40 | 16–22 Aug | Query rewriting and a pipeline that refreshes embeddings when documents change. |

### Evals · W41–W42
| Week | Dates | Work |
|---|---|---|
| W41 | 23–29 Aug | **LoRA fine-tuning** (training a small add-on to a model instead of all its weights) of Llama 3.2 1B on one narrow task. It fits in 8 GB. |
| W42 | 30 Aug – 5 Sep | **Eval-gated release:** the fine-tuned model ships only if it beats the base model on your test set. |

### Observability · W43–W44
| Week | Dates | Work |
|---|---|---|
| W43 | 6–12 Sep | One trace across gateway → RAG → Forge's batching scheduler: where does a request's time go, end to end? The gateway sends `traceparent` to Forge; inside Forge, add OpenTelemetry spans for queue wait, prefill and decode under that trace (architecture rule 2). |
| W44 | 13–19 Sep | Cost attribution: cost per request, per feature and per customer. |

### Routing · W45–W46
| Week | Dates | Work |
|---|---|---|
| W45 | 20–26 Sep | **Multi-tenancy** (one gateway shared safely by many customers): API keys, per-customer quotas, isolation. |
| W46 | 27 Sep – 3 Oct | Load-test the gateway with Forge and hosted backends together. |

### Agents · W47–W48
| Week | Dates | Work |
|---|---|---|
| W47 | 4–10 Oct | Multi-step agents that work across the whole platform through the gateway. Cost and failure analysis per task. |
| W48 | 11–17 Oct | Human approval before risky tool calls. Measure the success rate on the 20 tasks from pass 1 against this pass's version. |

### Pass 3 checkpoint (17 Oct 2027)
- [ ] Forge has continuous batching, PagedAttention and prefix caching, load-tested and compared with vLLM
- [ ] Signature tool released, with users
- [ ] Cohort 1 delivered

---

## 7. Wrap-up · W49–W52 · 18 Oct – 14 Nov 2027

| Week | Dates | Work |
|---|---|---|
| W49 | 18–24 Oct | Signature tool: docs, integrations, case studies |
| W50 | 25–31 Oct | Forge public release at its current state: README, benchmark charts against vLLM, demo video |
| W51 | 1–7 Nov | **Reference architecture for an LLM platform:** one deep piece covering all six parts, with your real numbers. This is the piece people bookmark. |
| W52 | 8–14 Nov | Year review against the goals in section 10; plan year 2 |

---

## 8. Audience and sales tracks

### Audience (1.25 h a day)
| When | What |
|---|---|
| W1 onwards | **5 short posts a week** (Monday to Friday), each one from that day's build, ideally with a number. LinkedIn is the main channel, X the second. 20 min a day of comments. |
| Every slot | One long write-up at the end of each slot, plus Forge's blog posts and design docs. That's about one every 2–3 weeks. |
| W14 onwards | Weekly newsletter. |
| W33 onwards | Short videos. Forge's kernel and batching work makes good visual material. Collaborations and podcast guest spots. |
| W50–W51 | The two biggest moments of the year: the Forge release and the reference architecture. |

### Sales (0.75 h a day)
| When | What |
|---|---|
| W1–W3 | Read *The Mom Test*. |
| W4–W8 | Read *SPIN Selling*. |
| W9 onwards | 5 personalised outreach messages each weekday (25 a week) to founders, CTOs and heads of engineering at startups building AI products. Open with your most relevant post. Aim for 2–3 calls a week. Keep a call log: problem, objections, next step. |
| By W26 | 50 discovery calls and the first paid build. Read *Never Split the Difference*. |
| W30–W33 | **Pre-sell cohort 1, "Senior AI Platform Engineer".** The curriculum is the six parts you built in passes 1 and 2. |
| W36–W43 | Run cohort 1, two evenings a week. These hours come out of the build block, the same way paid work does. |
| W33 onwards | Read *Million Dollar Consulting*. Raise prices with each new case study. |
| W45–W48 | Sell cohort 2. Aim to close a $10k+ engagement. |

---

## 9. Weekly scorecard (every Sunday)

| Metric | Weekly target |
|---|---|
| Hours of deep build work | 18 |
| Short posts | 5 |
| Long write-ups | 1 every 2–3 weeks (end of each slot) |
| Outreach messages (from W9) | 25 |
| Discovery calls (from W9) | 2–3 |
| Numbers added to `RESULTS.md` | At least 1 |
| Followers and newsletter subscribers | Track the trend, not the weekly number |

---

## 10. What month 12 should look like

- Every part of an LLM platform built by you, with numbers, and connected as one system.
- Forge with continuous batching, PagedAttention and prefix caching, benchmarked honestly against vLLM.
- One open-source tool that people actually use.
- Writing that people in the niche read, save and share.
- Money closed from both contracts and a cohort, and you know your close rate.
- You can design an LLM platform on a whiteboard, with costs, failure modes and trade-offs.

## 11. Year 2 (not scheduled here)
Forge weeks 19–24 (speculative decoding, quantization, two GPUs, full benchmark launch), a fourth pass, larger engagements and cohort 3.

---

## 12. Week 1 checklist (16–22 Nov 2026)
- [ ] Create `notes/` and `RESULTS.md` in this repo (Forge's day-one rule; neither exists yet)
- [ ] Tick days 9–29 in `PROGRESS.md` (only days 1–8 are ticked)
- [ ] Rewrite your LinkedIn headline around AI platform engineering
- [ ] Forge days 30–35
- [ ] 5 short posts about this week's Forge work
- [ ] Start *The Mom Test*
