# LangSmith “offline” (local-only) tracing: what’s possible

This note summarizes how to run **LangChain**, **LangGraph agents**, and **deepagents** such that **no traces/calls are stored in LangSmith Cloud**, plus what it takes to keep traces **inside your own infrastructure**.

## 0) What “offline” can mean (choose one)

### A) **No traces leave your machine**
- You **disable LangSmith tracing** entirely.
- You can still run your app locally; you just won’t get LangSmith trace views.

Docs:
- `LANGSMITH_TRACING=false` disables tracing: https://docs.langchain.com/langsmith/env-var
- Studio docs also explicitly state: “With tracing disabled, no data leaves your local server.” https://docs.langchain.com/oss/python/langchain/studio

### B) **Traces stored locally, with a LangSmith UI**
- This means **self-hosted LangSmith** (UI + API) in your own environment and pointing your apps at it.
- Note: docs state self-hosted LangSmith is an **Enterprise add-on**.

Docs:
- Self-hosted overview: https://docs.langchain.com/langsmith/self-hosted
- Local Docker (dev/test): https://docs.langchain.com/langsmith/docker
- `LANGSMITH_ENDPOINT` for self-hosted: https://docs.langchain.com/langsmith/env-var

## 1) LangChain

### 1.1 “Offline” mode (no cloud traces): disable tracing

**Option 1 — environment variable**
```bash
export LANGSMITH_TRACING=false
```
Source: https://docs.langchain.com/langsmith/env-var

**Option 2 — programmatic toggle (Python)**
```py
from langsmith import tracing_context

with tracing_context(enabled=False):
    # Run your LangChain pipeline here.
    ...
```
Source: https://docs.langchain.com/langsmith/trace-without-env-vars

**Option 3 — programmatic toggle (TypeScript)**
```ts
import { traceable } from "langsmith/traceable";

const fn = traceable(
  async () => "hello",
  { tracingEnabled: false }
);

await fn();
```
Source: https://docs.langchain.com/langsmith/trace-without-env-vars

### 1.2 “Keep traces private” (still uses LangSmith, but reduce what’s sent)

If you do use LangSmith (cloud or self-hosted), you can keep metadata while hiding sensitive payloads:

**Hide all inputs/outputs via env vars**
```bash
export LANGSMITH_HIDE_INPUTS=true
export LANGSMITH_HIDE_OUTPUTS=true
```
Source: https://docs.langchain.com/langsmith/mask-inputs-outputs

**Hide inputs/outputs via a LangSmith Client (Python)**
```py
import openai
from langsmith import Client
from langsmith.wrappers import wrap_openai

openai_client = wrap_openai(openai.Client())
langsmith_client = Client(
    hide_inputs=lambda inputs: {},
    hide_outputs=lambda outputs: {},
)

openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    langsmith_extra={"client": langsmith_client},
)
```
Source: https://docs.langchain.com/langsmith/mask-inputs-outputs

### 1.3 “Keep traces on-prem”: self-hosted LangSmith endpoint

To send traces to a self-hosted LangSmith instance:
```bash
export LANGSMITH_ENDPOINT="https://<your-langsmith-host>"
export LANGSMITH_API_KEY="<api-key-created-in-that-instance>"
export LANGSMITH_TRACING=true
```
Sources:
- `LANGSMITH_ENDPOINT` and `LANGSMITH_API_KEY`: https://docs.langchain.com/langsmith/env-var
- Self-hosted LangSmith: https://docs.langchain.com/langsmith/self-hosted

## 2) LangGraph agents (and Agent Server)

### 2.1 “Offline” mode with LangGraph CLI / local Agent Server

The LangGraph CLI and Agent Server can run locally; checkpoints/state are stored locally (disk or local Postgres/Redis depending on mode). Apart from CLI telemetry, “no data leaves the machine unless you have enabled tracing or your graph code explicitly contacts an external service.”

**Minimal local-only config**
```bash
# Opt out of CLI analytics
export LANGGRAPH_CLI_NO_ANALYTICS=1

# Ensure LangSmith tracing is off
export LANGSMITH_TRACING=false
```
Source: https://docs.langchain.com/langsmith/data-storage-and-privacy

### 2.2 Enabling tracing (cloud or self-hosted)

LangGraph’s observability pages show the basic env vars used to enable LangSmith tracing:
```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="<your-api-key>"
```
Source: https://docs.langchain.com/oss/python/langgraph/observability

For self-hosted LangSmith, also set `LANGSMITH_ENDPOINT` (see LangChain section above).

### 2.3 Best practices for traceability & observability (LangGraph)

**Attach tags/metadata for filtering and correlation**
```py
import langsmith as ls

@ls.traceable(tags=["env:dev"], metadata={"service": "agent-api"})
def handler(...):
    ...
```
Source: https://docs.langchain.com/langsmith/add-metadata-tags

**Mask sensitive data before it’s logged**
- Hide entire inputs/outputs (`LANGSMITH_HIDE_INPUTS`, `LANGSMITH_HIDE_OUTPUTS`): https://docs.langchain.com/langsmith/mask-inputs-outputs

**Use anonymizers for structured redaction (example: SSN patterns)**
```py
from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client
from langsmith.anonymizer import create_anonymizer

anonymizer = create_anonymizer([
    {"pattern": r"\b\d{3}-?\d{2}-?\d{4}\b", "replace": "<ssn>"}
])

tracer_client = Client(anonymizer=anonymizer)
tracer = LangChainTracer(client=tracer_client)

# Attach tracer via callbacks in your graph config (example shown in docs)
```
Source: https://docs.langchain.com/oss/python/langgraph/observability

**Sample traces in high-volume environments**
```bash
export LANGSMITH_TRACING_SAMPLING_RATE=0.25
```
Source: https://docs.langchain.com/langsmith/sample-traces

**Unify traces across services (distributed tracing)**
- When one service calls another (including Agent Server), propagate LangSmith trace headers so the request shows up as one trace.

Sources:
- General distributed tracing: https://docs.langchain.com/langsmith/distributed-tracing
- Agent Server distributed tracing: https://docs.langchain.com/langsmith/agent-server-distributed-tracing

**If you use Studio locally**
- Studio is a cloud UI that connects to your local server; docs note you can disable tracing so “no data leaves your local server.”

Source: https://docs.langchain.com/oss/python/langgraph/studio

### 2.4 Data handling notes (LangGraph local servers)

From the LangGraph CLI privacy doc:
- `langgraph dev` stores state to a local `.langgraph_api` directory.
- `langgraph up` stores checkpoints/assistants/etc. in local Postgres; Redis is used for pubsub.
- If tracing is disabled, “no user data is persisted externally unless your graph code explicitly contacts an external service.”

Source: https://docs.langchain.com/langsmith/data-storage-and-privacy

## 3) deepagents

deepagents is built on LangGraph and integrates with LangSmith for observability/evaluation/deployment, so the same “offline” controls apply: set `LANGSMITH_TRACING=false` to ensure nothing is sent to LangSmith.

Docs:
- Deep agents overview (ecosystem relationship): https://docs.langchain.com/oss/python/deepagents/overview

### 3.1 “Offline” mode (no cloud traces)
```bash
export LANGSMITH_TRACING=false
```
Source: https://docs.langchain.com/langsmith/env-var

### 3.2 Example deep agent (Python)
```py
from deepagents import create_deep_agent

research_instructions = """You are an expert researcher..."""

agent = create_deep_agent(
    system_prompt=research_instructions,
)
```
Source: https://docs.langchain.com/oss/python/deepagents/quickstart

### 3.3 Example deep agent (JavaScript/TypeScript)
```ts
import { createDeepAgent } from "deepagents";

const researchInstructions = `You are an expert researcher...`;

const agent = createDeepAgent({
  systemPrompt: researchInstructions,
});
```
Source: https://docs.langchain.com/oss/javascript/deepagents/quickstart

## Appendix: quick decision checklist

- Want **zero cloud storage**? Use `LANGSMITH_TRACING=false` (and for LangGraph CLI add `LANGGRAPH_CLI_NO_ANALYTICS=1`).
- Want LangSmith-style trace UI but **keep data in your infra**? You need **self-hosted LangSmith** and point `LANGSMITH_ENDPOINT` to it.
- Want to use LangSmith but reduce sensitive logging? Use `LANGSMITH_HIDE_INPUTS/OUTPUTS` and/or anonymizers before sending.

