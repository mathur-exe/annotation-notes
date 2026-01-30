## DeepAgents vs LangGraph vs LangChain – Architectural Notes

### High-level roles
- **LangChain**: Component & integration layer. Provides models, tools, retrievers, vector stores, etc. It is mostly about *primitives* and abstractions, not a specific agent architecture.
- **LangGraph**: Low-level orchestration engine for *stateful, long-running workflows/agents*. You define graphs (nodes, edges, state), durable execution, memory, and human-in-the-loop (HITL). It is intentionally unopinionated about prompts and higher-level agent patterns.
- **DeepAgents**: An *opinionated agent harness* built on LangChain + LangGraph. It targets “deep”, long-horizon agents (Claude Code / Manus / Deep Research style) and packages together planning, filesystem, subagents, and a tuned system prompt so you don’t have to design that from scratch.

### What DeepAgents adds beyond raw LangGraph/LangChain

1. **Opinionated “deep agent” recipe**
- Exposes a single factory: `create_deep_agent(...)`.
- Under the hood it compiles a LangGraph `StateGraph`, but you don’t see graph wiring; you get a ready-made deep agent.
- Ships with a detailed, Claude-Code–inspired system prompt that:
  - Teaches the model how to plan and track TODOs.
  - Explains how to use filesystem tools.
  - Explains how to delegate to subagents.
- LangGraph/LangChain give you the ability to *define* such prompts but do not ship a general-purpose deep-agent prompt as a standard component.

2. **Planning & TODOs as first-class behavior**
- Built-in tools:
  - `write_todos` – create/update a structured task list.
  - `read_todos` – inspect the current list and status.
- `TodoListMiddleware` is always included in `create_deep_agent`, and the default prompt strongly encourages using TODOs for long-horizon tasks (breaking down work, marking items done, updating as the plan changes).
- LangChain has `TodoListMiddleware`, but it’s not wired into:
  - A default deep-agent factory, nor
  - A comprehensive system prompt describing best practices.

3. **Filesystem + shell tools with pluggable backends**
- DeepAgents defines a cohesive set of tools:
  - `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`.
  - `execute` (when the backend supports sandboxed command execution).
- Backends (`deepagents.backends`):
  - `StateBackend`: ephemeral in-memory “filesystem” in agent state.
  - `FilesystemBackend`: real disk, rooted at a specified directory.
  - `StoreBackend`: persistent storage via LangGraph Store.
  - `CompositeBackend`: route different paths to different backends (e.g., `/memories/` → Store, everything else → State).
- Context management pattern:
  - Large tool results can be auto-offloaded into files.
  - Prompts teach the model to read/write files rather than overfilling context.
- In contrast:
  - LangGraph has durable state, checkpoints, and Store, but does not provide a standard, opinionated filesystem tool layer + backend abstraction focused on “agent-as-operating-system” use cases.
  - LangChain has file tools, but not this unified, backend-aware middleware stack wired into a deep-agent harness.

4. **Subagents via a standard `task` tool**
- `SubAgentMiddleware` exposes:
  - A `task` tool that lets the main agent spawn subagents for isolated subtasks.
  - Subagents defined by name, description, system prompt, tools, and optionally a distinct model and additional middleware.
  - Support for wrapping full LangGraph graphs as subagents (`CompiledSubAgent`).
- The default DeepAgents prompt explains:
  - When to use subagents vs. doing work in the main agent.
  - How to keep the supervisor’s context clean and let subagents “go deep” on individual tasks.
- LangGraph already lets you define subgraphs / nested graphs, and LangChain can implement “agents calling agents”, but they don’t ship a canonical `task` tool + subagent middleware as a one-call pattern.

5. **Curated middleware stack for long-horizon runs**
- DeepAgents composes multiple behaviors you’d otherwise have to wire yourself:
  - `TodoListMiddleware`: planning/TODO management.
  - `FilesystemMiddleware`: file tools + auto offloading of large results.
  - `SubAgentMiddleware`: subagent delegation via `task`.
  - `SummarizationMiddleware`: automatic summarization when context grows beyond a threshold (~170k tokens).
  - `AnthropicPromptCachingMiddleware`: prompt caching for Anthropic models to reduce repeated system-prompt costs.
  - `PatchToolCallsMiddleware`: repairs dangling tool calls after interruptions.
  - `HumanInTheLoopMiddleware`: structured HITL, with `interrupt_on` config allowing humans to approve/edit/reject certain tool calls.
- All of this is behind `create_deep_agent`; with LangGraph/LangChain you would choose and configure each of these patterns yourself.

6. **Long-term memory path conventions**
- Uses `CompositeBackend` to implement a clear scheme:
  - Default path → ephemeral (scratch work, temporary files).
  - Special path(s) like `/memories/` → durable Store backend.
- Prompts explicitly explain:
  - When and how to store persistent knowledge or user preferences under those paths.
  - How to retrieve past information to continue multi-session workflows.
- LangGraph provides Store APIs; DeepAgents turns them into a concrete, opinionated memory architecture.

7. **DeepAgents CLI (product layer)**
- Separate package: `deepagents-cli`.
- Terminal-based coding assistant similar to Claude Code:
  - Tools: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `shell`, `execute` (remote sandbox), `web_search`, `fetch_url`, `task`, `write_todos`.
  - HITL approvals for destructive operations (file writes, shell, external web calls, delegation).
- Persistent memory model:
  - Global memory under `~/.deepagents/<agent_name>/agent.md`.
  - Project-specific memory under `[project-root]/.deepagents/agent.md` and additional `.md` files.
  - Middleware that:
    - Injects this memory into the prompt as `<user_memory>` / `<project_memory>`.
    - Guides the agent on when to read/update these files (e.g., after feedback, when patterns emerge, when asked to remember something).
- Skills system:
  - Skills are directories with `SKILL.md` files, loaded from global and project locations.
  - Follows progressive disclosure: skill names/descriptions are listed in the system prompt; full instructions are read only when relevant via `read_file`.
  - Provides reusable workflows (e.g., web research, LangGraph docs helper) without embedding all instructions in every prompt.
- LangGraph/LangChain provide the foundations but not this CLI UX, memory layout, or skills mechanism out of the box.

### When to choose what

- **Use LangChain directly** when:
  - You want low-level control over models/tools/retrievers.
  - Your app is not heavily agentic or long-running.
  - You’re building simple chains/pipelines rather than a complex agent.

- **Use LangGraph (with LangChain)** when:
  - You need stateful, long-running workflows with explicit graphs, nodes, edges.
  - You want full control over architecture, memory, and HITL.
  - You’re comfortable designing your own agent prompt, tools, and orchestration.

- **Use DeepAgents** when:
  - You want a Claude Code / Manus / Deep Research–style agent without reimplementing all design patterns yourself.
  - You need built-in planning, filesystem tooling, subagents, summarization, and HITL, tuned to work well together.
  - You want a ready-made, terminal coding/research assistant (`deepagents-cli`) with persistent memory and skills.

### Key takeaway
DeepAgents is not “yet another framework” competing with LangGraph or LangChain. Instead:
- LangChain = components & integrations.
- LangGraph = low-level stateful orchestration engine.
- DeepAgents = opinionated deep-agent harness and CLI, built *on top* of LangChain + LangGraph, packaging best-practice patterns (planning, filesystem, subagents, memory, HITL) into a single, ready-made abstraction.

