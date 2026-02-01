# Q-Ravens Development Guide

This document provides code conventions, patterns, and instructions for AI assistants working on the Q-Ravens codebase.

## Project Overview

Q-Ravens is an autonomous QA agent swarm that uses LangGraph for multi-agent orchestration. It enables comprehensive web application testing through natural language instructions.

**Key Technologies:**
- **Agent Framework:** LangGraph 0.2.0+, LangChain 0.3.0+
- **LLM Support:** Anthropic, OpenAI, Google, Groq, Azure, Ollama
- **Browser Automation:** Playwright 1.40.0+
- **Configuration:** Pydantic 2.0+, pydantic-settings 2.0+
- **State Persistence:** aiosqlite 0.19.0+
- **CLI:** Typer, Rich

## Project Structure

```
src/q_ravens/
├── agents/              # Specialized AI agents
│   ├── base.py         # BaseAgent abstract class
│   ├── orchestrator.py # OrchestratorAgent (Project Manager)
│   ├── analyzer.py     # AnalyzerAgent (QA Analyst)
│   ├── designer.py     # DesignerAgent (Test Architect)
│   ├── executor.py     # ExecutorAgent (Automation Engineer)
│   ├── reporter.py     # ReporterAgent (QA Lead)
│   └── visual_agent.py # VisualAgent (See-Think-Act-Reflect)
├── core/               # Core workflow infrastructure
│   ├── graph.py        # LangGraph workflow definition
│   ├── runner.py       # QRavensRunner - main entry point
│   ├── state.py        # TypedDict state schema
│   ├── config.py       # Settings/configuration
│   ├── exceptions.py   # Custom exception hierarchy
│   ├── error_handler.py # Error handling utilities
│   ├── checkpoint.py   # State persistence
│   └── human_loop.py   # Human-in-the-loop
├── tools/              # External tools
│   ├── browser.py      # Playwright browser automation
│   ├── vision.py       # Vision/image processing
│   └── actions.py      # Action primitives
├── cli.py              # CLI interface (typer)
└── ui/                 # Streamlit chat interface (Phase 3)
    ├── __init__.py     # UI module exports
    ├── chat.py         # Main Streamlit chat application
    ├── components.py   # Reusable UI components
    └── settings.py     # UI configuration and settings
```

## Code Conventions

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `OrchestratorAgent`, `TestCase` |
| Functions/Methods | snake_case | `process()`, `invoke_llm()` |
| Constants | UPPER_CASE | `DEFAULT_RETRY_CONFIG` |
| Private Methods | Leading underscore | `_create_llm()`, `_invoke_with_retry()` |
| Agent Names | lowercase string | `"orchestrator"`, `"analyzer"` |

### Import Organization

Always organize imports in this order:

```python
# 1. Standard library
import asyncio
import json
from typing import Any, Optional

# 2. Third-party packages
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

# 3. Local application imports
from q_ravens.agents.base import BaseAgent
from q_ravens.core.state import QRavensState
from q_ravens.core.exceptions import QRavensError
```

### Type Hints

- **All functions must have type hints** for parameters and return types
- Use modern Python 3.11+ generics: `list[str]`, `dict[str, Any]`
- Use `Optional[T]` for nullable parameters
- Use `Annotated` types for LangGraph reducers
- Async functions return `dict[str, Any]` from `process()` methods

```python
async def invoke_llm(
    self,
    user_message: str,
    context: Optional[str] = None,
) -> str:
    """Invoke LLM with message."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def method_name(param1: str, param2: int) -> dict[str, Any]:
    """
    Short description of what the method does.

    Longer description with implementation details and context
    if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        SpecificError: When this error occurs
    """
```

**Module-level docstrings** are required at the top of every file.

### Async/Await Patterns

- All agent `process()` methods are `async`
- All LangGraph node functions are `async def`
- Use `async with` for context managers
- Use `await` for all I/O operations

```python
async def process(self, state: QRavensState) -> dict[str, Any]:
    """Process the current state."""
    result = await self._do_work()
    return {"result": result, "phase": WorkflowPhase.NEXT.value}
```

## Agent Implementation Pattern

Every agent must follow this structure:

```python
"""
Module docstring explaining the agent's role.
"""

from typing import Any, Optional
from langchain_core.messages import AIMessage

from q_ravens.agents.base import BaseAgent
from q_ravens.core.state import QRavensState, WorkflowPhase


class SpecializedAgent(BaseAgent):
    """
    Docstring explaining role and responsibilities.

    This agent handles X, Y, Z tasks in the Q-Ravens workflow.
    """

    # Class attributes (metadata)
    name = "agent_name"
    role = "Job Title"
    description = "What this agent does"

    def __init__(self, **kwargs):
        """Initialize the agent."""
        super().__init__(**kwargs)
        # Optional: initialize tools

    @property
    def system_prompt(self) -> str:
        """Return the system prompt defining agent behavior."""
        return """You are the Agent of Q-Ravens, a Role specialist.

        Your responsibilities:
        1. Task one
        2. Task two
        """

    async def process(self, state: QRavensState) -> dict[str, Any]:
        """
        Main entry point called by LangGraph.

        Args:
            state: Current workflow state

        Returns:
            State updates dictionary
        """
        # 1. Extract inputs with .get() and defaults
        target_url = state.get("target_url")
        errors = state.get("errors", [])

        # 2. Validate required inputs
        if not target_url:
            return {
                "phase": WorkflowPhase.ERROR.value,
                "errors": errors + ["Missing required input"],
                "current_agent": self.name,
            }

        try:
            # 3. Do work
            result = await self._do_something()

            # 4. Return state updates
            message = AIMessage(content=f"Completed task...")
            return {
                "result_field": result,
                "phase": WorkflowPhase.NEXT_PHASE.value,
                "next_agent": "next_agent_name",
                "messages": [message],
                "current_agent": self.name,
            }
        except Exception as e:
            return {
                "phase": WorkflowPhase.ERROR.value,
                "errors": errors + [f"Operation failed: {str(e)}"],
                "current_agent": self.name,
            }

    async def _do_something(self) -> ResultType:
        """Helper method with actual implementation."""
        pass


# Node function for LangGraph
async def agent_name_node(state: QRavensState) -> dict[str, Any]:
    """LangGraph node function for the Agent."""
    agent = SpecializedAgent()
    return await agent.process(state)
```

## State Management

### QRavensState Fields

```python
class QRavensState(TypedDict, total=False):
    # User input
    target_url: str
    user_request: str

    # Messages with LangGraph reducer
    messages: Annotated[list[BaseMessage], add_messages]

    # Workflow control
    phase: str
    current_agent: str
    next_agent: str
    iteration_count: int

    # Agent outputs
    analysis: AnalysisResult
    test_cases: list[TestCase]
    test_results: list[TestResult]
    report: TestReport  # Reporter output

    # Error handling
    errors: list[str]

    # Human-in-the-loop
    requires_user_input: bool
    user_input_prompt: str
```

### Workflow Phases

```python
class WorkflowPhase(Enum):
    INIT = "init"
    ANALYZING = "analyzing"
    DESIGNING = "designing"
    EXECUTING = "executing"
    REPORTING = "reporting"
    COMPLETED = "completed"
    ERROR = "error"
    WAITING_FOR_USER = "waiting_for_user"
```

### State Update Return Pattern

```python
return {
    # Required for workflow control
    "phase": WorkflowPhase.PHASE_NAME.value,
    "current_agent": self.name,

    # Optional: routing
    "next_agent": "agent_name",

    # Data fields (what this agent produces)
    "result_field": result_value,

    # Communication
    "messages": [AIMessage(content="Message")],

    # Error handling
    "errors": updated_errors_list,

    # Human loop (optional)
    "requires_user_input": True,
    "user_input_prompt": "Question for user?",
}
```

## LLM Invocation Pattern

```python
async def _generate_something(self, input_data: InputType) -> ResultType:
    """Generate something using LLM."""
    # Build context
    context = self._build_context(input_data)

    # Create prompt
    prompt = f"""Task description.

    {context}

    Specific instructions about output format.

    Return your response as JSON:
    ```json
    {{...}}
    ```
    """

    # Invoke LLM with retry logic built-in
    response = await self.invoke_llm(prompt)

    # Parse response
    result = self._parse_response(response)

    # Fallback if parsing fails
    if not result:
        result = self._generate_fallback()

    return result
```

## JSON Parsing Pattern

```python
def _parse_json_response(self, response: str) -> Optional[dict]:
    """Parse JSON from LLM response."""
    import re
    import json

    # Try markdown code blocks first
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON
    try:
        # Find JSON object or array
        json_match = re.search(r'[\[{][\s\S]*[\]}]', response)
        if json_match:
            return json.loads(json_match.group(0))
    except json.JSONDecodeError:
        pass

    return None
```

## Error Handling

### Custom Exception Hierarchy

```
QRavensError (base)
├── LLMError
│   ├── LLMConnectionError
│   ├── LLMRateLimitError
│   ├── LLMResponseError
│   └── LLMTimeoutError
├── AgentError
│   ├── AgentExecutionError
│   └── AgentTimeoutError
├── WorkflowError
│   ├── WorkflowStateError
│   └── MaxIterationsError
├── ToolError
│   ├── BrowserError
│   └── BrowserTimeoutError
└── ValidationError
```

### Error Response Pattern

```python
try:
    result = await self._do_work()
    return {
        "result": result,
        "phase": WorkflowPhase.NEXT.value,
        "current_agent": self.name,
    }
except Exception as e:
    return {
        "phase": WorkflowPhase.ERROR.value,
        "errors": errors + [f"Operation failed: {str(e)}"],
        "current_agent": self.name,
    }
```

## Adding a New Agent

1. **Create agent file** in `src/q_ravens/agents/`
2. **Implement the agent class** following the pattern above
3. **Add node function** at the bottom of the file
4. **Update `__init__.py`** to export the agent and node function
5. **Update `graph.py`** to add the node and routing logic
6. **Update state.py** if new state fields are needed

## Code Quality

### Linting (Ruff)

```bash
ruff check src/
ruff format src/
```

Rules enabled: E, F, I, N, W, UP
Line length: 100 characters

### Type Checking (MyPy)

```bash
mypy src/q_ravens/
```

Strict mode enabled.

### Testing

```bash
pytest tests/
```

Use pytest-asyncio for async tests.

## Key Files Reference

### Core Modules

| File | Purpose |
|------|---------|
| [base.py](src/q_ravens/agents/base.py) | BaseAgent with LLM integration and retry logic |
| [state.py](src/q_ravens/core/state.py) | QRavensState TypedDict and Pydantic models |
| [graph.py](src/q_ravens/core/graph.py) | LangGraph workflow definition |
| [runner.py](src/q_ravens/core/runner.py) | QRavensRunner entry point |
| [exceptions.py](src/q_ravens/core/exceptions.py) | Custom exception hierarchy |
| [config.py](src/q_ravens/core/config.py) | Settings and configuration |

### Testing Tools

| File | Purpose |
|------|---------|
| [browser.py](src/q_ravens/tools/browser.py) | Playwright browser automation |
| [lighthouse.py](src/q_ravens/tools/lighthouse.py) | Performance testing (Core Web Vitals) |
| [accessibility.py](src/q_ravens/tools/accessibility.py) | WCAG 2.1 accessibility testing (axe-core) |
| [vision.py](src/q_ravens/tools/vision.py) | Screenshot capture and SoM overlay |
| [actions.py](src/q_ravens/tools/actions.py) | Browser action primitives |

## Testing Tools Usage

### Lighthouse (Performance Testing)

```python
from q_ravens.tools.lighthouse import LighthouseTool, run_lighthouse_audit

# Quick audit
result = await run_lighthouse_audit("https://example.com")
print(f"Performance Score: {result.performance_score}")
print(f"LCP: {result.core_web_vitals.lcp_ms}ms")

# Detailed audit with tool instance
tool = LighthouseTool(device="desktop")
result = await tool.run_audit(
    url="https://example.com",
    categories=[PerformanceCategory.PERFORMANCE],
    save_report=True,
)
print(tool.format_summary(result))
```

**Core Web Vitals Thresholds:**
- LCP (Largest Contentful Paint): Good < 2.5s, Needs Improvement < 4s
- TBT (Total Blocking Time): Good < 200ms, Needs Improvement < 600ms
- CLS (Cumulative Layout Shift): Good < 0.1, Needs Improvement < 0.25

**Requirements:** `npm install -g lighthouse`

### Axe-core (Accessibility Testing)

```python
from q_ravens.tools.accessibility import AccessibilityTool, WCAGLevel

# Quick audit
result = await run_accessibility_audit("https://example.com")
print(f"WCAG AA Compliance: {result.estimated_compliance}%")
print(f"Critical Violations: {result.critical_count}")

# Detailed audit with tool instance
tool = AccessibilityTool(wcag_level=WCAGLevel.AA)
result = await tool.run_audit(
    url="https://example.com",
    include_passes=True,
)
print(tool.format_summary(result))

# Get remediation priorities
priorities = tool.get_remediation_priority(result)
for p in priorities[:5]:
    print(f"[{p['impact']}] {p['id']}: {p['description']}")
```

**Impact Levels:**
- Critical: Blocks access for some users
- Serious: Significantly impacts user experience
- Moderate: Moderately impacts user experience
- Minor: Minor impact on user experience

## Common Patterns

### Parameter Extraction

```python
# Always use .get() with defaults
target_url = state.get("target_url")
errors = state.get("errors", [])
analysis = state.get("analysis")
iteration = state.get("iteration_count", 0) + 1
```

### Pydantic Models

```python
class ResultModel(BaseModel):
    """Model for structured data."""

    field1: str = Field(description="Description of field1")
    field2: int = Field(default=0, description="Description of field2")
    optional_field: Optional[str] = Field(default=None)

    model_config = {"extra": "ignore"}
```

### Context Managers for Resources

```python
@asynccontextmanager
async def get_resource(self):
    """Get resource with proper cleanup."""
    resource = await self._create_resource()
    try:
        yield resource
    finally:
        await resource.cleanup()
```

## Chat UI (Streamlit)

The Q-Ravens Chat UI provides a conversational interface for running QA tests.

### Running the UI

```bash
# Install UI dependencies
pip install q-ravens[ui]

# Launch the UI
q-ravens ui

# Or with custom port/host
q-ravens ui --port 8080 --host 0.0.0.0
```

### UI Structure

| File | Purpose |
|------|---------|
| [chat.py](src/q_ravens/ui/chat.py) | Main Streamlit application |
| [components.py](src/q_ravens/ui/components.py) | Reusable UI components |
| [settings.py](src/q_ravens/ui/settings.py) | UI configuration and settings |

### UI Features

- **Chat Interface**: Natural language test requests
- **Progress Timeline**: Visual workflow progress indicator
- **Agent Status Cards**: Real-time agent activity display
- **Test Results Tab**: Organized test result display with pass/fail metrics
- **Report Tab**: Full test report with executive summary
- **Human-in-the-Loop**: Approval gates for test execution
- **Session Management**: New session/export functionality

### UI Components Usage

```python
from q_ravens.ui.components import (
    render_agent_card,
    render_test_result_card,
    render_progress_bar,
    apply_custom_css,
)

# Render an agent status card
render_agent_card("executor", is_active=True)

# Render a test result
render_test_result_card(
    test_name="Login Test",
    status="passed",
    category="Smoke Test",
    description="Verify user can login",
)

# Render a progress bar
render_progress_bar(current=5, total=10, label="Tests Completed")
```

### UI Settings

```python
from q_ravens.ui.settings import UISettings, TestScope

settings = UISettings(
    theme=Theme.DARK,
    default_test_scopes=[TestScope.SMOKE, TestScope.FUNCTIONAL],
    max_iterations=10,
    persist_state=True,
    require_human_approval=True,
)
```

## PRD Reference

The full Product Requirements Document is available at:
[Q-Ravens_PRD.pdf](documentation/Q-Ravens_Product_Requirement/Q-Ravens_PRD.pdf)

Key agents defined in PRD:
- **Orchestrator** (Project Manager): Central coordinator
- **Analyzer** (QA Analyst): Website analysis and mapping
- **Designer** (Test Architect): Test case generation
- **Executor** (Automation Engineer): Test execution
- **Reporter** (QA Lead): Report generation
- **DevOps** (Infrastructure): Environment management (Phase 4)
