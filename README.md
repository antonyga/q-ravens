# Q-Ravens

**Autonomous QA Agent Swarm** - AI-powered web application testing with semantic understanding

Q-Ravens is an open-source, autonomous multi-agent system that enables anyone to conduct comprehensive web application testing through simple natural language instructions. Unlike traditional test automation scripts, Q-Ravens agents **reason** about what they observe, **understand** page content in any language, and **adapt** to unexpected scenarios.

## Why Q-Ravens?

| Traditional Test Automation | Q-Ravens Agents |
|----------------------------|-----------------|
| Hardcoded test steps | LLM-generated test logic |
| Fixed selectors | Dynamic element discovery |
| Predefined assertions | Context-aware semantic validation |
| Fails on unexpected UI | Reasons about changes and adapts |
| One script = one test | One request = many tests |
| Language-specific | Multi-language understanding |

## Key Features

### Autonomous Agent Architecture
- **ReAct Pattern** - Agents reason, act, observe, and reflect
- **Multi-Agent Orchestration** - Specialized agents collaborate via LangGraph
- **Human-in-the-Loop** - Approval gates for critical decisions

### Semantic Verification
- **Language-Agnostic** - Understands error messages in English, Spanish, French, German, Portuguese
- **LLM-Powered Analysis** - Uses AI to interpret page content and determine test outcomes
- **Context-Aware** - Compares actual results against test intent, not just literal patterns

### Comprehensive Testing
- **Functional Testing** - UI interactions, navigation, forms, workflows
- **Performance Testing** - Core Web Vitals via Lighthouse (LCP, TBT, CLS)
- **Accessibility Testing** - WCAG 2.1 AA compliance via axe-core

### Multi-LLM Support
- Anthropic Claude (recommended)
- OpenAI GPT-4
- Google Gemini
- Groq (fast inference)
- Ollama (local, private)

## Agent Swarm

| Agent | Role | Responsibility |
|-------|------|----------------|
| **Orchestrator** | Project Manager | Coordinates workflow, routes between agents |
| **Analyzer** | QA Analyst | Analyzes website structure, identifies test targets |
| **Designer** | Test Architect | Generates test cases from natural language requests |
| **Executor** | Automation Engineer | Executes tests with Playwright, semantic verification |
| **Reporter** | QA Lead | Generates comprehensive test reports |
| **VisualAgent** | See-Think-Act-Reflect | Vision-based interaction for complex UIs |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/q-ravens/q-ravens.git
cd q-ravens

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e ".[dev]"

# Install Playwright browsers
playwright install

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run Q-Ravens
q-ravens --help
```

## Usage Examples

### CLI Usage
```bash
# Run smoke tests on a website
q-ravens test https://example.com --request "Run basic smoke tests"

# Test specific functionality
q-ravens test https://myapp.com --request "Test the login flow with valid and invalid credentials"

# Generate accessibility report
q-ravens test https://myapp.com --request "Check WCAG 2.1 AA compliance"
```

### Chat UI
```bash
# Launch the Streamlit chat interface
q-ravens ui

# With custom settings
q-ravens ui --port 8080 --host 0.0.0.0
```

## Configuration

Copy `.env.example` to `.env` and configure at least one LLM provider:

```env
# Required: At least one LLM provider
ANTHROPIC_API_KEY=your-key-here    # Claude (recommended)
OPENAI_API_KEY=your-key-here       # GPT-4
GOOGLE_API_KEY=your-key-here       # Gemini
GROQ_API_KEY=your-key-here         # Fast inference

# Optional: Local inference
OLLAMA_BASE_URL=http://localhost:11434

# Optional: Browser settings
HEADLESS=true
BROWSER_TIMEOUT=30000
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Request                            │
│              "Test login with invalid credentials"           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                        │
│              (Coordinates workflow phases)                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   Analyzer    │ │   Designer    │ │   Executor    │
│  (Discovery)  │ │ (Test Cases)  │ │  (Run Tests)  │
└───────────────┘ └───────────────┘ └───────┬───────┘
                                            │
                          ┌─────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Semantic Verification                       │
│  • Detects errors in ANY language                           │
│  • Uses LLM to interpret page meaning                       │
│  • Compares against test intent                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Reporter Agent                            │
│              (Generates test report)                         │
└─────────────────────────────────────────────────────────────┘
```

## Semantic Verification

Q-Ravens uses **semantic verification** to understand test outcomes beyond literal pattern matching:

```
Traditional Approach (Broken):
  Step: "Verify login succeeds"
  Found: error element with "Dirección de correo electrónico desconocida"
  Result: ✅ PASSED (just because it found an element)

Q-Ravens Semantic Verification:
  Step: "Verify login succeeds"
  Found: error element with "Dirección de correo electrónico desconocida"
  LLM Analysis: "This Spanish text means 'Unknown email address' - authentication FAILED"
  Result: ❌ FAILED (semantic understanding)
```

### Supported Languages
Authentication error detection in:
- 🇺🇸 English
- 🇪🇸 Spanish
- 🇫🇷 French
- 🇩🇪 German
- 🇧🇷 Portuguese

## Project Structure

```
src/q_ravens/
├── agents/              # Specialized AI agents
│   ├── base.py         # BaseAgent with LLM integration
│   ├── orchestrator.py # Workflow coordinator
│   ├── analyzer.py     # Website analysis
│   ├── designer.py     # Test case generation
│   ├── executor.py     # Test execution + semantic verification
│   ├── reporter.py     # Report generation
│   └── visual_agent.py # Vision-based interaction
├── core/               # Workflow infrastructure
│   ├── graph.py        # LangGraph workflow
│   ├── runner.py       # Main entry point
│   ├── state.py        # State management
│   └── config.py       # Configuration
├── tools/              # Automation tools
│   ├── browser.py      # Playwright integration
│   ├── lighthouse.py   # Performance testing
│   ├── accessibility.py # WCAG testing
│   └── vision.py       # Screenshot/SoM
└── ui/                 # Streamlit chat UI
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linting
ruff check src/
ruff format src/

# Run type checking
mypy src/q_ravens/

# Run tests
pytest tests/
```

## Roadmap

- [x] **Phase 1**: Core agent framework
- [x] **Phase 2**: Semantic verification, multi-language support
- [ ] **Phase 3**: Enhanced UI, session management
- [ ] **Phase 4**: DevOps agent, CI/CD integration
- [ ] **Phase 5**: Visual regression testing

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [Documentation](docs/)
- [PRD](documentation/Q-Ravens_Product_Requirement/Q-Ravens_PRD.pdf)
- [Issue Tracker](https://github.com/q-ravens/q-ravens/issues)
