# Q-Ravens

**Autonomous QA Agent Swarm** - AI-powered web application testing

Q-Ravens is an open-source, autonomous multi-agent system that enables anyone to conduct comprehensive web application testing through simple natural language instructions.

## Features

- **Natural Language Interface** - Describe tests in plain English
- **Multi-Agent Architecture** - Specialized agents for analysis, design, execution, and reporting
- **Multi-LLM Support** - Choose from Anthropic, OpenAI, Google, Groq, or local Ollama
- **Playwright Integration** - Reliable cross-browser automation

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

## Configuration

Copy `.env.example` to `.env` and add at least one LLM provider API key:

- `ANTHROPIC_API_KEY` - Claude (recommended)
- `OPENAI_API_KEY` - GPT-4
- `GOOGLE_API_KEY` - Gemini
- `GROQ_API_KEY` - Fast inference
- Or use Ollama for local, private inference

## Project Status

Currently in **Phase 1: Foundation** - building the core agent framework.

## License

MIT License - see [LICENSE](LICENSE) for details.
