"""
Q-Ravens CLI

Command-line interface for Q-Ravens testing system.
"""

import typer
from rich.console import Console
from rich.panel import Panel

from q_ravens import __version__
from q_ravens.core.config import settings

app = typer.Typer(
    name="q-ravens",
    help="Autonomous QA Agent Swarm - AI-powered web application testing",
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Show Q-Ravens version."""
    console.print(f"Q-Ravens v{__version__}")


@app.command()
def config() -> None:
    """Show current configuration (without sensitive data)."""
    console.print(
        Panel(
            f"""[bold]Q-Ravens Configuration[/bold]

Default LLM Provider: {settings.default_llm_provider.value}
Default Model: {settings.default_model}
Log Level: {settings.log_level}
Max Browsers: {settings.max_browsers}
Default Timeout: {settings.default_timeout}s

[dim]Provider Status:[/dim]
  Anthropic: {"[green]configured[/green]" if settings.anthropic_api_key else "[red]not configured[/red]"}
  OpenAI: {"[green]configured[/green]" if settings.openai_api_key else "[red]not configured[/red]"}
  Google: {"[green]configured[/green]" if settings.google_api_key else "[red]not configured[/red]"}
  Groq: {"[green]configured[/green]" if settings.groq_api_key else "[red]not configured[/red]"}
  Azure: {"[green]configured[/green]" if settings.azure_openai_api_key else "[red]not configured[/red]"}
  Ollama: [green]available[/green] (local)
""",
            title="[bold blue]Q-Ravens[/bold blue]",
        )
    )


@app.command()
def test(
    url: str = typer.Argument(..., help="URL of the website to test"),
    provider: str = typer.Option(
        None, "--provider", "-p", help="LLM provider to use"
    ),
) -> None:
    """Run tests on a website (coming soon)."""
    console.print(
        Panel(
            f"""[bold yellow]Coming Soon![/bold yellow]

Target URL: {url}
Provider: {provider or settings.default_llm_provider.value}

The test command will be implemented in Phase 1.
This will orchestrate the agent swarm to:
1. Analyze the website structure
2. Design test cases
3. Execute tests with Playwright
4. Generate reports
""",
            title="[bold blue]Q-Ravens Test[/bold blue]",
        )
    )


if __name__ == "__main__":
    app()
