"""
Q-Ravens CLI

Command-line interface for Q-Ravens testing system.
"""

import asyncio
from uuid import uuid4

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from q_ravens import __version__
from q_ravens.core.config import settings
from q_ravens.core.state import WorkflowPhase

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
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    auto_approve: bool = typer.Option(False, "--yes", "-y", help="Auto-approve all prompts"),
    no_persist: bool = typer.Option(False, "--no-persist", help="Don't persist session state"),
) -> None:
    """Run tests on a website."""
    from q_ravens.core.runner import QRavensRunner

    console.print(
        Panel(
            f"[bold]Target:[/bold] {url}\n[bold]Auto-approve:[/bold] {auto_approve}",
            title="[bold blue]Q-Ravens Test[/bold blue]",
        )
    )

    async def run_workflow():
        session_id = str(uuid4())
        runner = QRavensRunner(persist=not no_persist, session_id=session_id)
        graph = await runner._ensure_graph()

        # Initial state
        current_state = {
            "target_url": url,
            "user_request": f"Test the website at {url}",
            "session_id": session_id,
            "phase": WorkflowPhase.INIT.value,
            "iteration_count": 0,
            "max_iterations": 10,
            "messages": [],
            "errors": [],
            "test_cases": [],
            "test_results": [],
            "require_approval": not auto_approve,  # Skip approval if -y flag
        }

        config = {"configurable": {"thread_id": session_id}}
        final_state = None
        max_loops = 20  # Safety limit

        for loop in range(max_loops):
            # Run the workflow until it pauses or completes
            async for state in graph.astream(current_state, config, stream_mode="values"):
                if state and isinstance(state, dict):
                    final_state = state
                    phase = state.get("phase", "unknown")
                    current_agent = state.get("current_agent", "starting")
                    console.print(f"[cyan]{current_agent}[/cyan]: {phase}")

                    if verbose:
                        messages = state.get("messages", [])
                        if messages:
                            for msg in messages[-1:]:
                                content = getattr(msg, "content", str(msg))
                                console.print(f"  [dim]{content[:100]}...[/dim]")

            # Check if workflow needs user input
            if final_state and final_state.get("requires_user_input"):
                prompt = final_state.get("user_input_prompt", "Continue?")

                console.print()
                console.print(Panel(prompt, title="[bold yellow]Approval Required[/bold yellow]"))

                if auto_approve:
                    console.print("[green]Auto-approved[/green]")
                    user_response = "yes"
                else:
                    user_response = "yes" if Confirm.ask("Approve?") else "no"

                # Update state with user response
                approved = user_response.lower() in ["yes", "y", "true", "1"]
                current_state = dict(final_state)
                current_state["user_input"] = user_response
                current_state["requires_user_input"] = False
                current_state["approval_response"] = {"approved": approved}
                current_state["phase"] = WorkflowPhase.ANALYZING.value  # Resume from analysis

                continue

            # Check if workflow is complete
            phase = final_state.get("phase") if final_state else None
            if phase in [WorkflowPhase.COMPLETED.value, WorkflowPhase.ERROR.value]:
                break

            # If no user input needed and not complete, something is wrong
            break

        return final_state

    try:
        final_state = asyncio.run(run_workflow())

        # Display results
        if final_state and isinstance(final_state, dict):
            console.print()

            # Show analysis results
            analysis = final_state.get("analysis")
            if analysis:
                console.print(
                    Panel(
                        f"""[bold]Pages Discovered:[/bold] {len(analysis.pages_discovered)}
[bold]Technologies:[/bold] {', '.join(analysis.technology_stack) or 'None detected'}
[bold]Has Authentication:[/bold] {analysis.has_authentication}""",
                        title="[bold green]Analysis Results[/bold green]",
                    )
                )

            # Show test results
            test_results = final_state.get("test_results", [])
            if test_results:
                passed = sum(1 for r in test_results if r.status.value == "passed")
                failed = sum(1 for r in test_results if r.status.value == "failed")

                status_color = "green" if failed == 0 else "red"

                result_lines = []
                for r in test_results:
                    icon = "[green]PASS[/green]" if r.status.value == "passed" else "[red]FAIL[/red]"
                    result_lines.append(f"  {icon} {r.test_id}")
                    if r.error_message:
                        result_lines.append(f"       [dim]{r.error_message}[/dim]")

                console.print(
                    Panel(
                        f"""[bold]Total:[/bold] {len(test_results)} tests
[bold]Passed:[/bold] [{status_color}]{passed}[/{status_color}]
[bold]Failed:[/bold] [{status_color}]{failed}[/{status_color}]

[bold]Details:[/bold]
{chr(10).join(result_lines)}""",
                        title=f"[bold {status_color}]Test Results[/bold {status_color}]",
                    )
                )

            # Show errors if any
            errors = final_state.get("errors", [])
            if errors:
                console.print(
                    Panel(
                        "\n".join(errors),
                        title="[bold red]Errors[/bold red]",
                    )
                )

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def sessions() -> None:
    """List saved test sessions."""
    from q_ravens.core.checkpoint import CheckpointManager
    from datetime import datetime

    manager = CheckpointManager()
    sessions = manager.list_sessions()

    if not sessions:
        console.print("[dim]No saved sessions found.[/dim]")
        return

    console.print(Panel(
        "\n".join(
            f"  {s['session_id'][:8]}... | {s['size_kb']:.1f} KB | "
            f"{datetime.fromtimestamp(s['modified']).strftime('%Y-%m-%d %H:%M')}"
            for s in sessions
        ),
        title="[bold blue]Saved Sessions[/bold blue]",
    ))


if __name__ == "__main__":
    app()
