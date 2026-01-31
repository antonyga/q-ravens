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

                # Update state with user response - let human_node process it
                current_state = dict(final_state)
                current_state["user_input"] = user_response
                # Keep requires_user_input=True so routing goes to human_node
                # human_node will process the response and update state

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


@app.command()
def visual(
    url: str = typer.Argument(..., help="URL of the website to test"),
    goal: str = typer.Argument(..., help="Goal to accomplish (e.g., 'Click the login button')"),
    headless: bool = typer.Option(True, "--headless/--headed", help="Run browser in headless mode"),
    max_iterations: int = typer.Option(20, "--max-iterations", "-m", help="Maximum iterations"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    save_screenshots: bool = typer.Option(False, "--save-screenshots", "-s", help="Save screenshots to disk"),
    output_dir: str = typer.Option("./visual_output", "--output", "-o", help="Output directory for screenshots"),
) -> None:
    """Run visual-based testing using the See-Think-Act-Reflect loop."""
    from q_ravens.agents.visual_agent import VisualAgent
    from pathlib import Path

    console.print(
        Panel(
            f"[bold]Target:[/bold] {url}\n"
            f"[bold]Goal:[/bold] {goal}\n"
            f"[bold]Headless:[/bold] {headless}",
            title="[bold blue]Q-Ravens Visual Test[/bold blue]",
        )
    )

    async def run_visual_workflow():
        async with VisualAgent(headless=headless) as agent:
            agent._current_state = None

            # Override max iterations
            result = await agent.execute_goal(url, goal)

            # Save screenshots if requested
            if save_screenshots and result.iterations:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                for i, iteration in enumerate(result.iterations):
                    if iteration.visual_state:
                        from q_ravens.tools.vision import VisionService
                        vision = VisionService()
                        paths = vision.save_visual_state(
                            iteration.visual_state,
                            output_path,
                            prefix=f"iteration_{i+1}",
                        )
                        if verbose:
                            console.print(f"[dim]Saved screenshots for iteration {i+1}[/dim]")

            return result

    try:
        result = asyncio.run(run_visual_workflow())

        # Display results
        console.print()

        if result.success:
            status_color = "green"
            status_text = "SUCCESS"
        else:
            status_color = "red"
            status_text = "FAILED"

        # Summary panel
        duration = (result.completed_at - result.started_at).total_seconds() if result.completed_at else 0
        console.print(
            Panel(
                f"[bold]Status:[/bold] [{status_color}]{status_text}[/{status_color}]\n"
                f"[bold]Iterations:[/bold] {result.iteration_count}\n"
                f"[bold]Duration:[/bold] {duration:.1f}s\n"
                f"[bold]Failures:[/bold] {result.consecutive_failures}",
                title=f"[bold {status_color}]Results[/bold {status_color}]",
            )
        )

        # Goal decomposition info
        if result.goal_decomposition:
            steps_info = []
            for i, step in enumerate(result.goal_decomposition.steps):
                if i < result.goal_decomposition.current_step_index:
                    steps_info.append(f"  [green][DONE][/green] {step}")
                elif i == result.goal_decomposition.current_step_index:
                    steps_info.append(f"  [yellow][>>][/yellow] {step}")
                else:
                    steps_info.append(f"  [dim][--][/dim] {step}")

            console.print(
                Panel(
                    "\n".join(steps_info),
                    title="[bold]Goal Steps[/bold]",
                )
            )

        # Verbose output - iteration details
        if verbose and result.iterations:
            console.print("\n[bold]Iteration Details:[/bold]")
            for iteration in result.iterations:
                phase_color = "green" if iteration.success else "yellow"
                if iteration.error:
                    phase_color = "red"

                console.print(f"\n  [{phase_color}]Iteration {iteration.iteration_number}[/{phase_color}]")

                if iteration.reasoning:
                    console.print(f"    Scene: {iteration.reasoning.scene_description[:80]}...")
                    console.print(f"    Confidence: {iteration.reasoning.confidence.value}")

                if iteration.action_result:
                    console.print(f"    Action: {iteration.action_result.action_type} - {iteration.action_result.message}")

                if iteration.error:
                    console.print(f"    [red]Error: {iteration.error}[/red]")

        # Error message
        if result.error_message:
            console.print(
                Panel(
                    result.error_message,
                    title="[bold red]Error[/bold red]",
                )
            )

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def resume(
    session_id: str = typer.Argument(..., help="Session ID to resume (can be partial)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    auto_approve: bool = typer.Option(False, "--yes", "-y", help="Auto-approve all prompts"),
) -> None:
    """Resume a saved test session."""
    from q_ravens.core.checkpoint import CheckpointManager
    from q_ravens.core.runner import QRavensRunner

    manager = CheckpointManager()
    sessions_list = manager.list_sessions()

    # Find matching session (supports partial ID)
    matching = [s for s in sessions_list if s["session_id"].startswith(session_id)]

    if not matching:
        console.print(f"[red]No session found matching '{session_id}'[/red]")
        raise typer.Exit(1)

    if len(matching) > 1:
        console.print(f"[yellow]Multiple sessions match '{session_id}':[/yellow]")
        for s in matching:
            console.print(f"  {s['session_id']}")
        raise typer.Exit(1)

    full_session_id = matching[0]["session_id"]
    console.print(
        Panel(
            f"[bold]Session:[/bold] {full_session_id[:8]}...\n[bold]Auto-approve:[/bold] {auto_approve}",
            title="[bold blue]Resuming Session[/bold blue]",
        )
    )

    async def run_resume():
        runner = QRavensRunner(persist=True, session_id=full_session_id)

        # Get current state first
        current_state = await runner.get_state()
        if not current_state:
            console.print("[red]Failed to retrieve session state[/red]")
            return None

        phase = current_state.get("phase", "unknown")
        target_url = current_state.get("target_url", "unknown")
        console.print(f"[dim]Target: {target_url}[/dim]")
        console.print(f"[dim]Current phase: {phase}[/dim]")

        graph = await runner._ensure_graph()
        config = {"configurable": {"thread_id": full_session_id}}

        # Update state for approval handling
        current_state["require_approval"] = not auto_approve

        final_state = None
        max_loops = 20

        for loop in range(max_loops):
            async for state in graph.astream(current_state, config, stream_mode="values"):
                if state and isinstance(state, dict):
                    final_state = state
                    phase = state.get("phase", "unknown")
                    current_agent = state.get("current_agent", "resuming")
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

                # Update state with user response - let human_node process it
                current_state = dict(final_state)
                current_state["user_input"] = user_response
                # Keep requires_user_input=True so routing goes to human_node
                # human_node will process the response and update state

                continue

            # Check if workflow is complete
            phase = final_state.get("phase") if final_state else None
            if phase in [WorkflowPhase.COMPLETED.value, WorkflowPhase.ERROR.value]:
                break

            break

        return final_state

    try:
        final_state = asyncio.run(run_resume())

        if final_state and isinstance(final_state, dict):
            console.print()
            phase = final_state.get("phase")
            if phase == WorkflowPhase.COMPLETED.value:
                console.print("[green]Session completed successfully![/green]")
            elif phase == WorkflowPhase.ERROR.value:
                console.print("[red]Session ended with errors[/red]")
            else:
                console.print(f"[yellow]Session paused at phase: {phase}[/yellow]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
