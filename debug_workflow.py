"""Debug script to test the Q-Ravens workflow."""

import asyncio
import logging
import sys

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Enable debug logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

from q_ravens.core.runner import QRavensRunner
from q_ravens.core.state import WorkflowPhase


async def test_workflow():
    """Test the workflow with the specified URL and instruction."""
    url = "https://carlitos.com/"
    user_request = "test if the item: 'Tequila Ocho Blanco' is on the Menu page"

    print(f"\n{'='*60}")
    print(f"Q-Ravens Debug Test")
    print(f"URL: {url}")
    print(f"Request: {user_request}")
    print(f"{'='*60}\n")

    runner = QRavensRunner(persist=False)  # Don't persist for debugging

    state_count = 0
    last_phase = None
    last_agent = None

    try:
        async for state in runner.stream(url, user_request):
            state_count += 1
            phase = state.get("phase", "unknown")
            current_agent = state.get("current_agent", "unknown")
            requires_input = state.get("requires_user_input", False)
            user_input_prompt = state.get("user_input_prompt", "")
            approval_response = state.get("approval_response")
            user_input = state.get("user_input")
            next_agent = state.get("next_agent")
            test_cases = state.get("test_cases", [])
            errors = state.get("errors", [])

            # Only print when something changes
            if phase != last_phase or current_agent != last_agent:
                print(f"\n[State #{state_count}]")
                print(f"  Phase: {phase}")
                print(f"  Current Agent: {current_agent}")
                print(f"  Next Agent: {next_agent}")
                print(f"  Requires Input: {requires_input}")
                print(f"  User Input: {user_input}")
                print(f"  Approval Response: {approval_response}")
                print(f"  Test Cases: {len(test_cases)}")
                if errors:
                    print(f"  Errors: {errors}")

                if requires_input and user_input_prompt:
                    print(f"\n  === WAITING FOR USER INPUT ===")
                    prompt_preview = user_input_prompt[:300] if len(user_input_prompt) > 300 else user_input_prompt
                    print(f"  Prompt: {prompt_preview}...")

                last_phase = phase
                last_agent = current_agent

            # Check if we need to provide input (simulate approval)
            if requires_input:
                print(f"\n{'='*60}")
                print("WORKFLOW PAUSED FOR USER INPUT")
                prompt_preview = user_input_prompt[:500] if len(user_input_prompt) > 500 else user_input_prompt
                print(f"Prompt: {prompt_preview}")
                print(f"{'='*60}")

                # Simulate user approval
                print("\nSimulating user approval: 'yes'")

                # Now test the resume
                print("\n--- RESUMING WORKFLOW ---\n")

                async for resume_state in runner.stream_resume("yes"):
                    state_count += 1
                    r_phase = resume_state.get("phase", "unknown")
                    r_agent = resume_state.get("current_agent", "unknown")
                    r_requires = resume_state.get("requires_user_input", False)
                    r_approval = resume_state.get("approval_response")
                    r_next = resume_state.get("next_agent")
                    r_test_cases = resume_state.get("test_cases", [])
                    r_test_results = resume_state.get("test_results", [])
                    r_errors = resume_state.get("errors", [])

                    print(f"\n[Resume State #{state_count}]")
                    print(f"  Phase: {r_phase}")
                    print(f"  Current Agent: {r_agent}")
                    print(f"  Next Agent: {r_next}")
                    print(f"  Requires Input: {r_requires}")
                    print(f"  Approval Response: {r_approval}")
                    print(f"  Test Cases: {len(r_test_cases)}")
                    print(f"  Test Results: {len(r_test_results)}")
                    if r_errors:
                        print(f"  Errors: {r_errors}")

                    if r_phase == WorkflowPhase.COMPLETED.value:
                        print("\n*** WORKFLOW COMPLETED ***")
                        break
                    if r_phase == WorkflowPhase.ERROR.value:
                        print(f"\n*** WORKFLOW ERROR ***")
                        print(f"Errors: {r_errors}")
                        break

                break  # Exit the outer loop after resume

            # Check for completion
            if phase == WorkflowPhase.COMPLETED.value:
                print("\n*** WORKFLOW COMPLETED ***")
                break
            if phase == WorkflowPhase.ERROR.value:
                print(f"\n*** WORKFLOW ERROR ***")
                print(f"Errors: {errors}")
                break

    except Exception as e:
        print(f"\n*** EXCEPTION: {type(e).__name__}: {e} ***")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Total states processed: {state_count}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(test_workflow())
