"""
Q-Ravens Agents Module

Contains the specialized AI agents:
- Orchestrator: Central coordinator (Project Manager)
- Analyzer: Website analysis (QA Analyst)
- Designer: Test case design (Test Architect) - Coming in Phase 2
- Executor: Test execution (Automation Engineer)
- Reporter: Report generation (QA Lead) - Coming in Phase 2
- VisualAgent: See-Think-Act-Reflect cognitive loop agent
- DevOps: Infrastructure management - Coming in Phase 4
"""

from q_ravens.agents.base import BaseAgent
from q_ravens.agents.orchestrator import OrchestratorAgent, orchestrator_node
from q_ravens.agents.analyzer import AnalyzerAgent, analyzer_node
from q_ravens.agents.executor import ExecutorAgent, executor_node
from q_ravens.agents.visual_agent import VisualAgent, LoopState, LoopPhase, run_visual_test

__all__ = [
    "BaseAgent",
    "OrchestratorAgent",
    "orchestrator_node",
    "AnalyzerAgent",
    "analyzer_node",
    "ExecutorAgent",
    "executor_node",
    "VisualAgent",
    "LoopState",
    "LoopPhase",
    "run_visual_test",
]
