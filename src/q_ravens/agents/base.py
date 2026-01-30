"""
Q-Ravens Base Agent

Abstract base class for all Q-Ravens agents.
Provides common functionality for LLM interaction and state management.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from q_ravens.core.config import LLMProvider, settings
from q_ravens.core.state import QRavensState


class BaseAgent(ABC):
    """
    Abstract base class for all Q-Ravens agents.

    Each agent has:
    - A name and role description
    - Access to an LLM
    - A system prompt defining its behavior
    - A process method that transforms state
    """

    name: str = "base_agent"
    role: str = "Base Agent"
    description: str = "Abstract base agent"

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the agent.

        Args:
            llm: Pre-configured LLM instance (optional)
            provider: LLM provider to use (defaults to settings)
            model: Model name to use (defaults to settings)
        """
        self.llm = llm or self._create_llm(provider, model)

    def _create_llm(
        self,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
    ) -> BaseChatModel:
        """Create an LLM instance based on configuration."""
        provider = provider or settings.default_llm_provider
        model = model or settings.default_model

        if provider == LLMProvider.ANTHROPIC:
            api_key = settings.get_api_key(LLMProvider.ANTHROPIC)
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not configured")
            return ChatAnthropic(
                model=model,
                api_key=api_key,
                max_tokens=4096,
            )

        elif provider == LLMProvider.OPENAI:
            api_key = settings.get_api_key(LLMProvider.OPENAI)
            if not api_key:
                raise ValueError("OPENAI_API_KEY not configured")
            return ChatOpenAI(
                model=model or "gpt-4o",
                api_key=api_key,
            )

        elif provider == LLMProvider.GOOGLE:
            api_key = settings.get_api_key(LLMProvider.GOOGLE)
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not configured")
            return ChatGoogleGenerativeAI(
                model=model or "gemini-2.0-flash",
                google_api_key=api_key,
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt that defines this agent's behavior."""
        pass

    @abstractmethod
    async def process(self, state: QRavensState) -> dict[str, Any]:
        """
        Process the current state and return updates.

        This is the main entry point called by LangGraph.
        Must return a dictionary of state updates.

        Args:
            state: Current workflow state

        Returns:
            Dictionary of state field updates
        """
        pass

    async def invoke_llm(
        self,
        user_message: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Invoke the LLM with the agent's system prompt.

        Args:
            user_message: The user/task message
            context: Optional additional context

        Returns:
            The LLM's response text
        """
        messages = [SystemMessage(content=self.system_prompt)]

        if context:
            messages.append(HumanMessage(content=f"Context:\n{context}"))

        messages.append(HumanMessage(content=user_message))

        response = await self.llm.ainvoke(messages)
        return response.content

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, role={self.role})>"
