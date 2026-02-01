"""
Q-Ravens Base Agent

Abstract base class for all Q-Ravens agents.
Provides common functionality for LLM interaction and state management.
"""

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from q_ravens.core.config import LLMProvider, settings
from q_ravens.core.state import QRavensState
from q_ravens.core.exceptions import (
    LLMConfigurationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    RetryableConnectionError,
    RetryableRateLimitError,
    RetryableTimeoutError,
    is_retryable,
)

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """
        Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff calculation
            jitter: Whether to add random jitter to delays
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """
        Calculate the delay for a given retry attempt.

        Args:
            attempt: The current retry attempt (0-indexed)

        Returns:
            Delay in seconds before next retry
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add up to 25% random jitter
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount

        return delay


# Default retry configuration
DEFAULT_RETRY_CONFIG = RetryConfig()


class BaseAgent(ABC):
    """
    Abstract base class for all Q-Ravens agents.

    Each agent has:
    - A name and role description
    - Access to an LLM
    - A system prompt defining its behavior
    - A process method that transforms state
    - Retry logic for transient failures
    """

    name: str = "base_agent"
    role: str = "Base Agent"
    description: str = "Abstract base agent"

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize the agent.

        Args:
            llm: Pre-configured LLM instance (optional)
            provider: LLM provider to use (defaults to settings)
            model: Model name to use (defaults to settings)
            api_key: API key for the LLM provider (overrides env vars)
            retry_config: Configuration for retry behavior
        """
        self.llm = llm or self._create_llm(provider, model, api_key)
        self.retry_config = retry_config or DEFAULT_RETRY_CONFIG

    def _create_llm(
        self,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> BaseChatModel:
        """Create an LLM instance based on configuration."""
        provider = provider or settings.default_llm_provider
        model = model or settings.default_model

        if provider == LLMProvider.ANTHROPIC:
            key = api_key or settings.get_api_key(LLMProvider.ANTHROPIC)
            if not key:
                raise LLMConfigurationError(
                    "ANTHROPIC_API_KEY not configured",
                    details={"provider": provider.value},
                )
            return ChatAnthropic(
                model=model,
                api_key=key,
                max_tokens=4096,
            )

        elif provider == LLMProvider.OPENAI:
            key = api_key or settings.get_api_key(LLMProvider.OPENAI)
            if not key:
                raise LLMConfigurationError(
                    "OPENAI_API_KEY not configured",
                    details={"provider": provider.value},
                )
            return ChatOpenAI(
                model=model or "gpt-4o",
                api_key=key,
            )

        elif provider == LLMProvider.GOOGLE:
            key = api_key or settings.get_api_key(LLMProvider.GOOGLE)
            if not key:
                raise LLMConfigurationError(
                    "GOOGLE_API_KEY not configured",
                    details={"provider": provider.value},
                )
            return ChatGoogleGenerativeAI(
                model=model or "gemini-2.0-flash",
                google_api_key=key,
            )

        elif provider == LLMProvider.GROQ:
            key = api_key or settings.get_api_key(LLMProvider.GROQ)
            if not key:
                raise LLMConfigurationError(
                    "GROQ_API_KEY not configured",
                    details={"provider": provider.value},
                )
            # Groq uses OpenAI-compatible API
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=model or "llama-3.3-70b-versatile",
                api_key=key,
            )

        else:
            raise LLMConfigurationError(
                f"Unsupported provider: {provider}",
                details={"provider": str(provider)},
            )

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

        Includes automatic retry logic for transient failures.

        Args:
            user_message: The user/task message
            context: Optional additional context

        Returns:
            The LLM's response text

        Raises:
            LLMConnectionError: If connection fails after all retries
            LLMRateLimitError: If rate limited after all retries
            LLMTimeoutError: If request times out after all retries
            LLMResponseError: If response is invalid
        """
        messages = [SystemMessage(content=self.system_prompt)]

        if context:
            messages.append(HumanMessage(content=f"Context:\n{context}"))

        messages.append(HumanMessage(content=user_message))

        return await self._invoke_with_retry(messages)

    async def _invoke_with_retry(self, messages: list) -> str:
        """
        Invoke the LLM with retry logic for transient failures.

        Args:
            messages: List of messages to send to the LLM

        Returns:
            The LLM's response text

        Raises:
            Various LLM exceptions after retries are exhausted
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                response = await self.llm.ainvoke(messages)

                if response is None or not hasattr(response, "content"):
                    raise LLMResponseError(
                        "Invalid response from LLM",
                        details={"response": str(response)},
                    )

                return response.content

            except Exception as e:
                last_error = e
                wrapped_error = self._wrap_llm_error(e)

                # Log the error
                logger.warning(
                    f"LLM invocation failed (attempt {attempt + 1}/{self.retry_config.max_retries + 1}): {e}"
                )

                # Check if we should retry
                if not is_retryable(wrapped_error):
                    logger.error(f"Non-retryable error, raising immediately: {e}")
                    raise wrapped_error from e

                # Don't sleep after the last attempt
                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.get_delay(attempt)

                    # Check for rate limit header
                    if isinstance(wrapped_error, LLMRateLimitError) and wrapped_error.retry_after:
                        delay = max(delay, wrapped_error.retry_after)

                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)

        # All retries exhausted
        logger.error(f"All {self.retry_config.max_retries + 1} attempts failed")
        raise self._wrap_llm_error(last_error) from last_error

    def _wrap_llm_error(self, error: Exception) -> Exception:
        """
        Wrap a raw exception in an appropriate Q-Ravens exception.

        Args:
            error: The original exception

        Returns:
            A wrapped Q-Ravens exception
        """
        error_msg = str(error).lower()
        error_type = type(error).__name__.lower()

        # Rate limit errors
        if "rate" in error_msg or "429" in error_msg or "rate_limit" in error_type:
            retry_after = None
            # Try to extract retry-after from error
            if hasattr(error, "response") and hasattr(error.response, "headers"):
                retry_after_str = error.response.headers.get("retry-after")
                if retry_after_str:
                    try:
                        retry_after = float(retry_after_str)
                    except ValueError:
                        pass
            return RetryableRateLimitError(
                f"Rate limit exceeded: {error}",
                retry_after=retry_after,
                details={"original_error": str(error)},
            )

        # Timeout errors
        if "timeout" in error_msg or "timeout" in error_type:
            return RetryableTimeoutError(
                f"Request timed out: {error}",
                details={"original_error": str(error)},
            )

        # Connection errors
        if any(term in error_msg or term in error_type for term in [
            "connection", "connect", "network", "unreachable", "503", "502"
        ]):
            return RetryableConnectionError(
                f"Connection failed: {error}",
                details={"original_error": str(error)},
            )

        # Default to response error (non-retryable)
        return LLMResponseError(
            f"LLM error: {error}",
            details={"original_error": str(error)},
        )

    @classmethod
    def from_state(cls, state: dict) -> "BaseAgent":
        """
        Create an agent instance with LLM configuration from state.

        Args:
            state: The workflow state containing LLM configuration

        Returns:
            An agent instance configured with the state's LLM settings
        """
        provider = None
        model = None
        api_key = None

        # Extract LLM config from state if present
        if state.get("llm_provider"):
            try:
                provider = LLMProvider(state["llm_provider"])
            except ValueError:
                logger.warning(f"Invalid LLM provider in state: {state['llm_provider']}")

        if state.get("llm_model"):
            model = state["llm_model"]

        if state.get("llm_api_key"):
            api_key = state["llm_api_key"]

        return cls(provider=provider, model=model, api_key=api_key)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, role={self.role})>"
