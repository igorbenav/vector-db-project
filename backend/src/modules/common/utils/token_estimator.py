"""Token estimation utilities for AI models.

This module provides improved token estimation for different AI models without
requiring external tokenization libraries. The estimates are more accurate than
simple word counting while remaining lightweight.
"""

import re
from typing import Any, Dict


class TokenEstimator:
    """Improved token estimation for various AI models.

    Provides more accurate token estimates than simple word counting by considering:
    - Subword tokenization patterns
    - Punctuation and special characters
    - Model-specific tokenization characteristics
    - Code and technical content patterns
    """

    MODEL_MULTIPLIERS = {
        "openai": 1.3,
        "gpt": 1.3,
        "anthropic": 1.25,
        "claude": 1.25,
        "google": 1.35,
        "gemini": 1.35,
        "default": 1.3,
    }

    @staticmethod
    def estimate_tokens(text: str, model: str = "default") -> int:
        """Estimate token count for given text and model.

        Uses improved heuristics that account for:
        - Word boundaries and subword patterns
        - Punctuation and special characters
        - Code patterns and technical content
        - Model-specific tokenization differences

        Args:
            text: Input text to estimate tokens for
            model: Model name/provider for model-specific estimation

        Returns:
            Estimated token count as integer

        Example:
            >>> estimator = TokenEstimator()
            >>> tokens = estimator.estimate_tokens("Hello world!", "openai")
            >>> print(f"Estimated tokens: {tokens}")
        """
        if not text or not text.strip():
            return 0

        multiplier = TokenEstimator._get_model_multiplier(model)

        words = TokenEstimator._count_words(text)

        base_estimate = words * multiplier

        adjustments = (
            TokenEstimator._punctuation_adjustment(text)
            + TokenEstimator._code_adjustment(text)
            + TokenEstimator._special_chars_adjustment(text)
            + TokenEstimator._length_adjustment(text)
        )

        final_estimate = max(1, int(base_estimate + adjustments))

        return final_estimate

    @staticmethod
    def _get_model_multiplier(model: str) -> float:
        """Get model-specific multiplier for token estimation."""
        model_lower = model.lower()

        for key, multiplier in TokenEstimator.MODEL_MULTIPLIERS.items():
            if key in model_lower:
                return multiplier

        return TokenEstimator.MODEL_MULTIPLIERS["default"]

    @staticmethod
    def _count_words(text: str) -> int:
        """Count words using improved tokenization."""
        words = [word for word in text.split() if word.strip()]
        return len(words)

    @staticmethod
    def _punctuation_adjustment(text: str) -> float:
        """Adjust for punctuation that often becomes separate tokens."""
        punct_chars = re.findall(r'[.!?,:;(){}[\]"\'-]', text)
        return len(punct_chars) * 0.3

    @staticmethod
    def _code_adjustment(text: str) -> float:
        """Adjust for code patterns that increase token density."""
        adjustments = 0.0

        code_blocks = re.findall(r"```.*?```", text, re.DOTALL)
        adjustments += len(code_blocks) * 5

        inline_code = re.findall(r"`[^`]+`", text)
        adjustments += len(inline_code) * 2

        urls = re.findall(r"https?://\S+|/\S+|\S+\.\S+", text)
        adjustments += len(urls) * 3

        prog_patterns = re.findall(r"[=<>!&|+\-*/%^~]|def |class |import |from ", text)
        adjustments += len(prog_patterns) * 0.5

        return adjustments

    @staticmethod
    def _special_chars_adjustment(text: str) -> float:
        """Adjust for special characters and Unicode."""
        adjustments = 0.0

        numbers = re.findall(r"\d+", text)
        adjustments += len(numbers) * 0.2

        special_chars = re.findall(r'[^\w\s.,!?;:\'"()-]', text)
        adjustments += len(special_chars) * 0.5

        newlines = text.count("\n")
        adjustments += newlines * 0.5

        return adjustments

    @staticmethod
    def _length_adjustment(text: str) -> float:
        """Adjust based on text length characteristics."""
        text_length = len(text)

        if text_length > 5000:
            return text_length * 0.0001
        elif text_length > 1000:
            return text_length * 0.00005

        return 0.0

    @staticmethod
    def estimate_conversation_tokens(messages: list, model: str = "default") -> Dict[str, Any]:
        """Estimate tokens for a full conversation.

        Args:
            messages: List of message dictionaries with 'content' and 'role' fields
            model: Model name for estimation

        Returns:
            Dictionary with token breakdown:
            - total_tokens: Total estimated tokens
            - user_tokens: Tokens from user messages
            - assistant_tokens: Tokens from assistant messages
            - system_tokens: Tokens from system messages
            - message_count: Number of messages
        """
        breakdown = {
            "total_tokens": 0,
            "user_tokens": 0,
            "assistant_tokens": 0,
            "system_tokens": 0,
            "message_count": len(messages),
        }

        for message in messages:
            content = message.get("content", "")
            role = message.get("role", "user")

            tokens = TokenEstimator.estimate_tokens(content, model)
            breakdown["total_tokens"] += tokens

            if role == "user":
                breakdown["user_tokens"] += tokens
            elif role == "assistant":
                breakdown["assistant_tokens"] += tokens
            elif role == "system":
                breakdown["system_tokens"] += tokens

        conversation_overhead = len(messages) * 3
        breakdown["total_tokens"] += conversation_overhead

        return breakdown


def estimate_message_tokens(content: str, model: str = "default") -> int:
    """Convenience function for estimating tokens in a single message.

    Args:
        content: Message content to estimate
        model: Model name for estimation

    Returns:
        Estimated token count
    """
    return TokenEstimator.estimate_tokens(content, model)
