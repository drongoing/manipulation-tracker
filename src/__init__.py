"""
Manipulation Accumulation Tracker

A system for tracking how manipulation accumulates across conversation turns in AI systems.
Built on Anthropic's BLOOM framework for automated behavioral evaluations.
"""

from .models import (
    TurnScore,
    ConversationAnalysis,
    AggregateResults,
    EscalationPattern
)

__version__ = "0.1.0"

__all__ = [
    "TurnScore",
    "ConversationAnalysis",
    "AggregateResults",
    "EscalationPattern"
]
