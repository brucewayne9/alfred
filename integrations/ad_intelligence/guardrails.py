"""Confirmation guardrails for ad financial mutations.

All tools that change budgets, bids, or campaign/ad-set/ad status must include
a `confirmed` boolean parameter. If confirmed=False (default), the tool returns
a preview of the change with a prompt for Mike to approve. Only when confirmed=True
does the actual mutation execute.
"""


def guardrail_response(action_description: str, details: dict) -> dict:
    """Return a confirmation prompt instead of executing the mutation."""
    return {
        "status": "awaiting_confirmation",
        "action": action_description,
        "details": details,
        "message": f"I'd like to {action_description}. Here are the details — shall I proceed?",
        "confirm_instruction": "Call this tool again with confirmed=True to execute.",
    }
