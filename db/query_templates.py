"""Allowlisted SQL templates — no raw LLM SQL."""

QUERY_TEMPLATES: dict[str, str] = {
    "get_example_row": "SELECT 1 AS id, 'ok' AS status LIMIT 1",
    # Add real templates here; use $1, $2 for parameters
    "get_user_by_email": "SELECT id, email FROM users WHERE email = $1 LIMIT 100",
}

MAX_ROWS = 100
