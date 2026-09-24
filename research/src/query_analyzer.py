from __future__ import annotations

import re


# Words that usually indicate a question about
# relationships between code elements.
STRUCTURAL_WORDS = {
    "calls",
    "called",
    "invokes",
    "invoked",
    "depends",
    "dependency",
    "before",
    "after",
    "precedes",
    "follows",
    "relationship",
    "chain",
}


# Phrases that usually indicate a question
# about a specific code symbol/name.
IDENTIFIER_PHRASES = {
    "defined",
    "definition",
    "used",
    "usage",
    "reference",
    "references",
}


def looks_like_identifier(text: str) -> bool:
    """
    Detect code-like names such as:

    validate_input
    getUserProfile
    AuthManager
    BLUETOOTH_SETTINGS
    """

    return bool(
        re.search(
            r"\b[a-zA-Z_][a-zA-Z0-9_]*(?:_[a-zA-Z0-9_]+)+\b",
            text,
        )
    )


def classify_query(query: str) -> tuple[str, str]:
    """
    Classify a developer query into:

    SEMANTIC
    IDENTIFIER
    STRUCTURAL
    """

    q = query.lower().strip()

    # -----------------------------------------
    # 1. Check for structural questions first
    # -----------------------------------------

    structural_matches = [
        word
        for word in STRUCTURAL_WORDS
        if re.search(r"\b" + re.escape(word) + r"\b", q)
    ]

    if structural_matches:
        reason = (
            "Structural clue detected: "
            + ", ".join(structural_matches)
        )
        return "STRUCTURAL", reason

    # -----------------------------------------
    # 2. Check for identifier/symbol questions
    # -----------------------------------------

    identifier_phrase_found = any(
        phrase in q
        for phrase in IDENTIFIER_PHRASES
    )

    code_name_found = looks_like_identifier(query)

    if identifier_phrase_found or code_name_found:

        reasons = []

        if identifier_phrase_found:
            reasons.append("identifier-related wording")

        if code_name_found:
            reasons.append("code-like identifier detected")

        return "IDENTIFIER", "; ".join(reasons)

    # -----------------------------------------
    # 3. Otherwise treat it as semantic
    # -----------------------------------------

    return (
        "SEMANTIC",
        "No strong structural or identifier clue detected",
    )


if __name__ == "__main__":

    test_queries = [
        "How is the input validated?",
        "Which function converts spaces into underscores?",
        "Where is validate_input defined?",
        "Where is BLUETOOTH_SETTINGS used?",
        "Which function calls authenticateUser before processing?",
    ]

    print("Query Analyzer Test")
    print("=" * 60)

    for query in test_queries:

        category, reason = classify_query(query)

        print()
        print("Query:", query)
        print("Category:", category)
        print("Reason:", reason)