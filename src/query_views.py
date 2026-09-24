from __future__ import annotations

import re


# ==========================================================
# BASIC TEXT CLEANING
# ==========================================================

def normalize_text(text: str) -> str:
    """
    Normalize repeated whitespace.
    """

    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    return text


def find_section(
    text: str,
    start_name: str,
    end_name: str | None = None,
) -> str:
    """
    Extract text between two section markers.
    """

    start_pattern = (
        rf"-+\s*{re.escape(start_name)}\s*-+"
    )

    start_match = re.search(
        start_pattern,
        text,
        flags=re.IGNORECASE,
    )

    if not start_match:
        return ""

    start = start_match.end()

    if end_name is None:
        return normalize_text(
            text[start:]
        )

    end_pattern = (
        rf"-+\s*{re.escape(end_name)}\s*-+"
    )

    end_match = re.search(
        end_pattern,
        text[start:],
        flags=re.IGNORECASE,
    )

    if not end_match:
        return normalize_text(
            text[start:]
        )

    end = start + end_match.start()

    return normalize_text(
        text[start:end]
    )


def split_sentences(text: str) -> list[str]:
    """
    Split ordinary prose into sentences.
    """

    text = normalize_text(text)

    if not text:
        return []

    return re.split(
        r"(?<=[.!?])\s+",
        text,
    )


# ==========================================================
# CORE VIEW
# ==========================================================

def extract_core(text: str) -> str:
    """
    Keep the first two and last two sentences of the
    main problem statement.

    This attempts to preserve the problem context and
    objective while reducing some story/background text.
    """

    input_match = re.search(
        r"-+\s*Input\s*-+",
        text,
        flags=re.IGNORECASE,
    )

    if input_match:
        problem_text = text[
            :input_match.start()
        ]
    else:
        problem_text = text

    sentences = split_sentences(
        problem_text
    )

    if len(sentences) <= 4:
        return normalize_text(
            " ".join(sentences)
        )

    selected = (
        sentences[:2]
        + sentences[-2:]
    )

    return normalize_text(
        " ".join(selected)
    )


# ==========================================================
# INPUT VIEW
# ==========================================================

def extract_input(text: str) -> str:
    """
    Extract only the Input section.
    """

    return find_section(
        text,
        "Input",
        "Output",
    )


# ==========================================================
# OUTPUT VIEW
# ==========================================================

def extract_output(text: str) -> str:
    """
    Extract only the Output section.
    """

    return find_section(
        text,
        "Output",
        "Examples",
    )


# ==========================================================
# RULE HELPERS
# ==========================================================

STRONG_RULE_PATTERNS = [
    r"\bmay not\b",
    r"\bmust not\b",
    r"\bcannot\b",
    r"\bcan't\b",
    r"\bnot allowed\b",
    r"\bonly if\b",
    r"\bnot possible\b",
    r"\bat most\b",
    r"\bat least\b",
    r"\bexactly\b",
    r"\bno more than\b",
    r"\bno less than\b",
    r"\bdoesn'?t exceed\b",
    r"\bdoes not exceed\b",
    r"\bpossibly zero\b",
    r"\bpositive integer\b",
    r"\bnegative integer\b",
]


def extract_rule_fragments_from_problem(
    text: str,
) -> list[str]:
    """
    Extract only small fragments around strong rule phrases
    from the problem statement.

    We deliberately do NOT use broad words such as:
    "given", "minimum", "maximum", "number of", "consists of".

    This prevents the Rules view from copying the whole
    problem statement.
    """

    input_match = re.search(
        r"-+\s*Input\s*-+",
        text,
        flags=re.IGNORECASE,
    )

    if input_match:
        problem_text = text[
            :input_match.start()
        ]
    else:
        problem_text = text

    sentences = split_sentences(
        problem_text
    )

    fragments = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        lower = sentence.lower()

        matches = []

        for pattern in STRONG_RULE_PATTERNS:

            for match in re.finditer(
                pattern,
                lower,
                flags=re.IGNORECASE,
            ):
                matches.append(
                    (
                        match.start(),
                        match.end(),
                    )
                )

        if not matches:
            continue

        # --------------------------------------------------
        # Extract a short piece around each strong rule.
        # --------------------------------------------------

        for start, end in matches:

            fragment_start = max(
                0,
                start - 90,
            )

            fragment_end = min(
                len(sentence),
                end + 180,
            )

            fragment = sentence[
                fragment_start:fragment_end
            ].strip()

            # Avoid tiny fragments.
            if len(fragment) >= 15:

                if len(fragment) > 250:
                    fragment = (
                        fragment[:250]
                        + "..."
                    )

                fragments.append(
                    fragment
                )

    return fragments


def extract_numeric_rules(
    input_text: str,
) -> list[str]:
    """
    Extract only numerical constraints and allowed-character
    information from the Input section.
    """

    sentences = split_sentences(
        input_text
    )

    results = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        lower = sentence.lower()

        # --------------------------------------------------
        # Numerical constraints
        #
        # Examples:
        # 1 <= n <= 10^5
        # 1 <= |s| <= 500000
        # x <= 10^18
        # n > 0
        # --------------------------------------------------

        has_numeric_constraint = bool(
            re.search(
                r"""
                (?:
                    \|?[A-Za-z_][A-Za-z0-9_]*\|?
                    |
                    \|[A-Za-z_][A-Za-z0-9_]*\|
                )
                \s*
                (?:<=|>=|<|>|≤|≥|=)
                \s*
                (?:
                    \d+
                    |
                    10\s*\^?\s*\d+
                )
                """,
                sentence,
                flags=re.IGNORECASE | re.VERBOSE,
            )
        )

        # --------------------------------------------------
        # Range expressions
        # --------------------------------------------------

        has_range = bool(
            re.search(
                r"""
                \d+
                \s*
                (?:to|-)
                \s*
                (?:
                    \d+
                    |
                    10\s*\^?\s*\d+
                )
                """,
                sentence,
                flags=re.IGNORECASE | re.VERBOSE,
            )
        )

        # --------------------------------------------------
        # Allowed input format
        # --------------------------------------------------

        has_allowed_format = bool(
            re.search(
                r"""
                \b(
                    consists\s+of
                    |
                    contains\s+only
                    |
                    may\s+contain
                    |
                    only\s+contains
                )\b
                """,
                lower,
                flags=re.IGNORECASE | re.VERBOSE,
            )
        )

        if (
            has_numeric_constraint
            or has_range
            or has_allowed_format
        ):

            # Keep Input constraint sentences short.
            if len(sentence) > 300:
                sentence = (
                    sentence[:300]
                    + "..."
                )

            results.append(
                sentence
            )

    return results


# ==========================================================
# RULE VIEW
# ==========================================================

def extract_rules(text: str) -> str:
    """
    Build a concise Rules/Constraints view from:

    1. Strong rule fragments in the problem statement.
    2. Numerical/input constraints.

    The goal is NOT to summarize the entire problem.
    """

    problem_rule_fragments = (
        extract_rule_fragments_from_problem(
            text
        )
    )

    input_text = extract_input(
        text
    )

    numeric_rules = extract_numeric_rules(
        input_text
    )

    # Combine.
    all_rules = (
        problem_rule_fragments
        + numeric_rules
    )

    # --------------------------------------------------
    # Remove duplicates.
    # --------------------------------------------------

    seen = set()
    unique_rules = []

    for rule in all_rules:

        rule = normalize_text(
            rule
        )

        key = rule.lower()

        if (
            rule
            and key not in seen
        ):
            seen.add(key)
            unique_rules.append(
                rule
            )

    # --------------------------------------------------
    # Keep the Rules view reasonably small.
    # --------------------------------------------------

    return normalize_text(
        " ".join(unique_rules[:8])
    )


# ==========================================================
# FOUR-VIEW DECOMPOSITION
# ==========================================================

def decompose_query(
    text: str,
) -> dict[str, str]:
    """
    Convert one programming problem into
    four retrieval-focused views.
    """

    return {
        "core": extract_core(text),
        "input": extract_input(text),
        "output": extract_output(text),
        "rules": extract_rules(text),
    }


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    import mteb

    print(
        "Loading AppsRetrieval test data..."
    )

    task = mteb.get_task(
        "AppsRetrieval"
    )

    task.load_data()

    test_data = (
        task.dataset[
            "default"
        ][
            "test"
        ]
    )

    queries = test_data[
        "queries"
    ]

    # --------------------------------------------------
    # Test first 3 real queries.
    # --------------------------------------------------

    for item in queries.select(
        range(3)
    ):

        print()
        print("=" * 70)
        print(
            "QUERY ID:",
            item["id"]
        )
        print("=" * 70)

        views = decompose_query(
            item["text"]
        )

        for name, content in views.items():

            print()
            print(
                f"[{name.upper()}]"
            )

            print(
                content[:1000]
            )

            if len(content) > 1000:
                print("...")