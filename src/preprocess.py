"""Query and code-snippet preprocessing.

This is the highest-leverage file for improving retrieval accuracy (P0).
The guideline explicitly calls out query categorization/preprocessing,
snippet categorization/pre-post-processing, and multi-pass retrieval as
places to improve on a plain embedding baseline — none of that is done
here yet beyond basic cleanup. Treat the functions below as a starting
point, not a finished pipeline.
"""

import re


def clean_query(text: str) -> str:
    """Normalize a natural-language query before embedding.

    Current behaviour: collapse whitespace only. APPS-style problem
    statements can be long (problem description + input/output spec +
    examples) — for retrieval, the boilerplate sections likely add noise.

    TODO (team): consider stripping/deprioritizing "Input:"/"Output:"/
    "Example:" sections, or extracting just the first paragraph, and
    A/B test NDCG@10 with vs. without on a held-out subset before
    committing to a rule.
    """
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_snippet(code: str) -> str:
    """Normalize a code snippet before embedding.

    Current behaviour: collapse blank lines and trailing whitespace, keep
    comments (comments can carry useful signal for retrieval, unlike for
    code generation). Leaves indentation and identifiers untouched.

    TODO (team): try comment stripping, docstring-only extraction, or
    prepending a short auto-generated one-line description of what the
    snippet does (costs indexing time — measure it) and see which one
    actually moves NDCG@10 rather than assuming.
    """
    if not code:
        return ""
    lines = [line.rstrip() for line in code.splitlines()]
    # collapse runs of blank lines to a single blank line
    cleaned_lines = []
    prev_blank = False
    for line in lines:
        is_blank = len(line) == 0
        if is_blank and prev_blank:
            continue
        cleaned_lines.append(line)
        prev_blank = is_blank
    return "\n".join(cleaned_lines).strip()
