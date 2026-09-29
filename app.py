import json
import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st


# ============================================================
# PRISM GENAI HACKATHON 3.0
# THEME 01 - AGENTIC CODE INTELLIGENCE
# TEAM RINITA
#
# FAST DASHBOARD
# UI loads immediately.
# Retrieval model runs in a background worker.
# ============================================================


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    __file__
).resolve().parent


WORKER_FILE = (
    ROOT
    / "retrieval_worker.py"
)


CACHE_DIR = (
    ROOT
    / "data"
    / ".dashboard_cache"
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


READY_FILE = (
    CACHE_DIR
    / "worker_ready.json"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PRISM Code Retrieval",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:#090d1b;
}

.block-container {
    max-width:1180px;
    padding-top:1.4rem;
    padding-bottom:2rem;
}

#MainMenu {
    visibility:hidden;
}

footer {
    visibility:hidden;
}

section[data-testid="stSidebar"] {
    background:#f1f3f8;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    background:#0d1224;
    border:1px solid #202743;
    border-radius:18px;
    padding:1.5rem 1.8rem;
    margin-bottom:1.2rem;
}

.badge {
    display:inline-block;
    background:#19213b;
    border:1px solid #303958;
    color:#dce2ef;
    border-radius:999px;
    padding:.3rem .75rem;
    font-size:.72rem;
    font-weight:800;
    letter-spacing:.05em;
    margin-bottom:.8rem;
}

.hero-title {
    color:white;
    font-size:2.65rem;
    font-weight:850;
    line-height:1.05;
}

.hero-subtitle {
    color:#aeb7c8;
    font-size:.95rem;
    margin-top:.7rem;
    line-height:1.55;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

.sidebar-title {
    color:#2d3342;
    font-size:1.2rem;
    font-weight:800;
    margin-bottom:1.2rem;
}

.sidebar-heading {
    color:#303747;
    font-size:.75rem;
    font-weight:800;
    letter-spacing:.1em;
    margin-top:1rem;
    margin-bottom:.45rem;
}

.sidebar-text {
    color:#4e5667;
    font-size:.85rem;
    line-height:1.5;
    margin-bottom:.8rem;
}

.weight {
    background:white;
    border:1px solid #dfe3ea;
    border-radius:10px;
    padding:.65rem .75rem;
    margin-bottom:.45rem;
}

.weight-name {
    color:#6d7482;
    font-size:.74rem;
}

.weight-value {
    color:#252b38;
    font-size:1.25rem;
    font-weight:800;
}


/* =========================================================
   SECTION
   ========================================================= */

.section-title {
    color:white;
    font-size:1.1rem;
    font-weight:800;
    margin-top:1.1rem;
    margin-bottom:.3rem;
}

.caption {
    color:#8e97aa;
    font-size:.83rem;
    margin-bottom:.7rem;
}


/* =========================================================
   QUERY
   ========================================================= */

div[data-testid="stTextArea"] textarea {
    background:#f2f4f8 !important;
    color:#252a36 !important;
    border:1px solid #dce1e9 !important;
    border-radius:10px !important;
}


/* =========================================================
   BUTTON
   ========================================================= */

div.stButton > button {
    width:100%;
    background:#ff4b4b;
    color:white;
    border:none;
    border-radius:10px;
    font-weight:800;
    min-height:44px;
}

div.stButton > button:hover {
    background:#ff5b5b;
    color:white;
}


/* =========================================================
   STATUS
   ========================================================= */

.status-ready {
    background:#101d22;
    border:1px solid #28434a;
    border-radius:11px;
    padding:.72rem 1rem;
    color:#b9d7dc;
    font-size:.82rem;
    margin-bottom:.8rem;
}

.status-loading {
    background:#171725;
    border:1px solid #36364c;
    border-radius:11px;
    padding:.72rem 1rem;
    color:#c1c5d3;
    font-size:.82rem;
    margin-bottom:.8rem;
}


/* =========================================================
   METRICS
   ========================================================= */

.metric {
    background:#10162b;
    border:1px solid #242d49;
    border-radius:13px;
    padding:.85rem;
}

.metric-label {
    color:#8992a5;
    font-size:.7rem;
    font-weight:750;
    text-transform:uppercase;
    letter-spacing:.08em;
}

.metric-value {
    color:white;
    font-size:1.4rem;
    font-weight:850;
    margin-top:.2rem;
}


/* =========================================================
   QUERY VIEWS
   ========================================================= */

.view {
    background:#10162b;
    border:1px solid #242d49;
    border-radius:13px;
    padding:.9rem;
    min-height:120px;
}

.view-name {
    color:white;
    font-weight:800;
    font-size:.78rem;
    letter-spacing:.08em;
}

.view-weight {
    float:right;
    color:#ff6b6b;
    font-weight:850;
}

.view-content {
    color:#abb4c4;
    font-size:.8rem;
    line-height:1.5;
    margin-top:.5rem;
}


/* =========================================================
   RESULTS
   ========================================================= */

.result {
    background:#10162b;
    border:1px solid #242d49;
    border-radius:13px;
    padding:.9rem;
    margin-top:.8rem;
}

.rank {
    display:inline-block;
    background:#1c2748;
    color:#dce4fa;
    border-radius:999px;
    padding:.22rem .52rem;
    font-size:.68rem;
    font-weight:800;
}

.result-name {
    color:white;
    font-weight:800;
    margin-top:.45rem;
}

.result-score {
    color:#ff6c6c;
    font-size:.76rem;
    font-weight:750;
    margin-top:.15rem;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    border-top:1px solid #202743;
    margin-top:2rem;
    padding-top:.9rem;
    text-align:center;
    color:#7d8799;
    font-size:.74rem;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">

<div class="badge">
SAMSUNG PRISM • THEME 01 • AGENTIC CODE INTELLIGENCE
</div>

<div class="hero-title">
🔎 PRISM Code Retrieval
</div>

<div class="hero-subtitle">
<strong>Problem-Focused Multi-View Retrieval</strong><br>
Natural-language programming query → focused evidence → ranked code snippets
</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">Retrieval Configuration</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-heading">QUERY VIEWS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="sidebar-text">
<strong>CORE</strong><br>
Main programming task
</div>

<div class="sidebar-text">
<strong>INPUT</strong><br>
Input requirements and signals
</div>

<div class="sidebar-text">
<strong>OUTPUT</strong><br>
Expected result or behavior
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        '<div class="sidebar-heading">FROZEN WEIGHTS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="weight">
<div class="weight-name">Core</div>
<div class="weight-value">50%</div>
</div>

<div class="weight">
<div class="weight-name">Input</div>
<div class="weight-value">25%</div>
</div>

<div class="weight">
<div class="weight-name">Output</div>
<div class="weight-value">25%</div>
</div>

<div class="weight">
<div class="weight-name">Model</div>
<div class="weight-value" style="font-size:.92rem;">
MiniLM + ONNX
</div>
</div>

<div class="weight">
<div class="weight-name">Dimension</div>
<div class="weight-value">384</div>
</div>

<div class="weight">
<div class="weight-name">Benchmark</div>
<div class="weight-value" style="font-size:.92rem;">
MTEB AppsRetrieval
</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# BACKGROUND WORKER
# ============================================================

@st.cache_resource
def start_worker():

    # --------------------------------------------------------
    # If worker script does not exist
    # --------------------------------------------------------

    if not WORKER_FILE.exists():

        return None


    # --------------------------------------------------------
    # Windows
    # --------------------------------------------------------

    creation_flags = 0

    if os.name == "nt":

        creation_flags = (
            subprocess.CREATE_NO_WINDOW
        )


    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    environment = os.environ.copy()

    environment[
        "TOKENIZERS_PARALLELISM"
    ] = "false"

    environment[
        "HF_HUB_DISABLE_PROGRESS_BARS"
    ] = "1"

    environment[
        "TRANSFORMERS_VERBOSITY"
    ] = "error"


    # --------------------------------------------------------
    # Start background process
    # --------------------------------------------------------

    process = subprocess.Popen(
        [
            sys.executable,
            str(WORKER_FILE)
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        bufsize=1,
        cwd=str(ROOT),
        creationflags=creation_flags,
    )


    return process


# ============================================================
# START WORKER WITHOUT BLOCKING UI
# ============================================================

worker = start_worker()


# ============================================================
# ENGINE STATUS
# ============================================================

def get_engine_status():

    if not READY_FILE.exists():

        return False, 0


    try:

        with READY_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)


        if data.get("ready") is True:

            return (
                True,
                int(
                    data.get(
                        "corpus_size",
                        0
                    )
                )
            )

    except Exception:

        pass


    return False, 0


engine_ready, corpus_size = (
    get_engine_status()
)


# ============================================================
# STATUS DISPLAY
# ============================================================

if engine_ready:

    st.markdown(
        f"""
<div class="status-ready">
<strong style="color:white;">
● Retrieval Engine READY
</strong>
&nbsp;&nbsp;|&nbsp;&nbsp;
ONNX AVX2
&nbsp;&nbsp;|&nbsp;&nbsp;
Corpus: {corpus_size}
&nbsp;&nbsp;|&nbsp;&nbsp;
384-D
</div>
""",
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
<div class="status-loading">
<strong style="color:white;">
● Retrieval Engine warming up
</strong>
&nbsp;&nbsp;|&nbsp;&nbsp;
Dashboard is already ready
&nbsp;&nbsp;|&nbsp;&nbsp;
The ML worker is loading in the background.
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# QUERY
# ============================================================

st.markdown(
    '<div class="section-title">Programming Query</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="caption">'
    'Enter a natural-language programming problem.'
    '</div>',
    unsafe_allow_html=True,
)


query = st.text_area(
    "Programming Query",
    value="How is the input validated?",
    height=105,
    label_visibility="collapsed",
)


clicked = st.button(
    "🔎 Search Code",
    use_container_width=True,
)


# ============================================================
# SEND QUERY TO WORKER
# ============================================================

def worker_search(
    process,
    query
):

    if process is None:

        return {
            "error":
            "Retrieval worker could not start."
        }


    if process.poll() is not None:

        return {
            "error":
            "Retrieval worker stopped."
        }


    request = {
        "command": "search",
        "query": query
    }


    try:

        process.stdin.write(
            json.dumps(
                request
            )
            + "\n"
        )

        process.stdin.flush()


        response_line = (
            process.stdout.readline()
        )


        if not response_line:

            return {
                "error":
                "No response from retrieval worker."
            }


        return json.loads(
            response_line
        )


    except Exception as exc:

        return {
            "error": str(exc)
        }


# ============================================================
# SEARCH
# ============================================================

if clicked:

    query = query.strip()


    if not query:

        st.warning(
            "Please enter a programming query."
        )


    elif not engine_ready:

        st.warning(
            "Retrieval engine is still warming up. "
            "Please wait a few seconds and click Search Code again."
        )


    else:

        result = worker_search(
            worker,
            query
        )


        if "error" in result:

            st.error(
                result["error"]
            )


        else:

            st.session_state[
                "results"
            ] = result["results"]

            st.session_state[
                "views"
            ] = result["views"]

            st.session_state[
                "latency"
            ] = result["latency_ms"]

            st.session_state[
                "query"
            ] = query


# ============================================================
# SHOW RESULTS
# ============================================================

if "results" in st.session_state:

    results = st.session_state[
        "results"
    ]

    views = st.session_state[
        "views"
    ]

    latency = st.session_state[
        "latency"
    ]

    query = st.session_state[
        "query"
    ]


    # ========================================================
    # SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">Retrieval Summary</div>',
        unsafe_allow_html=True,
    )


    c1, c2, c3, c4 = (
        st.columns(4)
    )


    with c1:

        st.markdown(
            f"""
<div class="metric">
<div class="metric-label">Corpus</div>
<div class="metric-value">{corpus_size}</div>
</div>
""",
            unsafe_allow_html=True,
        )


    with c2:

        st.markdown(
            f"""
<div class="metric">
<div class="metric-label">Top-K</div>
<div class="metric-value">{len(results)}</div>
</div>
""",
            unsafe_allow_html=True,
        )


    with c3:

        st.markdown(
            f"""
<div class="metric">
<div class="metric-label">Retrieval</div>
<div class="metric-value">{latency:.2f} ms</div>
</div>
""",
            unsafe_allow_html=True,
        )


    with c4:

        st.markdown(
            """
<div class="metric">
<div class="metric-label">Embedding</div>
<div class="metric-value">384-D</div>
</div>
""",
            unsafe_allow_html=True,
        )


    # ========================================================
    # QUERY VIEWS
    # ========================================================

    st.markdown(
        '<div class="section-title">Query Views</div>',
        unsafe_allow_html=True,
    )


    v1, v2, v3 = (
        st.columns(3)
    )


    with v1:

        st.markdown(
            f"""
<div class="view">

<span class="view-name">
CORE
</span>

<span class="view-weight">
50%
</span>

<div class="view-content">
{views["core"]}
</div>

</div>
""",
            unsafe_allow_html=True,
        )


    with v2:

        st.markdown(
            f"""
<div class="view">

<span class="view-name">
INPUT
</span>

<span class="view-weight">
25%
</span>

<div class="view-content">
{views["input"]}
</div>

</div>
""",
            unsafe_allow_html=True,
        )


    with v3:

        st.markdown(
            f"""
<div class="view">

<span class="view-name">
OUTPUT
</span>

<span class="view-weight">
25%
</span>

<div class="view-content">
{views["output"]}
</div>

</div>
""",
            unsafe_allow_html=True,
        )


    # ========================================================
    # RESULTS
    # ========================================================

    st.markdown(
        '<div class="section-title">Ranked Code Results</div>',
        unsafe_allow_html=True,
    )


    st.markdown(
        f"""
<div style="
background:#0d1224;
border:1px solid #202743;
border-radius:10px;
padding:.75rem 1rem;
color:#aeb6c6;
margin-bottom:.6rem;
">

<strong style="color:white;">
Query:
</strong>

{query}

</div>
""",
        unsafe_allow_html=True,
    )


    for result in results:

        st.markdown(
            f"""
<div class="result">

<span class="rank">
RANK #{result["rank"]}
</span>

<div class="result-name">
{result["name"]}
</div>

<div class="result-score">
Similarity: {result["score"]:.4f}
</div>

</div>
""",
            unsafe_allow_html=True,
        )


        st.code(
            result["code"],
            language="python"
        )


# ============================================================
# INITIAL STATE
# ============================================================

else:

    st.markdown(
        """
<div style="
background:#10162b;
border:1px solid #242d49;
border-radius:13px;
padding:1rem;
margin-top:1rem;
">

<div style="
color:white;
font-weight:800;
margin-bottom:.3rem;
">
Ready for retrieval
</div>

<div style="
color:#949daf;
font-size:.82rem;
line-height:1.5;
">

The dashboard loads independently from the ML engine.
The ONNX retrieval worker runs in the background, so the
web interface does not wait for model initialization.

</div>

</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">

<strong style="color:#cbd3e3;">
Team Rinita
</strong>

&nbsp;•&nbsp;

PRISM GenAI Hackathon 3.0

&nbsp;•&nbsp;

Theme 01 — Agentic Code Intelligence

&nbsp;•&nbsp;

Problem-Focused Multi-View Code Retrieval

</div>
""",
    unsafe_allow_html=True,
)