"""
Streamlit PDF → Quiz Generator & Grader (Gemini-powered)
=========================================================

Single-file Streamlit app that:
- Ingests a PDF and auto-generates a quiz with Google Gemini (MCQ/TF/Short Answer)
- Lets users take the quiz with a live countdown timer
- Grades automatically, shows score, itemized feedback, and explanations
- Supports review mode, answer reveal, and per-question flags
- Exports quiz and results to CSV/JSON
- Supports regeneration with different settings (difficulty, #questions, format mix)
- Offline fallback generator (simple heuristic) if no API key is provided

Deployment targets: Streamlit Community Cloud / local

Requirements (add to requirements.txt if you split files):
---------------------------------------------------------
streamlit==1.36.0
PyPDF2==3.0.1
google-generativeai==0.7.2
pandas==2.2.2
python-dateutil==2.9.0.post0

Environment / Secrets:
----------------------
- Set `GEMINI_API_KEY` as an environment variable or Streamlit secret.
  In Streamlit Cloud, add it under `Secrets` as:
  [secrets]
  GEMINI_API_KEY = "your-key"

Run:
-----
streamlit run app.py

"""

from __future__ import annotations
import os
import json
import time
import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
from PyPDF2 import PdfReader

# Optional: Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# ------------------------------
# Utilities
# ------------------------------

def get_api_key() -> str | None:
    # Priority: st.secrets → env var
    key = None
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
        key = st.secrets["GEMINI_API_KEY"]
    if not key:
        key = os.getenv("GEMINI_API_KEY")
    return key


def extract_text_from_pdf(file) -> str:
    """Extract text from an uploaded PDF (BytesIO) with PyPDF2."""
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n".join(pages)
    # Clean minimal
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text[:120_000]  # safety cap


def chunk_text(text: str, max_chars: int = 9000) -> List[str]:
    """Chunk text into pieces under max_chars to keep prompts sane."""
    chunks = []
    buf = []
    total = 0
    for line in text.split("\n"):
        if total + len(line) + 1 > max_chars and buf:
            chunks.append("\n".join(buf))
            buf, total = [], 0
        buf.append(line)
        total += len(line) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks or [text]


@dataclass
class Question:
    id: str
    stem: str
    type: str  # 'mcq' | 'tf' | 'short'
    options: List[str]
    answer: Any  # index for mcq, 'True'/'False' for tf, string for short
    explanation: str | None
    points: float = 1.0


# ------------------------------
# Gemini prompt & generation
# ------------------------------

SYSTEM_INSTRUCTIONS = (
    "You are an expert assessment designer. Generate fair, rigorous quizzes from source text."
)

GENERATION_GUIDE = (
    """
Return JSON only, following this schema:
{
  "questions": [
    {
      "stem": "<clear question>",
      "type": "mcq" | "tf" | "short",
      "options": ["A", "B", "C", "D"],   // for tf use ["True","False"], for short can be []
      "answer": 0 or 1 or 2 or 3 | "True" | "False" | "<short exact>",
      "explanation": "<1-3 lines explaining the correct answer>",
      "points": 1
    }, ...
  ]
}
Rules:
- Prefer MCQ unless user requests otherwise; include some TF and Short Answer if mix is enabled.
- Ensure options are unambiguous, plausible, and mutually exclusive.
- Answers must be recoverable from the provided text; avoid outside facts.
- Keep stems concise; avoid negatives like "NOT" unless emphasized.
- Keep short-answer keys specific (few words), not full sentences.
- Avoid duplicates; diversify coverage across the text.
"""
)


def gemini_generate_questions(text: str, num_q: int, difficulty: str, allow_types: List[str], mix_types: bool) -> List[Question]:
    api_key = get_api_key()
    if not (GEMINI_AVAILABLE and api_key):
        return heuristic_generate_questions(text, num_q, allow_types, mix_types)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Build a short, focused prompt (we chunk if needed)
    chunks = chunk_text(text, max_chars=9000)
    prompt_header = (
        f"Create {num_q} quiz questions from the text. Difficulty: {difficulty}. "
        f"Allowed types: {', '.join(allow_types)}. Mix types: {mix_types}. {GENERATION_GUIDE}"
    )

    collected: List[Question] = []

    for idx, chunk in enumerate(chunks):
        needed = max(0, num_q - len(collected))
        if needed == 0:
            break
        prompt = f"{SYSTEM_INSTRUCTIONS}\n\n{prompt_header}\n\nSOURCE TEXT (part {idx+1}/{len(chunks)}):\n" + chunk
        try:
            resp = model.generate_content(prompt)
            raw = resp.text
            # Extract JSON portion if content includes extra text
            json_str = raw
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = raw[start:end+1]
            data = json.loads(json_str)
            for q in data.get("questions", [])[:needed]:
                qtype = q.get("type", "mcq").lower()
                if qtype not in allow_types:
                    continue
                question = Question(
                    id=str(uuid.uuid4()),
                    stem=q.get("stem", "").strip(),
                    type=qtype,
                    options=list(q.get("options", []) or ([] if qtype=="short" else ["True","False"])),
                    answer=q.get("answer"),
                    explanation=(q.get("explanation") or "").strip() or None,
                    points=float(q.get("points", 1)),
                )
                # Basic sanity checks
                if not question.stem:
                    continue
                if question.type == "mcq" and (not question.options or not isinstance(question.answer, int)):
                    continue
                if question.type == "tf" and str(question.answer) not in {"True", "False"}:
                    continue
                if question.type == "short" and not str(question.answer).strip():
                    continue
                collected.append(question)
        except Exception:
            continue

    if len(collected) < num_q:
        # Pad with heuristic generator
        leftover = num_q - len(collected)
        collected.extend(heuristic_generate_questions(text, leftover, allow_types, mix_types))

    return collected[:num_q]


# ------------------------------
# Heuristic (offline) generation fallback
# ------------------------------

def heuristic_generate_questions(text: str, num_q: int, allow_types: List[str], mix_types: bool) -> List[Question]:
    """Very simple keyword-based generator as a fallback when no API key."""
    sentences = [s.strip() for s in text.replace("\n", " ").split('.') if s.strip()]
    questions: List[Question] = []

    def make_mcq(sentence: str) -> Question:
        words = [w for w in sentence.split() if w.istitle() or len(w) > 6]
        key = (words[0] if words else sentence.split()[0]) if sentence.split() else "Answer"
        opts = [str(key), "None of the above", "All of the above", "Not specified"]
        return Question(
            id=str(uuid.uuid4()),
            stem=f"According to the text, which best fits: '{sentence[:80]}…'?",
            type="mcq",
            options=opts,
            answer=0,
            explanation="Derived from the provided sentence.",
            points=1.0,
        )

    def make_tf(sentence: str) -> Question:
        return Question(
            id=str(uuid.uuid4()),
            stem=f"True or False: {sentence[:100]}.",
            type="tf",
            options=["True","False"],
            answer="True",
            explanation="Heuristic guess based on included sentence.",
            points=1.0,
        )

    def make_short(sentence: str) -> Question:
        token = sentence.split()[0] if sentence.split() else "N/A"
        return Question(
            id=str(uuid.uuid4()),
            stem=f"Fill in the key term from this line: {sentence[:80]}…",
            type="short",
            options=[],
            answer=token,
            explanation="Exact match expected (case-insensitive).",
            points=1.0,
        )

    factories = []
    if mix_types:
        for t in ["mcq","tf","short"]:
            if t in allow_types:
                factories.append(t)
    else:
        factories = [allow_types[0]] if allow_types else ["mcq"]

    makers = {
        "mcq": make_mcq,
        "tf": make_tf,
        "short": make_short,
    }

    i = 0
    while len(questions) < num_q and i < max(10*num_q, 300):
        if not sentences:
            # safety placeholder
            sentences = ["This document discusses key concepts and definitions relevant to the topic."]
        s = sentences[i % len(sentences)]
        t = factories[len(questions) % len(factories)]
        q = makers[t](s)
        questions.append(q)
        i += 1
    return questions[:num_q]


# ------------------------------
# Grading
# ------------------------------

def normalize_short_answer(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def grade_response(q: Question, user_answer: Any) -> tuple[bool, float]:
    if q.type == "mcq":
        correct = (user_answer == q.answer)
        return correct, q.points if correct else 0.0
    elif q.type == "tf":
        correct = str(user_answer) == str(q.answer)
        return correct, q.points if correct else 0.0
    else:  # short
        correct = normalize_short_answer(str(user_answer)) == normalize_short_answer(str(q.answer))
        return correct, q.points if correct else 0.0


# ------------------------------
# Streamlit App
# ------------------------------

def reset_quiz_state():
    for k in [
        "questions", "responses", "graded", "score", "max_score",
        "start_time", "end_time", "time_limit", "flags", "show_answers",
        "quiz_id", "result_rows"
    ]:
        if k in st.session_state:
            del st.session_state[k]


def ensure_state_defaults():
    st.session_state.setdefault("questions", [])
    st.session_state.setdefault("responses", {})
    st.session_state.setdefault("graded", False)
    st.session_state.setdefault("score", 0.0)
    st.session_state.setdefault("max_score", 0.0)
    st.session_state.setdefault("flags", set())
    st.session_state.setdefault("show_answers", False)
    st.session_state.setdefault("quiz_id", str(uuid.uuid4()))
    st.session_state.setdefault("result_rows", [])


def time_left() -> int:
    if "end_time" not in st.session_state:
        return 0
    return max(0, int(st.session_state["end_time"] - time.time()))


def start_timer(minutes: int):
    st.session_state["time_limit"] = minutes
    st.session_state["start_time"] = time.time()
    st.session_state["end_time"] = st.session_state["start_time"] + minutes*60


def render_header():
    st.title("📄➡️🧠 PDF → Quiz (Gemini)")
    st.caption("Upload a PDF, auto-generate a quiz, take it with a timer, then get instant grading & feedback.")


def sidebar_controls():
    with st.sidebar:
        st.header("⚙️ Settings")
        allow_types = st.multiselect(
            "Question types",
            ["mcq","tf","short"],
            default=["mcq","tf","short"],
            help="Select which types the generator may use."
        )
        mix_types = st.toggle("Mix types", value=True)
        num_q = st.slider("Number of questions", 3, 30, 10)
        difficulty = st.selectbox("Difficulty", ["easy","medium","hard"], index=1)
        time_mins = st.slider("Time limit (minutes)", 1, 120, 10)
        st.divider()
        st.markdown("**Generation**")
        regen = st.button("🔁 Regenerate Quiz", use_container_width=True)
        st.markdown(":grey[Regeneration requires an uploaded PDF or previous text.]")

    return {
        "allow_types": allow_types or ["mcq"],
        "mix_types": mix_types,
        "num_q": num_q,
        "difficulty": difficulty,
        "time_mins": time_mins,
        "regen": regen,
    }


def render_timer():
    if "end_time" in st.session_state:
        left = time_left()
        mins, secs = divmod(left, 60)
        st.progress(1 - (left / (st.session_state["time_limit"]*60)), text=f"⏳ Time left: {mins:02d}:{secs:02d}")
        if left == 0 and not st.session_state.get("graded", False):
            st.warning("Time is up! Submitting automatically…")
            grade_quiz()
            st.experimental_rerun()


def create_quiz_from_text(text: str, opts: Dict[str, Any]):
    questions = gemini_generate_questions(
        text=text,
        num_q=opts["num_q"],
        difficulty=opts["difficulty"],
        allow_types=opts["allow_types"],
        mix_types=opts["mix_types"],
    )
    st.session_state["questions"] = questions
    st.session_state["responses"] = {}
    st.session_state["graded"] = False
    st.session_state["score"] = 0.0
    st.session_state["max_score"] = sum(q.points for q in questions)
    st.session_state["quiz_id"] = str(uuid.uuid4())


def grade_quiz():
    rows = []
    total = 0.0
    for q in st.session_state["questions"]:
        user_answer = st.session_state["responses"].get(q.id)
        correct, pts = grade_response(q, user_answer)
        total += pts
        rows.append({
            "quiz_id": st.session_state["quiz_id"],
            "question_id": q.id,
            "type": q.type,
            "stem": q.stem,
            "user_answer": user_answer,
            "correct_answer": q.answer,
            "correct": correct,
            "points": q.points,
            "earned": pts,
            "explanation": q.explanation or "",
        })
    st.session_state["score"] = total
    st.session_state["graded"] = True
    st.session_state["result_rows"] = rows


def export_results():
    df = pd.DataFrame(st.session_state.get("result_rows", []))
    if df.empty:
        st.info("No results to export yet.")
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download results (CSV)", csv, file_name=f"quiz_results_{st.session_state['quiz_id']}.csv", mime="text/csv")
    st.download_button("⬇️ Download results (JSON)", df.to_json(orient="records").encode("utf-8"), file_name=f"quiz_results_{st.session_state['quiz_id']}.json", mime="application/json")


def export_quiz():
    qs = [asdict(q) for q in st.session_state.get("questions", [])]
    if not qs:
        st.info("No quiz to export yet.")
        return
    data = json.dumps(qs, indent=2).encode("utf-8")
    st.download_button("⬇️ Download quiz (JSON)", data, file_name=f"quiz_{st.session_state['quiz_id']}.json", mime="application/json")


def render_question(q: Question):
    st.markdown(f"**Q{list_id(q.id)}. {q.stem}**")

    key = f"resp_{q.id}"
    default = st.session_state["responses"].get(q.id)

    if q.type == "mcq":
        idx = st.radio("Select one:", options=list(range(len(q.options))), format_func=lambda i: q.options[i], key=key, index=default if isinstance(default, int) else None)
        st.session_state["responses"][q.id] = idx
    elif q.type == "tf":
        idx = st.radio("Select:", options=["True","False"], key=key, index=["True","False"].index(default) if default in ["True","False"] else None)
        st.session_state["responses"][q.id] = idx
    else:  # short
        val = st.text_input("Your answer (short text):", value=default or "", key=key)
        st.session_state["responses"][q.id] = val

    # Flagging
    flagged = q.id in st.session_state["flags"]
    col1, col2 = st.columns([1,3])
    with col1:
        if st.button("🚩 Flag", key=f"flag_{q.id}"):
            if flagged:
                st.session_state["flags"].discard(q.id)
            else:
                st.session_state["flags"].add(q.id)
            st.experimental_rerun()
    with col2:
        if st.session_state.get("graded", False) and st.session_state.get("show_answers", False):
            user_answer = st.session_state["responses"].get(q.id)
            correct, pts = grade_response(q, user_answer)
            st.info(f"Answer: {format_answer(q)} | You: {format_user_answer(q, user_answer)} | {'✅ Correct' if correct else '❌ Incorrect'} | +{pts}/{q.points}")
            if q.explanation:
                st.caption(f"Explanation: {q.explanation}")

    st.divider()


def list_id(qid: str) -> int:
    ids = [q.id for q in st.session_state.get("questions", [])]
    return ids.index(qid) + 1


def format_answer(q: Question) -> str:
    if q.type == "mcq":
        try:
            return q.options[q.answer]
        except Exception:
            return str(q.answer)
    return str(q.answer)


def format_user_answer(q: Question, ua: Any) -> str:
    if ua is None:
        return "(no answer)"
    if q.type == "mcq" and isinstance(ua, int):
        try:
            return q.options[ua]
        except Exception:
            return str(ua)
    return str(ua)


def main():
    st.set_page_config(page_title="PDF → Quiz (Gemini)", page_icon="🧠", layout="wide")
    ensure_state_defaults()
    render_header()

    opts = sidebar_controls()

    with st.container(border=True):
        st.subheader("1) Upload PDF & Generate Quiz")
        file = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)
        colA, colB = st.columns([1,1])
        with colA:
            gen_btn = st.button("✨ Generate Quiz", use_container_width=True)
        with colB:
            clear_btn = st.button("🗑️ Reset", use_container_width=True)

        if clear_btn:
            reset_quiz_state()
            st.experimental_rerun()

        if gen_btn or (opts["regen"] and (file or st.session_state.get("_cached_text"))):
            if not file and not st.session_state.get("_cached_text"):
                st.error("Please upload a PDF first.")
            else:
                with st.spinner("Generating quiz…"):
                    text = st.session_state.get("_cached_text")
                    if file:
                        text = extract_text_from_pdf(file)
                        st.session_state["_cached_text"] = text
                    if not text:
                        st.error("Could not extract any text from the PDF.")
                    else:
                        create_quiz_from_text(text, opts)
                        start_timer(opts["time_mins"])
                        st.success("Quiz generated!")
                        st.experimental_rerun()

    if st.session_state.get("questions"):
        with st.container(border=True):
            st.subheader("2) Take the Quiz")
            render_timer()
            for q in st.session_state["questions"]:
                render_question(q)

            left, right = st.columns([1,1])
            with left:
                submit = st.button("✅ Submit & Grade", use_container_width=True)
            with right:
                st.session_state["show_answers"] = st.toggle("Show answers after grading", value=st.session_state.get("show_answers", False))

            if submit and not st.session_state.get("graded", False):
                grade_quiz()
                st.experimental_rerun()

    if st.session_state.get("graded", False):
        with st.container(border=True):
            st.subheader("3) Results & Review")
            st.metric("Score", f"{st.session_state['score']} / {st.session_state['max_score']}")
            pct = (st.session_state['score'] / max(1.0, st.session_state['max_score'])) * 100
            st.progress(pct/100.0, text=f"{pct:.1f}%")

            if st.session_state.get("show_answers", False):
                st.info("Scroll up to see the correct answers and explanations under each question.")

            st.markdown("**Flagged questions**")
            flags = [list_id(fid) for fid in st.session_state.get("flags", set())]
            st.write(flags if flags else "(none)")

            st.divider()
            st.markdown("**Export**")
            export_results()
            export_quiz()

    with st.expander("ℹ️ About & Tips"):
        st.markdown(
            "- For best results, upload PDFs with clear, text-based content (not scanned images).\n"
            "- Use 'Regenerate Quiz' in the sidebar to try different mixes and difficulty.\n"
            "- Short-answer grading expects exact (case-insensitive) matches.\n"
            "- If no Gemini key is found, the app uses a simple offline generator."
        )
        key = get_api_key()
        if key:
            st.success("Gemini API key detected.")
        else:
            st.warning("No Gemini API key found. Set GEMINI_API_KEY for higher-quality questions.")


if __name__ == "__main__":
    main()
