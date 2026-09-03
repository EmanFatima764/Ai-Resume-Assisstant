import json
import os
import re
from typing import Any

import streamlit as st
import requests
from pypdf import PdfReader


# -----------------------------
# App configuration
# -----------------------------
MODEL_NAME = "gemini-3.6-flash"
MAX_RESUME_CHARS = 50000

st.set_page_config(
    page_title="Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
)


# -----------------------------
# Helpers
# -----------------------------
def get_api_key() -> str | None:
    """Read the Gemini API key from Streamlit secrets or an environment variable."""
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass

    key = os.getenv("GEMINI_API_KEY")
    return key.strip() if key else None


def extract_pdf_text(uploaded_file) -> str:
    """Extract text from every page of an uploaded PDF."""
    reader = PdfReader(uploaded_file)
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n\n".join(pages).strip()


def clean_json_text(text: str) -> str:
    """Remove accidental Markdown code fences before JSON parsing."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def analyze_resume(resume_text: str, api_key: str) -> dict:
    url = (
        "https://generativelanguage.googleapis.com/"
        "v1beta/models/gemini-2.5-flash:generateContent"
    )

    prompt = f"""
You are an expert ATS resume evaluator.

Analyze this resume and return ONLY valid JSON.

Give an estimated ATS score from 0 to 100.

Return exactly this structure:

{{
  "ats_score": 0,
  "summary": "short explanation",
  "strengths": ["strength 1", "strength 2"],
  "improvements": ["improvement 1", "improvement 2"],
  "ats_warnings": ["warning 1"],
  "missing_keywords": ["keyword 1", "keyword 2"]
}}

Do not invent information.

Resume:
{resume_text}
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    }

    response = requests.post(
        url,
        params={"key": api_key},
        json=payload,
        timeout=90
    )

    response.raise_for_status()

    data = response.json()

    text = data["candidates"][0]["content"]["parts"][0]["text"]

    return json.loads(text)
    # Defensive validation/clamping so bad model values do not break the UI.
    result["ats_score"] = max(0, min(100, int(result["ats_score"])))

    for key in result["category_scores"]:
        result["category_scores"][key] = max(
            0, min(100, int(result["category_scores"][key]))
        )

    return result


def score_label(score: int) -> str:
    if score >= 85:
        return "Excellent ATS readiness"
    if score >= 70:
        return "Good ATS readiness"
    if score >= 50:
        return "Needs improvement"
    return "High improvement needed"


# -----------------------------
# UI
# -----------------------------
st.title("📄 Resume ATS Analyzer")
st.caption(
    "Upload a PDF resume to get an estimated ATS-readiness score, "
    "strengths, warnings, and actionable improvements."
)

with st.sidebar:
    st.header("Settings")
    st.write(f"Model: `{MODEL_NAME}`")
    st.info(
        "This tool provides an AI-based estimate. Different ATS systems use "
        "different parsing and ranking rules, so the score is not an official "
        "score from any specific ATS."
    )

uploaded_file = st.file_uploader(
    "Upload your resume (PDF only)",
    type=["pdf"],
    help="Use a text-based PDF for best results. Scanned/image-only PDFs may not contain extractable text.",
)

if uploaded_file is not None:
    st.write(f"**File:** {uploaded_file.name}")

    if st.button("Analyze Resume", type="primary", use_container_width=True):
        api_key = get_api_key()

        if not api_key:
            st.error(
                "Gemini API key not found. Add GEMINI_API_KEY to "
                "Streamlit Secrets before analyzing a resume."
            )
            st.stop()

        try:
            with st.spinner("Reading your resume..."):
                resume_text = extract_pdf_text(uploaded_file)

            if not resume_text:
                st.error(
                    "No selectable text was found in this PDF. "
                    "Please upload a text-based PDF rather than a scanned image PDF."
                )
                st.stop()

            if len(resume_text) > MAX_RESUME_CHARS:
                st.warning(
                    "The PDF contains a large amount of extracted text. "
                    "Only the first 50,000 characters will be analyzed."
                )
                resume_text = resume_text[:MAX_RESUME_CHARS]

            with st.spinner("Analyzing ATS readiness with Gemini..."):
                result = analyze_resume(resume_text, api_key)

            st.success("Analysis complete.")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric("ATS Score", f"{result['ats_score']}/100")
                st.progress(result["ats_score"] / 100)
                st.write(f"**{score_label(result['ats_score'])}**")

            with col2:
                st.subheader("Summary")
                st.write(result["summary"])

            st.divider()

            st.subheader("Category Scores")
            category_labels = {
                "format": "Format",
                "keywords": "Keywords",
                "experience": "Experience",
                "skills": "Skills",
                "clarity": "Clarity",
            }

            cols = st.columns(5)
            for col, key in zip(cols, category_labels):
                with col:
                    st.metric(
                        category_labels[key],
                        f"{result['category_scores'][key]}/100",
                    )

            st.divider()

            left, right = st.columns(2)

            with left:
                st.subheader("Strengths")
                if result["strengths"]:
                    for item in result["strengths"]:
                        st.markdown(f"- {item}")
                else:
                    st.write("No major strengths were identified.")

                st.subheader("ATS Warnings")
                if result["ats_warnings"]:
                    for item in result["ats_warnings"]:
                        st.markdown(f"- {item}")
                else:
                    st.write("No major ATS warnings were identified.")

            with right:
                st.subheader("Key Improvements")
                if result["improvements"]:
                    for index, item in enumerate(result["improvements"], 1):
                        st.markdown(f"**{index}.** {item}")
                else:
                    st.write("No major improvements were identified.")

                st.subheader("Potential Missing Keywords")
                if result["missing_keywords"]:
                    for item in result["missing_keywords"]:
                        st.markdown(f"- `{item}`")
                else:
                    st.write("No obvious missing keywords were identified.")

            st.caption(
                "Tip: For the most useful keyword analysis, provide a resume "
                "tailored to the type of role you are applying for."
            )

        except Exception as exc:
            st.error(
                "The analysis could not be completed. Check your Gemini API key, "
                "internet connection, PDF format, and deployment logs."
            )
            with st.expander("Technical details"):
                st.code(str(exc))
else:
    st.info("Upload a PDF resume above to begin.")
