import json
import os
import re
from typing import Any

import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader


# -----------------------------
# App configuration
# -----------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")
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


def analyze_resume(resume_text: str, api_key: str) -> dict[str, Any]:
    """Ask Gemini to evaluate the resume and return structured JSON."""
    client = genai.Client(api_key=api_key)

    schema = {
        "type": "OBJECT",
        "properties": {
            "ats_score": {
                "type": "INTEGER",
                "description": "Overall ATS-readiness score from 0 to 100.",
            },
            "summary": {
                "type": "STRING",
                "description": "Short explanation of the score.",
            },
            "category_scores": {
                "type": "OBJECT",
                "properties": {
                    "format": {"type": "INTEGER"},
                    "keywords": {"type": "INTEGER"},
                    "experience": {"type": "INTEGER"},
                    "skills": {"type": "INTEGER"},
                    "clarity": {"type": "INTEGER"},
                },
                "required": [
                    "format",
                    "keywords",
                    "experience",
                    "skills",
                    "clarity",
                ],
            },
            "strengths": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "improvements": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "ats_warnings": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "missing_keywords": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
        },
        "required": [
            "ats_score",
            "summary",
            "category_scores",
            "strengths",
            "improvements",
            "ats_warnings",
            "missing_keywords",
        ],
    }

    prompt = f"""
You are an expert ATS resume evaluator and career coach.

Analyze the resume text below for general ATS compatibility and recruiter
readability. Do NOT claim that your score is an official score from a
particular ATS vendor. It is an estimated ATS-readiness score.

Score the resume from 0 to 100 using these dimensions:
- format: simple, parseable structure; standard headings; no obvious parsing risks
- keywords: relevant job-related terminology and measurable keywords
- experience: clear responsibilities, achievements, impact, and metrics
- skills: relevant technical/professional skills stated clearly
- clarity: concise language, consistency, grammar, and easy scanning

Important:
1. Do not invent information that is not in the resume.
2. Do not penalize a resume merely because it has no photo.
3. Prefer standard section names such as Summary, Experience, Education, Skills,
   Projects, Certifications.
4. Flag possible ATS risks such as tables, columns, graphics, unusual symbols,
   headers/footers, or contact information that may be difficult to parse.
   Only flag them when the extracted text provides evidence or when the issue
   is reasonably inferable.
5. Give practical, specific improvements.
6. Missing keywords should be based only on skills/terms that would reasonably
   strengthen the resume for the roles suggested by the resume itself.
7. Keep the output concise and useful.

Resume:
---BEGIN RESUME---
{resume_text}
---END RESUME---
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=2500,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    raw = response.text
    if not raw:
        raise ValueError("Gemini returned an empty response.")

    result = json.loads(clean_json_text(raw))

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
