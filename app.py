import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import ats_scoring
import excel_store
import import_pipeline

load_dotenv()

APPLICANTS_PATH = "Applicants.xlsx"


def _init_state():
    defaults = {
        "drive_files": {},
        "drive_file_meta": {},
        "skills_input": "",
        "last_report": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _job_description_section():
    st.header("1. Job Description")
    jd_text = st.text_area(
        "Paste the job description",
        height=180,
        placeholder="e.g. We're hiring a Full Stack Engineer with 3+ years of experience in "
        "Python, React, and AWS...",
        key="jd_text",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔍 Auto-extract required skills from JD") and jd_text.strip():
            extracted = ats_scoring.extract_skills_from_jd(jd_text)
            # Must be set before the text_input(key="skills_input") below is
            # instantiated this run -- Streamlit ignores a widget's `value=`
            # once its session_state key already exists, so priming the key
            # itself is the only way a button click can update it.
            st.session_state.skills_input = ", ".join(extracted)
            if not extracted:
                st.warning("No known skills recognized in that JD text. Add them manually below.")

    skills_input = st.text_input(
        "Required skills (comma-separated -- edit the auto-extracted list or type your own)",
        key="skills_input",
    )
    required_skills = [s.strip().lower() for s in skills_input.split(",") if s.strip()]

    min_experience = st.number_input(
        "Minimum years of experience required", min_value=0.0, max_value=40.0, value=0.0, step=0.5
    )

    use_semantic = st.checkbox(
        "Enable semantic similarity scoring (slower, downloads a small model on first use)"
    )

    st.subheader("Recommendation thresholds")
    t1, t2 = st.columns(2)
    with t1:
        shortlist_threshold = st.slider("Shortlist if ATS Score >=", 0, 100, int(ats_scoring.SHORTLIST_THRESHOLD))
    with t2:
        maybe_threshold = st.slider("Maybe if ATS Score >=", 0, 100, int(ats_scoring.MAYBE_THRESHOLD))

    jd = ats_scoring.JobDescription(
        raw_text=jd_text, required_skills=required_skills, min_experience_years=min_experience
    )
    return jd, use_semantic, float(shortlist_threshold), float(maybe_threshold)


def _resume_source_section():
    st.header("2. Provide Resumes")
    tab_upload, tab_drive = st.tabs(["⬆️ Upload files", "🔗 Google Drive link"])

    uploaded_files = None
    with tab_upload:
        uploaded_files = st.file_uploader(
            "Choose PDF/DOC/DOCX files", type=["pdf", "doc", "docx"], accept_multiple_files=True
        )

    drive_link = None
    with tab_drive:
        if not (os.environ.get("GOOGLE_DRIVE_API_KEY") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")):
            st.caption(
                "⚠️ No Google Drive API credentials configured (`GOOGLE_DRIVE_API_KEY` in `.env`). "
                "Falling back to unauthenticated access, which Google may rate-limit."
            )
        st.caption(
            "Paste a Google Drive **folder** link shared as 'Anyone with the link can view'. "
            "All PDF/DOC/DOCX files inside it (including subfolders) will be imported."
        )
        drive_link = st.text_input("Google Drive folder link", key="drive_link_input")

    return uploaded_files, drive_link


def _run_import_section(jd, use_semantic, shortlist_threshold, maybe_threshold, uploaded_files, drive_link):
    st.header("3. Run Import")
    can_run = bool(uploaded_files) or bool(drive_link and drive_link.strip())

    if st.button("🚀 Run Import & Scoring", disabled=not can_run):
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        counters = st.empty()

        state = {"total": 0}

        def progress_cb(done, total, filename):
            state["total"] = total
            frac = (done / total) if total else 0.0
            progress_bar.progress(frac)
            status_text.text(f"Processing ({done}/{total}): {filename}")

        upload_bytes = None
        if uploaded_files:
            upload_bytes = {f.name: f.getvalue() for f in uploaded_files}

        with st.spinner("Running import pipeline..."):
            report = import_pipeline.run_import(
                drive_link=drive_link.strip() if drive_link and drive_link.strip() else None,
                jd=jd,
                applicants_path=APPLICANTS_PATH,
                uploaded_files=upload_bytes,
                progress_cb=progress_cb,
                use_semantic=use_semantic,
                shortlist_threshold=shortlist_threshold,
                maybe_threshold=maybe_threshold,
            )

        st.session_state.last_report = report
        progress_bar.progress(1.0)
        status_text.text("Done.")

        for w in report.drive_warnings:
            st.warning(f"⚠️ {w}")

        counters.markdown(
            f"**Total:** {report.total_files} &nbsp;|&nbsp; "
            f"**Imported:** {report.imported} &nbsp;|&nbsp; "
            f"**Updated:** {report.updated} &nbsp;|&nbsp; "
            f"**Duplicates skipped:** {report.duplicates_skipped} &nbsp;|&nbsp; "
            f"**Failed:** {report.failed} &nbsp;|&nbsp; "
            f"**Progress:** {report.progress_pct}%"
        )

        if report.outcomes:
            report_path = import_pipeline.save_import_report(
                report, f"import_report_{report.started_at.replace(':', '-').replace(' ', '_')}.xlsx"
            )
            with open(report_path, "rb") as f:
                st.download_button(
                    "📥 Download Import Report", f, file_name=Path(report_path).name
                )


def _dashboard_section():
    st.header("4. Candidate Dashboard")

    if not Path(APPLICANTS_PATH).exists():
        st.info("💡 No candidates imported yet. Run an import above to populate the dashboard.")
        return

    applicants, _dedup = excel_store.load_applicants(APPLICANTS_PATH)
    if applicants.empty:
        st.info("💡 No candidates imported yet. Run an import above to populate the dashboard.")
        return

    numeric_score = pd.to_numeric(applicants["ATS Score"], errors="coerce").fillna(0.0)

    st.subheader("Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Applicants", len(applicants))
    m2.metric("Shortlisted ✅", int((applicants["Recommendation"] == "Shortlist").sum()))
    m3.metric("Rejected ❌", int((applicants["Recommendation"] == "Reject").sum()))
    m4.metric("Average ATS Score", f"{numeric_score.mean():.1f}" if len(applicants) else "0")

    all_skills = (
        applicants["Skills"].dropna().str.split(",").explode().str.strip()
    )
    all_skills = all_skills[all_skills != ""]
    if not all_skills.empty:
        st.subheader("Top Skills")
        st.bar_chart(all_skills.value_counts().head(10))

    st.subheader("Filters")
    f1, f2 = st.columns(2)
    with f1:
        rec_filter = st.multiselect(
            "Recommendation", options=["Shortlist", "Maybe", "Reject"],
            default=["Shortlist", "Maybe", "Reject"],
        )
    with f2:
        min_score_filter = st.slider("Minimum ATS Score", 0, 100, 0)

    filtered = applicants[
        applicants["Recommendation"].isin(rec_filter) & (numeric_score >= min_score_filter)
    ]

    st.subheader(f"Candidates ({len(filtered)})")
    dashboard_view = filtered.rename(columns={
        "Resume File Name": "Resume Link",
        "Imported Time": "Imported Date",
    })[[
        "Name", "Email", "Phone", "Experience", "Skills", "ATS Score",
        "Recommendation", "Status", "Resume Link", "Imported Date",
    ]]
    st.dataframe(dashboard_view, width="stretch")

    with open(APPLICANTS_PATH, "rb") as f:
        st.download_button(
            "📥 Download Applicants.xlsx", f, file_name="Applicants.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def main():
    st.set_page_config(page_title="ATS Resume Shortlisting Tool", layout="wide")
    st.title("📄 ATS Resume Import & Shortlisting Pipeline")
    st.write(
        "Import resumes from Google Drive (or upload directly), score them against a job "
        "description, and track every candidate in a running Excel database."
    )

    _init_state()

    jd, use_semantic, shortlist_threshold, maybe_threshold = _job_description_section()
    uploaded_files, drive_link = _resume_source_section()
    _run_import_section(jd, use_semantic, shortlist_threshold, maybe_threshold, uploaded_files, drive_link)
    _dashboard_section()


if __name__ == "__main__":
    main()
