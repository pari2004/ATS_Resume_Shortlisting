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
        "required_skills_input": "",
        "preferred_skills_input": "",
        "last_report": None,
        "last_jd": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _recommendation_tier(recommendation_cell: str) -> str:
    """"Shortlist - Strong React and Node.js experience" -> "Shortlist".
    Works unchanged for cells with no appended sentence too."""
    if not isinstance(recommendation_cell, str) or not recommendation_cell:
        return ""
    return recommendation_cell.split(" - ", 1)[0].strip()


def _job_description_section():
    st.header("1. Job Description")
    jd_text = st.text_area(
        "Paste the job description",
        height=200,
        placeholder="Paste any job description -- software, sales, HR, finance, healthcare, "
        "engineering, etc. The domain and required skills are detected automatically below.",
        key="jd_text",
    )

    if st.button("🔍 Analyze Job Description") and jd_text.strip():
        jd = ats_scoring.analyze_job_description(jd_text)
        st.session_state.last_jd = jd
        # Must be set before the widgets below are instantiated this run --
        # Streamlit ignores a widget's `value=` once its session_state key
        # already exists, so priming the key itself is the only way a
        # button click can update it.
        st.session_state.required_skills_input = ", ".join(jd.required_skills)
        st.session_state.preferred_skills_input = ", ".join(jd.preferred_skills)
        st.session_state["min_exp_input"] = jd.min_experience_years
        st.session_state["preferred_exp_input"] = jd.preferred_experience_years

    detected = st.session_state.last_jd
    if detected:
        d1, d2 = st.columns(2)
        d1.markdown(f"**Detected Job Title:** {detected.title or '_(not detected)_'}")
        d2.markdown(f"**Detected Domain:** {detected.domain}")
        if detected.soft_skills:
            st.caption("Soft skills mentioned: " + ", ".join(detected.soft_skills))
        if detected.keywords:
            with st.expander(f"Keywords ({len(detected.keywords)})"):
                st.write(", ".join(detected.keywords))
        if detected.responsibilities:
            with st.expander(f"Responsibilities ({len(detected.responsibilities)})"):
                for line in detected.responsibilities:
                    st.markdown(f"- {line}")

    s1, s2 = st.columns(2)
    with s1:
        required_input = st.text_input(
            "Required skills (comma-separated, editable)", key="required_skills_input"
        )
    with s2:
        preferred_input = st.text_input(
            "Preferred skills (comma-separated, editable)", key="preferred_skills_input"
        )
    required_skills = [s.strip().lower() for s in required_input.split(",") if s.strip()]
    preferred_skills = [s.strip().lower() for s in preferred_input.split(",") if s.strip()]

    e1, e2, e3 = st.columns(3)
    with e1:
        min_experience = st.number_input(
            "Required years of experience", min_value=0.0, max_value=40.0, step=0.5, key="min_exp_input"
        )
    with e2:
        preferred_experience = st.number_input(
            "Preferred years of experience", min_value=0.0, max_value=40.0, step=0.5, key="preferred_exp_input"
        )
    with e3:
        education_options = ["", "high school", "diploma", "bachelor's", "master's", "phd"]
        default_edu = detected.education_requirement if detected else ""
        education_requirement = st.selectbox(
            "Education requirement", options=education_options,
            index=education_options.index(default_edu) if default_edu in education_options else 0,
        )

    certs_default = ", ".join(detected.certifications_required) if detected else ""
    certs_input = st.text_input("Certifications required (comma-separated)", value=certs_default)
    certifications_required = [c.strip().lower() for c in certs_input.split(",") if c.strip()]

    with st.expander("⚖️ Scoring weights (advanced)"):
        st.caption("Don't need to sum to 100 -- automatically normalized.")
        w1, w2, w3 = st.columns(3)
        with w1:
            w_required = st.slider("Required skills %", 0, 100, 40)
            w_preferred = st.slider("Preferred skills %", 0, 100, 20)
        with w2:
            w_experience = st.slider("Experience %", 0, 100, 15)
            w_education = st.slider("Education %", 0, 100, 10)
        with w3:
            w_certifications = st.slider("Certifications %", 0, 100, 10)
            w_quality = st.slider("Resume quality %", 0, 100, 5)

    weights = ats_scoring.ScoringWeights(
        required_skills=w_required, preferred_skills=w_preferred, experience=w_experience,
        education=w_education, certifications=w_certifications, resume_quality=w_quality,
    )

    st.subheader("Recommendation thresholds")
    t1, t2 = st.columns(2)
    with t1:
        shortlist_threshold = st.slider("Shortlist if ATS Score >=", 0, 100, int(ats_scoring.SHORTLIST_THRESHOLD))
    with t2:
        maybe_threshold = st.slider("Maybe if ATS Score >=", 0, 100, int(ats_scoring.MAYBE_THRESHOLD))

    jd = ats_scoring.JobDescription(
        raw_text=jd_text,
        title=detected.title if detected else "",
        domain=detected.domain if detected else "General / Other",
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        soft_skills=detected.soft_skills if detected else [],
        min_experience_years=min_experience,
        preferred_experience_years=preferred_experience,
        education_requirement=education_requirement,
        certifications_required=certifications_required,
        keywords=detected.keywords if detected else [],
        responsibilities=detected.responsibilities if detected else [],
    )
    return jd, weights, float(shortlist_threshold), float(maybe_threshold)


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
        # A queued clear must be applied *before* the text_input below is
        # instantiated this run -- Streamlit forbids writing to a widget's
        # session_state key after that widget has already been drawn, so
        # the button (further down, drawn after the input) can only queue
        # the clear for the next rerun, not apply it immediately.
        if st.session_state.pop("_clear_drive_link_pending", False):
            st.session_state.drive_link_input = ""

        link_col, clear_col = st.columns([5, 1])
        with link_col:
            drive_link = st.text_input(
                "Google Drive folder link", key="drive_link_input", label_visibility="collapsed"
            )
        with clear_col:
            if st.button("🗑️ Clear"):
                st.session_state["_clear_drive_link_pending"] = True
                st.rerun()

    return uploaded_files, drive_link


def _run_import_section(jd, weights, shortlist_threshold, maybe_threshold, uploaded_files, drive_link):
    st.header("3. Run Import")
    can_run = bool(uploaded_files) or bool(drive_link and drive_link.strip())

    if st.button("🚀 Run Import & Scoring", disabled=not can_run):
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        counters = st.empty()

        def progress_cb(done, total, filename):
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
                weights=weights,
                shortlist_threshold=shortlist_threshold,
                maybe_threshold=maybe_threshold,
            )

        st.session_state.last_report = report
        progress_bar.progress(1.0)
        status_text.text("Done.")

        for w in report.drive_warnings:
            st.warning(f"⚠️ {w}")

        counters.markdown(
            f"**Domain:** {jd.domain} &nbsp;|&nbsp; "
            f"**Total:** {report.total_files} &nbsp;|&nbsp; "
            f"**Imported:** {report.imported} &nbsp;|&nbsp; "
            f"**Updated:** {report.updated} &nbsp;|&nbsp; "
            f"**Rescored:** {report.rescored} &nbsp;|&nbsp; "
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
    tiers = applicants["Recommendation"].apply(_recommendation_tier)

    st.subheader("Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Applicants", len(applicants))
    m2.metric("Shortlisted ✅", int((tiers == "Shortlist").sum()))
    m3.metric("Rejected ❌", int((tiers == "Reject").sum()))
    m4.metric("Average ATS Score", f"{numeric_score.mean():.1f}" if len(applicants) else "0")

    domains_present = applicants["Detected Job Domain"].dropna()
    domains_present = domains_present[domains_present != ""]
    if not domains_present.empty:
        st.caption("Domains represented in this candidate pool: " + ", ".join(sorted(domains_present.unique())))

    c1, c2 = st.columns(2)
    with c1:
        all_skills = applicants["Skills"].dropna().str.split(",").explode().str.strip()
        all_skills = all_skills[all_skills != ""]
        if not all_skills.empty:
            st.subheader("Skill Distribution")
            st.bar_chart(all_skills.value_counts().head(10))

    with c2:
        missing = applicants["Missing Skills"].dropna().str.split(",").explode().str.strip()
        missing = missing[missing != ""]
        if not missing.empty:
            st.subheader("Top Missing Skills")
            st.bar_chart(missing.value_counts().head(10))

    exp_years = applicants["Experience"].str.extract(r"([\d.]+)\s*years?")[0]
    exp_years = pd.to_numeric(exp_years, errors="coerce").dropna()
    if not exp_years.empty:
        st.subheader("Experience Distribution")
        bins = pd.cut(exp_years, bins=[0, 1, 2, 3, 5, 8, 12, 100],
                       labels=["0-1", "1-2", "2-3", "3-5", "5-8", "8-12", "12+"], right=False)
        st.bar_chart(bins.value_counts().sort_index())

    st.subheader("Filters")
    f1, f2, f3 = st.columns(3)
    with f1:
        rec_filter = st.multiselect(
            "Recommendation", options=["Shortlist", "Maybe", "Reject"],
            default=["Shortlist", "Maybe", "Reject"],
        )
    with f2:
        min_score_filter = st.slider("Minimum ATS Score", 0, 100, 0)
    with f3:
        domain_options = ["All"] + sorted(domains_present.unique().tolist()) if not domains_present.empty else ["All"]
        domain_filter = st.selectbox("Domain", options=domain_options)

    mask = tiers.isin(rec_filter) & (numeric_score >= min_score_filter)
    if domain_filter != "All":
        mask &= applicants["Detected Job Domain"] == domain_filter
    filtered = applicants[mask]

    st.subheader(f"Candidates ({len(filtered)})")
    dashboard_view = filtered.rename(columns={
        "Resume File Name": "Resume Link",
        "Imported Time": "Imported Date",
    })[[
        "Name", "Email", "Phone", "Experience", "Skills", "Detected Job Domain",
        "ATS Score", "Recommendation", "Status", "Resume Link", "Imported Date",
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
        "Import resumes from Google Drive (or upload directly), score them against any "
        "job description -- software, sales, HR, finance, healthcare, engineering, and more -- "
        "and track every candidate in a running Excel database."
    )

    _init_state()

    jd, weights, shortlist_threshold, maybe_threshold = _job_description_section()
    uploaded_files, drive_link = _resume_source_section()
    _run_import_section(jd, weights, shortlist_threshold, maybe_threshold, uploaded_files, drive_link)
    _dashboard_section()


if __name__ == "__main__":
    main()
