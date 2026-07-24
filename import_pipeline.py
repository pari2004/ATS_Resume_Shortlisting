"""Orchestrates the full pipeline: Drive fetch -> parse -> score -> upsert
into Applicants.xlsx -> import report. One bad file never aborts the batch --
every per-file failure is caught, logged, and counted, and the loop moves on.
"""

import logging
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

import ats_scoring
import excel_store
import resume_parser
from drive_utils import DriveFetchError, fetch_pdfs_from_drive

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ats_tool.log"), logging.StreamHandler()],
)
logger = logging.getLogger("drive_utils")

ProgressCallback = Callable[[int, int, str], None]


@dataclass
class FileOutcome:
    file_name: str
    status: str  # "imported" | "updated" | "rescored" | "failed" | "skipped_unsupported"
    reason: str = ""
    ats_score: Optional[float] = None
    processing_seconds: float = 0.0


@dataclass
class ImportReport:
    total_files: int = 0
    imported: int = 0
    updated: int = 0
    rescored: int = 0
    failed: int = 0
    unsupported_skipped: int = 0
    drive_warnings: List[str] = field(default_factory=list)
    outcomes: List[FileOutcome] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    total_seconds: float = 0.0

    @property
    def processed(self) -> int:
        return self.imported + self.updated + self.rescored + self.failed

    @property
    def progress_pct(self) -> float:
        return round(self.processed / self.total_files * 100.0, 1) if self.total_files else 0.0

    def to_rows(self) -> List[dict]:
        return [
            {
                "File Name": o.file_name,
                "Status": o.status,
                "Reason": o.reason,
                "ATS Score": o.ats_score if o.ats_score is not None else "",
                "Processing Time (s)": round(o.processing_seconds, 2),
            }
            for o in self.outcomes
        ]


def _process_one_file(
    filename: str,
    file_bytes: bytes,
    jd: ats_scoring.JobDescription,
    applicants,
    dedup,
    file_meta: Dict[str, Dict[str, str]],
    staging_dir: Path,
    weights: ats_scoring.ScoringWeights,
    shortlist_threshold: float,
    maybe_threshold: float,
):
    started = time.time()
    text, is_scanned, ok = resume_parser.extract_text(file_bytes, filename, staging_dir)

    if not ok:
        elapsed = time.time() - started
        logger.warning(f"Failed to extract text from {filename}")
        return applicants, dedup, FileOutcome(
            file_name=filename, status="failed", reason="Could not extract any text",
            processing_seconds=elapsed,
        )

    parsed = resume_parser.parse_resume(text, filename, is_scanned=is_scanned)
    result = ats_scoring.score_resume(
        parsed, jd, weights=weights,
        shortlist_threshold=shortlist_threshold, maybe_threshold=maybe_threshold,
    )

    meta = file_meta.get(filename, {})
    matched_skills = result.matched_required_skills + result.matched_preferred_skills
    # "Recommendation" stays exact-match filterable (Shortlist/Maybe/Reject
    # as its own leading token) with the domain-aware sentence appended --
    # dashboard filtering/counting splits on " - " to recover the tier.
    recommendation_cell = result.recommendation
    if result.recommendation_text:
        recommendation_cell = f"{result.recommendation} - {result.recommendation_text}"
    record = {
        "Name": parsed.name,
        "Email": parsed.email,
        "Phone": parsed.phone,
        "Experience": f"{parsed.total_experience_years} years"
        + (f" ({parsed.current_designation} at {parsed.current_company})"
           if parsed.current_designation or parsed.current_company else ""),
        "Skills": ", ".join(parsed.skills),
        "Detected Job Domain": jd.domain,
        "Education": "; ".join(parsed.education),
        "ATS Score": result.ats_score,
        "Skill Match %": result.required_skill_match_pct,
        "Experience Match %": result.experience_match_pct,
        "Matched Skills": ", ".join(matched_skills),
        "Missing Skills": ", ".join(result.missing_required_skills),
        "Recommendation": recommendation_cell,
        "Status": "New",
        "Resume File Name": filename,
        "Google Drive File ID": meta.get("file_id", ""),
        "Google Drive URL": meta.get("url", ""),
    }

    applicants, dedup, action = excel_store.upsert_candidate(
        applicants, dedup, record, resume_hash=parsed.resume_hash
    )
    elapsed = time.time() - started
    status_map = {"inserted": "imported", "updated": "updated", "rescored": "rescored"}
    outcome = FileOutcome(
        file_name=filename,
        status=status_map[action],
        ats_score=result.ats_score,
        processing_seconds=elapsed,
    )
    logger.info(f"Processed {filename} in {elapsed:.2f}s -> {outcome.status} (ATS {result.ats_score})")
    return applicants, dedup, outcome


def run_import(
    drive_link: Optional[str],
    jd: ats_scoring.JobDescription,
    applicants_path: str = "Applicants.xlsx",
    uploaded_files: Optional[Dict[str, bytes]] = None,
    progress_cb: Optional[ProgressCallback] = None,
    weights: Optional[ats_scoring.ScoringWeights] = None,
    shortlist_threshold: float = ats_scoring.SHORTLIST_THRESHOLD,
    maybe_threshold: float = ats_scoring.MAYBE_THRESHOLD,
    replace_existing: bool = False,
) -> ImportReport:
    """Runs the full pipeline for either a Drive folder link or a pre-fetched
    dict of {filename: bytes} (e.g. from Streamlit's file uploader). Exactly
    one of ``drive_link`` / ``uploaded_files`` should be given.

    ``replace_existing=True`` discards whatever is currently in
    ``applicants_path`` before this run's candidates are added, so the saved
    workbook ends up containing only this run's results instead of the usual
    append/update-in-place history."""
    report = ImportReport(started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    run_start = time.time()

    file_meta: Dict[str, Dict[str, str]] = {}
    if drive_link:
        try:
            fetch_result = fetch_pdfs_from_drive(drive_link)
        except DriveFetchError as e:
            report.drive_warnings.append(str(e))
            report.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return report
        files = dict(fetch_result)
        file_meta = fetch_result.file_meta
        report.drive_warnings.extend(fetch_result.warnings)
        report.unsupported_skipped = sum(
            1 for w in fetch_result.warnings if "unsupported type" in w
        )
    else:
        files = dict(uploaded_files or {})

    report.total_files = len(files)
    if progress_cb:
        progress_cb(0, report.total_files, "starting")

    if replace_existing:
        applicants = pd.DataFrame(columns=excel_store.COLUMNS)
        dedup = pd.DataFrame(columns=excel_store.DEDUP_COLUMNS)
    else:
        applicants, dedup = excel_store.load_applicants(applicants_path)
    weights = weights or ats_scoring.ScoringWeights()

    with tempfile.TemporaryDirectory() as staging_dir_str:
        staging_dir = Path(staging_dir_str)
        for i, (filename, file_bytes) in enumerate(files.items(), start=1):
            try:
                applicants, dedup, outcome = _process_one_file(
                    filename, file_bytes, jd, applicants, dedup, file_meta,
                    staging_dir, weights, shortlist_threshold, maybe_threshold,
                )
            except Exception as e:
                logger.error(f"Unexpected error processing {filename}: {e}")
                outcome = FileOutcome(file_name=filename, status="failed", reason=str(e))

            report.outcomes.append(outcome)
            if outcome.status == "imported":
                report.imported += 1
            elif outcome.status == "updated":
                report.updated += 1
            elif outcome.status == "rescored":
                report.rescored += 1
            else:
                report.failed += 1

            if progress_cb:
                progress_cb(i, report.total_files, filename)

    excel_store.save_applicants(applicants, dedup, applicants_path)

    report.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report.total_seconds = round(time.time() - run_start, 2)
    logger.info(
        f"Import complete: {report.imported} imported, {report.updated} updated, "
        f"{report.rescored} rescored, {report.failed} failed, "
        f"{report.unsupported_skipped} unsupported, in {report.total_seconds}s"
    )
    return report


def save_import_report(report: ImportReport, path: str) -> str:
    import pandas as pd

    rows = report.to_rows()
    if not rows:
        Path(path).with_suffix(".txt").write_text(
            "No files were processed in this run.\n"
            + "\n".join(report.drive_warnings),
            encoding="utf-8",
        )
        return str(Path(path).with_suffix(".txt"))

    df = pd.DataFrame(rows)
    df.to_excel(path, index=False, engine="openpyxl")
    return path
