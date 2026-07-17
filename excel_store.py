"""Applicants.xlsx: the running candidate database. Loads/creates the sheet,
and upserts candidate records without ever discarding unrelated existing rows
-- new candidates are appended, resumes that match an existing candidate
(by email, phone, or exact resume hash) update that candidate's row in place.

The workbook has two sheets:
  - "Applicants": exactly the user-facing columns (COLUMNS below).
  - "_dedup_index": Candidate ID -> Resume Hash, kept out of the visible
    Applicants sheet since it's an internal bookkeeping detail, not part of
    the requested schema, but still needed to detect "this exact resume was
    already imported" without re-reading every past resume's file.
"""

import logging
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger("drive_utils")

COLUMNS = [
    "Candidate ID",
    "Name",
    "Email",
    "Phone",
    "Experience",
    "Skills",
    "Detected Job Domain",
    "Education",
    "ATS Score",
    "Skill Match %",
    "Experience Match %",
    "Matched Skills",
    "Missing Skills",
    "Recommendation",
    "Status",
    "Resume File Name",
    "Google Drive File ID",
    "Google Drive URL",
    "Imported Time",
]

APPLICANTS_SHEET = "Applicants"
DEDUP_SHEET = "_dedup_index"
DEDUP_COLUMNS = ["Candidate ID", "Resume Hash"]


def load_applicants(path: str = "Applicants.xlsx") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (applicants_df, dedup_df)."""
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS), pd.DataFrame(columns=DEDUP_COLUMNS)

    applicants = pd.read_excel(path, sheet_name=APPLICANTS_SHEET, engine="openpyxl", dtype=str)
    for col in COLUMNS:
        if col not in applicants.columns:
            applicants[col] = ""
    applicants = applicants[COLUMNS].fillna("")

    try:
        dedup = pd.read_excel(path, sheet_name=DEDUP_SHEET, engine="openpyxl", dtype=str)
        for col in DEDUP_COLUMNS:
            if col not in dedup.columns:
                dedup[col] = ""
        dedup = dedup[DEDUP_COLUMNS].fillna("")
    except ValueError:
        # Sheet not present (e.g. a workbook created before this feature existed).
        dedup = pd.DataFrame(columns=DEDUP_COLUMNS)

    return applicants, dedup


def _normalize_phone(phone: str) -> str:
    return "".join(ch for ch in str(phone) if ch.isdigit())


def _to_cell_str(value) -> str:
    """Coerces any record value (float ATS scores, int counts, etc.) to a
    plain string. Required because pandas' ``dtype=str`` on ``read_excel``
    (pandas >= 3.0) produces a strict StringDtype that raises on assigning a
    raw int/float via ``.at[]`` -- everything in this sheet is text anyway."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _next_candidate_id(df: pd.DataFrame) -> str:
    max_n = 0
    for cid in df["Candidate ID"].dropna():
        val = str(cid).strip().upper()
        if val.startswith("CAND-"):
            try:
                max_n = max(max_n, int(val.split("-", 1)[1]))
            except ValueError:
                continue
    return f"CAND-{max_n + 1:04d}"


def find_existing_candidate(
    applicants: pd.DataFrame, dedup: pd.DataFrame, email: str, phone: str, resume_hash: str
) -> Tuple[int, str]:
    """Returns (row_index_in_applicants, match_reason) or (-1, "")."""
    if applicants.empty:
        return -1, ""

    if resume_hash and not dedup.empty:
        hit = dedup.index[dedup["Resume Hash"] == resume_hash]
        if len(hit):
            candidate_id = dedup.at[hit[0], "Candidate ID"]
            rows = applicants.index[applicants["Candidate ID"] == candidate_id]
            if len(rows):
                return rows[0], "hash"

    if email:
        email_norm = email.strip().lower()
        hits = applicants.index[applicants["Email"].str.strip().str.lower() == email_norm]
        if len(hits):
            return hits[0], "email"

    if phone:
        phone_norm = _normalize_phone(phone)
        if phone_norm:
            hits = applicants.index[applicants["Phone"].apply(_normalize_phone) == phone_norm]
            if len(hits):
                return hits[0], "phone"

    return -1, ""


def upsert_candidate(
    applicants: pd.DataFrame, dedup: pd.DataFrame, record: dict, resume_hash: str
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """``record`` must contain keys matching COLUMNS except 'Candidate ID' and
    'Imported Time' (assigned/refreshed here). Returns (applicants, dedup,
    action) where action is "inserted" / "updated" / "rescored".

    An exact resume-hash match (the identical file already imported) still
    refreshes the row rather than being skipped -- the resume content hasn't
    changed, but the *scoring* (ATS Score, domain, matched/missing skills,
    recommendation) depends on whatever Job Description this run is scoring
    against, which may well be different from the JD used the first time
    this file was imported. Skipping silently would leave stale scores from
    a JD that's no longer relevant."""
    email = record.get("Email", "")
    phone = record.get("Phone", "")

    row_idx, reason = find_existing_candidate(applicants, dedup, email, phone, resume_hash)

    record["Imported Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if row_idx >= 0:
        candidate_id = applicants.at[row_idx, "Candidate ID"]
        for col in COLUMNS:
            if col == "Candidate ID":
                continue
            applicants.at[row_idx, col] = _to_cell_str(record.get(col, ""))
        dedup = dedup[dedup["Candidate ID"] != candidate_id]
        dedup = pd.concat(
            [dedup, pd.DataFrame([{"Candidate ID": candidate_id, "Resume Hash": resume_hash}])],
            ignore_index=True,
        )
        action = "rescored" if reason == "hash" else "updated"
        logger.info(
            f"{'Rescored' if action == 'rescored' else 'Updated'} candidate {candidate_id} "
            f"({reason} match): {record.get('Resume File Name')}"
        )
        return applicants, dedup, action

    candidate_id = _next_candidate_id(applicants)
    record["Candidate ID"] = candidate_id
    new_row = {col: _to_cell_str(record.get(col, "")) for col in COLUMNS}
    applicants = pd.concat([applicants, pd.DataFrame([new_row])], ignore_index=True)
    dedup = pd.concat(
        [dedup, pd.DataFrame([{"Candidate ID": candidate_id, "Resume Hash": resume_hash}])],
        ignore_index=True,
    )
    logger.info(f"Inserted new candidate {candidate_id}: {record.get('Resume File Name')}")
    return applicants, dedup, "inserted"


def save_applicants(
    applicants: pd.DataFrame, dedup: pd.DataFrame, path: str = "Applicants.xlsx"
) -> None:
    """Atomic write: build the new workbook next to the target, then replace
    it, so a crash mid-write can never leave ``path`` truncated/corrupted."""
    target_dir = Path(path).resolve().parent
    fd, tmp_name = tempfile.mkstemp(suffix=".xlsx", dir=str(target_dir))
    os.close(fd)
    try:
        with pd.ExcelWriter(tmp_name, engine="openpyxl") as writer:
            applicants[COLUMNS].to_excel(writer, sheet_name=APPLICANTS_SHEET, index=False)
            dedup[DEDUP_COLUMNS].to_excel(writer, sheet_name=DEDUP_SHEET, index=False)
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
