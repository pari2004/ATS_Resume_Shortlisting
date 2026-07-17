# ATS Resume Import & Shortlisting Tool

A production pipeline that imports resumes from a Google Drive folder (or direct
upload), parses them into structured candidate data, scores each one against a
job description, and maintains a running `Applicants.xlsx` candidate database
with a Streamlit dashboard on top.

```
Google Drive folder -> Read folder -> Download PDF/DOC/DOCX
    -> Extract resume text -> Match against Job Description
    -> Calculate ATS Score -> Generate Candidate Summary
    -> Store results in Applicants.xlsx -> Display in ATS Dashboard
```

## Architecture

| Module | Responsibility |
|---|---|
| `drive_utils.py` | Google Drive folder import. Tries, in order: Drive API v3 (service account / OAuth / API key), PyDrive2, then unauthenticated `gdown` scraping as a last resort. Filters to PDF/DOC/DOCX, skips everything else, and never lets one bad file abort the whole folder. |
| `resume_parser.py` | Text extraction (PDF via pdfplumber + OCR fallback, DOCX via docx2txt, best-effort legacy DOC) and structured field parsing (name, email, phone, address, skills, experience, education, certifications, projects, LinkedIn/GitHub/portfolio, total experience, current company/role). |
| `skills_taxonomy.py` | Shared skills vocabulary used both to find skills in a resume and to auto-extract required skills from a pasted job description. |
| `ats_scoring.py` | Scores a parsed resume against a `JobDescription` (skill match %, experience match %, optional semantic similarity) into a 0-100 ATS Score and a Shortlist/Maybe/Reject recommendation. |
| `excel_store.py` | The `Applicants.xlsx` database: load/create, and upsert candidates (append new, update on duplicate, never touch unrelated rows). Atomic writes so a crash mid-save can't corrupt the file. |
| `import_pipeline.py` | Orchestrates the whole run: fetch -> parse -> score -> upsert -> import report, with a progress callback and per-file error isolation. |
| `app.py` | Streamlit dashboard: JD input, resume source (upload or Drive link), run button with live progress, candidate table + summary metrics, downloads. |
| `main.py` | The original standalone CLI keyword-matcher, kept for simple local folder scans that don't need Drive import or full ATS scoring. |

## Installation

1. Python 3.8+
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Tesseract OCR (for scanned PDFs)
- **Windows**: https://github.com/UB-Mannheim/tesseract/wiki
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

## Google Drive setup

The importer works without any setup via `gdown` (unauthenticated), but that path
gets rate-limited by Google under repeated/heavy use. For reliable imports,
configure a **Drive API key**:

1. https://console.cloud.google.com/ -> create/select a project.
2. **APIs & Services -> Library** -> enable **Google Drive API**.
3. **APIs & Services -> Credentials -> Create Credentials -> API key**.
4. (Recommended) Restrict the key to the Drive API.
5. Create a `.env` file in the project root:
   ```
   GOOGLE_DRIVE_API_KEY=your-key-here
   ```
6. The target folder still needs to be shared as **"Anyone with the link -> Viewer"**.

For private folders, use a service account instead: set
`GOOGLE_APPLICATION_CREDENTIALS` to the path of a service-account JSON key file
that has been granted access to the folder.

## Running the dashboard

```bash
streamlit run app.py
```

1. **Job Description** -- paste the JD text, click "Auto-extract required
   skills" (editable afterwards), and set a minimum years-of-experience.
2. **Provide Resumes** -- upload files directly, or paste a Google Drive
   folder link.
3. **Run Import** -- shows live Total/Processed/Skipped/Failed/Progress %,
   then offers the import report for download.
4. **Candidate Dashboard** -- every candidate ever imported (from
   `Applicants.xlsx`), with filters and summary metrics (Total Applicants,
   Shortlisted, Rejected, Average ATS Score, Top Skills).

## `Applicants.xlsx` schema

`Applicants.xlsx` is never overwritten wholesale: each import run appends new
candidates and updates existing ones (matched by email, phone, or an exact
resume-content hash), leaving every other row untouched.

Columns: Candidate ID, Name, Email, Phone, Experience, Skills, Education, ATS
Score, Skill Match %, Experience Match %, Missing Skills, Recommendation,
Status, Resume File Name, Google Drive File ID, Google Drive URL, Imported
Time.

(The workbook also has a `_dedup_index` sheet mapping Candidate ID -> resume
content hash -- internal bookkeeping for exact-duplicate detection, not part
of the visible schema.)

## ATS scoring

- **Skill Match %** -- required skills (from the JD) found in the resume,
  matched exactly, fuzzily (typo-tolerant), or via a plain text search.
- **Experience Match %** -- candidate's total years of experience vs. the
  JD's minimum, capped at 100%.
- **ATS Score (0-100)** -- 60% skill match + 25% experience match + 15%
  semantic similarity (if enabled; otherwise the 15% is redistributed across
  the other two).
- **Recommendation** -- Shortlist (>=75), Maybe (50-74), Reject (<50) by
  default; both thresholds are adjustable in the dashboard.

## Known limitations

- **Legacy `.doc` files**: there's no reliable pure-Python parser for the old
  binary Word format. The pipeline makes a best effort and otherwise logs a
  clear failure asking for a `.docx` or PDF re-save, without aborting the rest
  of the batch.
- **Parsing is heuristic, not NLP-based**: field extraction relies on regexes
  and common resume conventions (section headers, date ranges, etc.). Unusual
  resume layouts may leave some fields blank -- by design, a missing field is
  preferred over a wrong guess.
- **Unauthenticated Drive access (`gdown`)** is rate-limited by Google under
  repeated use; configure a Drive API key (above) for reliable imports.

## Testing

```bash
python -m unittest discover -v
```

Covers: Drive URL parsing and all three Drive backends (mocked), text
extraction and field parsing (PDF/DOCX/DOC, including corrupt files), ATS
scoring math and thresholds, the `Applicants.xlsx` upsert/dedup logic, and the
full import pipeline end-to-end (public folder, duplicate resumes, invalid
PDF, DOCX mix, empty folder, and a 100-resume batch).

## Legacy CLI (`main.py`)

For a quick local-folder keyword scan without Drive import or full ATS
scoring:
```bash
python main.py --folder "path/to/resumes" --keywords "Python, SQL, AWS" --threshold 0.6
```
See `--help` for all options. Output: `shortlisting_report.xlsx`,
`shortlisted/` folder, `logs/processed_files.txt`, `ats_tool.log`.

## License

MIT
