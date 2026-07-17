"""Resume text extraction (PDF/DOCX/DOC) and structured field parsing.

Field extraction here is regex/heuristic-based on purpose (no NLP model
download, works fully offline, fast enough for 100+ resume batches). It is
best-effort: resumes are wildly inconsistent in layout, so fields that can't
be confidently found are left empty/zero rather than guessed wrong.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pdfplumber
import pytesseract
from pdf2image import convert_from_path

import domain_taxonomy

try:
    import phonenumbers
except ImportError:  # pragma: no cover - dependency is in requirements.txt
    phonenumbers = None

logger = logging.getLogger("drive_utils")  # share the ats_tool.log handler

SUPPORTED_TEXT_EXTENSIONS = {".pdf", ".docx", ".doc"}

# pdfplumber renders icon-font glyphs (FontAwesome bullets for phone/email/
# LinkedIn icons in templated resumes) that have no Unicode mapping as
# literal "(cid:131)" placeholders. They're pure noise for every downstream
# consumer (name/section parsing), so strip them right at extraction time.
_CID_ARTIFACT_RE = re.compile(r"\(cid:\d+\)")


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_text_from_pdf_path(pdf_path: Path) -> Tuple[str, bool]:
    """Extracts text from a PDF already on disk (text layer first, OCR
    fallback for scanned PDFs). Shared by the Drive-import pipeline and the
    legacy CLI tool (``main.py``) so there's one implementation of this."""
    text = ""
    is_scanned = False

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if len(text.strip()) == 0:
            logger.info(f"No text layer in {pdf_path.name}, trying OCR...")
            is_scanned = True
            images = convert_from_path(str(pdf_path))
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"

        return _CID_ARTIFACT_RE.sub(" ", text), is_scanned
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path.name}: {e}")
        return "", True


def extract_text_from_pdf_bytes(pdf_bytes: bytes, tmp_path: Path) -> Tuple[str, bool]:
    """Same as :func:`extract_text_from_pdf_path`, for PDF content that only
    exists in memory (e.g. just downloaded from Drive) -- stages it to
    ``tmp_path`` first since pdfplumber/pdf2image need a real file path."""
    tmp_path.write_bytes(pdf_bytes)
    return extract_text_from_pdf_path(tmp_path)


def extract_text_from_docx_bytes(docx_bytes: bytes, tmp_path: Path) -> str:
    import docx2txt

    tmp_path.write_bytes(docx_bytes)
    try:
        return docx2txt.process(str(tmp_path)) or ""
    except Exception as e:
        logger.error(f"Error extracting text from {tmp_path.name}: {e}")
        return ""


def extract_text_from_legacy_doc_bytes(doc_bytes: bytes, tmp_path: Path) -> str:
    """Legacy binary .doc has no reliable pure-Python parser. Best effort:
    try docx2txt (works for some mislabeled .doc files that are actually
    .docx), then fall back to a crude latin-1 strings scan so at least some
    plain text can be pulled out. Returns "" if nothing usable is found --
    callers should treat that as a failed/skip-worthy file."""
    tmp_path.write_bytes(doc_bytes)

    import docx2txt
    try:
        text = docx2txt.process(str(tmp_path)) or ""
        if text.strip():
            return text
    except Exception:
        pass

    try:
        raw = doc_bytes.decode("latin-1", errors="ignore")
        printable = re.findall(r"[ -~]{4,}", raw)
        text = "\n".join(printable)
        # Legacy .doc binary streams are mostly non-text control data; a
        # short salvage means this heuristic didn't find real content.
        return text if len(text) > 200 else ""
    except Exception as e:
        logger.error(f"Error salvaging text from legacy .doc {tmp_path.name}: {e}")
        return ""


def extract_text(file_bytes: bytes, filename: str, staging_dir: Path) -> Tuple[str, bool, bool]:
    """Returns (text, is_scanned, extraction_ok). extraction_ok is False when
    no usable text could be pulled out at all (caller should count this file
    as Failed but keep processing the rest of the batch)."""
    ext = Path(filename).suffix.lower()
    # A filename containing "/" or "\" (rare, but possible from an upload or
    # an unusual Drive file name) would otherwise be interpreted as a
    # sub-path and crash when staged to disk -- flatten it to a single
    # path segment.
    safe_filename = re.sub(r"[\\/]+", "_", filename)
    tmp_path = staging_dir / safe_filename

    if ext == ".pdf":
        text, is_scanned = extract_text_from_pdf_bytes(file_bytes, tmp_path)
        return text, is_scanned, bool(text.strip())

    if ext == ".docx":
        text = extract_text_from_docx_bytes(file_bytes, tmp_path)
        return text, False, bool(text.strip())

    if ext == ".doc":
        text = extract_text_from_legacy_doc_bytes(file_bytes, tmp_path)
        if not text.strip():
            logger.warning(
                f"{filename}: legacy .doc format could not be read. Please "
                "re-save as .docx or PDF and re-upload."
            )
        return text, False, bool(text.strip())

    return "", False, False


# ---------------------------------------------------------------------------
# Structured field extraction
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_CANDIDATE_RE = re.compile(r"(\+?\d[\d\-\.\s\(\)]{7,16}\d)")
_LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/\S+", re.IGNORECASE)
_GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/\S+", re.IGNORECASE)
_GENERIC_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(?:/\S*)?")
_YEARS_EXPERIENCE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?experience", re.IGNORECASE
)

_MONTHS = (
    "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    "january|february|march|april|june|july|august|september|october|november|december"
)
_DATE_RANGE_RE = re.compile(
    rf"(?:({_MONTHS})[a-z]*\.?\s*)?(\d{{4}})\s*(?:-|to|–|—)\s*"
    rf"(?:(?:({_MONTHS})[a-z]*\.?\s*)?(\d{{4}})|present|current)",
    re.IGNORECASE,
)
# Numeric equivalent, e.g. "06/2025 - 07/2025" or "03/2023 - Present" --
# common on internship-heavy/early-career resumes that skip month names.
_NUMERIC_DATE_RANGE_RE = re.compile(
    r"(\d{1,2})/(\d{4})\s*(?:-|to|–|—)\s*(?:(\d{1,2})/(\d{4})|present|current)",
    re.IGNORECASE,
)
# Last-resort fallback when no date range is found at all -- a bare stated
# duration like "(5 months)" or "2 months" with no calendar dates, common on
# resumes that list internship durations without start/end dates.
_DURATION_PHRASE_RE = re.compile(
    r"(\d+)\s*\+?\s*years?(?:\s+(\d+)\s*months?)?|(\d+)\s*\+?\s*months?",
    re.IGNORECASE,
)

_SECTION_ALIASES = {
    "skills": ["skills", "technical skills", "core competencies", "key skills"],
    "experience": [
        "experience", "work experience", "professional experience",
        "employment history", "work history", "internship experience",
        "internships", "internship",
    ],
    "education": ["education", "academic background", "academic qualifications"],
    "certifications": ["certifications", "certificates", "licenses"],
    "projects": ["projects", "personal projects", "academic projects"],
    "summary": ["summary", "objective", "profile", "about"],
    "languages": ["languages", "language proficiency", "languages known"],
    "achievements": ["achievements", "accomplishments", "awards", "honors"],
}


def _split_sections(text: str) -> dict:
    """Splits resume text into named sections based on common header lines.
    A "line" is treated as a header if it's short, and matches (loosely) one
    of the known section names. Returns {section_name: block_text}; any text
    before the first recognized header is not attributed to any section."""
    lines = text.split("\n")
    header_positions = []  # (line_index, section_name)

    alias_to_section = {}
    for section, aliases in _SECTION_ALIASES.items():
        for alias in aliases:
            alias_to_section[alias] = section

    for i, line in enumerate(lines):
        stripped = line.strip().lower().strip(":").strip()
        if not stripped or len(stripped) > 40:
            continue
        if stripped in alias_to_section:
            header_positions.append((i, alias_to_section[stripped]))

    sections = {}
    for idx, (line_no, section) in enumerate(header_positions):
        end = header_positions[idx + 1][0] if idx + 1 < len(header_positions) else len(lines)
        block = "\n".join(lines[line_no + 1:end]).strip()
        if section in sections:
            sections[section] += "\n" + block
        else:
            sections[section] = block

    return sections


def _extract_email(text: str) -> str:
    match = _EMAIL_RE.search(text)
    return match.group(0) if match else ""


def _extract_phone(text: str) -> str:
    for candidate in _PHONE_CANDIDATE_RE.findall(text):
        digits = re.sub(r"\D", "", candidate)
        if len(digits) < 7 or len(digits) > 15:
            continue
        if phonenumbers:
            for region in ("IN", "US", None):
                try:
                    parsed = phonenumbers.parse(candidate, region)
                    if phonenumbers.is_valid_number(parsed):
                        return phonenumbers.format_number(
                            parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL
                        )
                except Exception:
                    continue
        return candidate.strip()
    return ""


def _extract_link(text: str, pattern: re.Pattern) -> str:
    match = pattern.search(text)
    if not match:
        return ""
    url = match.group(0).rstrip(".,)")
    if not url.lower().startswith("http"):
        url = "https://" + url
    return url


_ADDRESS_LABEL_RE = re.compile(r"address\s*[:\-]\s*(.+)", re.IGNORECASE)
_ZIP_OR_STATE_LINE_RE = re.compile(
    r"^[A-Za-z .,'-]+,\s*[A-Za-z]{2,}\s*[- ]?\s*\d{4,6}$|"  # City, ST 12345
    r"^[A-Za-z .,'-]+,\s*[A-Za-z .]+,\s*(India|USA|United States|UK)$",
    re.IGNORECASE,
)


def _extract_address(text: str) -> str:
    label_match = _ADDRESS_LABEL_RE.search(text)
    if label_match:
        return label_match.group(1).strip()[:120]

    for line in text.split("\n")[:20]:
        line = line.strip()
        if 5 < len(line) <= 100 and _ZIP_OR_STATE_LINE_RE.match(line):
            return line

    return ""


def _extract_portfolio(text: str, linkedin: str, github: str) -> str:
    for line in text.split("\n"):
        if re.search(r"portfolio|personal\s*website", line, re.IGNORECASE):
            match = _GENERIC_URL_RE.search(line)
            if match:
                url = match.group(0).rstrip(".,)")
                if "linkedin.com" in url.lower() or "github.com" in url.lower():
                    continue
                if not url.lower().startswith("http"):
                    url = "https://" + url
                return url
    return ""


# Words that show up capitalized/title-cased in resumes but are never a
# person's name -- section headers ("PROFESSIONAL EXPERIENCE"), job-title
# fragments folded into filenames ("Aarif_FullStackDeveloper.pdf"), etc.
# Checked case-insensitively per word, not per substring, so it won't reject
# an actual name that merely contains one of these as a substring.
_NAME_BLOCKLIST = {
    "resume", "cv", "curriculum", "vitae", "experience", "professional",
    "education", "skills", "summary", "objective", "profile", "about",
    "projects", "certifications", "certificate", "contact", "details",
    "personal", "career", "work", "employment", "history", "references",
    "address", "phone", "email", "linkedin", "github", "portfolio",
    "achievements", "awards", "languages", "interests", "hobbies",
    "declaration", "technical", "academic", "overview", "qualifications",
    "employer", "position", "designation", "responsibilities", "present",
    "developer", "engineer", "manager", "analyst", "consultant", "intern",
    "senior", "junior", "lead", "fullstack", "frontend", "backend",
    "full", "stack", "mern", "mean", "lamp", "web", "mobile", "cloud",
    "freelancer", "fresher", "student", "graduate", "trainee",
    "se", "sde", "swe", "qa", "ui", "ux", "hr", "po", "pm",
    "cto", "ceo", "cfo", "coo", "vp",
}

# Strict "person name" token used only for FILENAME parsing (kept narrow --
# filenames are noisier and more likely to contain bare acronyms like "SE"
# or "V2" that shouldn't be mistaken for initials): a proper Titlecase word,
# or a single initial letter with an optional trailing period.
_FILENAME_NAME_TOKEN_RE = re.compile(r"^(?:[A-Z][a-z]+(?:['-][A-Za-z]+)*|[A-Z]\.?)$")


def _looks_like_name_word(word: str) -> bool:
    """Shape check for a single word found in resume BODY TEXT (looser than
    the filename check): either a normal Titlecase word, or an ALL-CAPS token
    (covers resumes that print the name in caps, e.g. "AARIF KHAN", and
    multi-letter initial clusters like "M.V."). Blocklist filtering happens
    separately wherever this is used."""
    if not word or not word[0].isalpha() or not word[0].isupper():
        return False
    if word.isupper():
        return True
    return bool(re.match(r"^[A-Z][a-z'-]*\.?$", word))


def _leading_name_run(line: str, max_words: int = 3) -> str:
    """Pulls a leading run of name-shaped words off a line that may have a
    job title or other content tacked on afterwards (e.g. "Pankaj Yadav Full
    Stack Developer | MERN Stack | ..."), stopping at the first word that
    doesn't look like part of a name or is blocklisted."""
    run = []
    for raw in line.replace("|", " ").split():
        core = raw.strip(",;:|()[]")
        if not core:
            break
        if core.rstrip(".").lower() in _NAME_BLOCKLIST:
            break
        if not _looks_like_name_word(core):
            break
        run.append(core)
        if len(run) >= max_words:
            break
    return " ".join(run)


def _line_is_full_name(line: str) -> bool:
    words = [w.strip(",;:|()[]") for w in line.split()]
    words = [w for w in words if w]
    if not (2 <= len(words) <= 4):
        return False
    if any(w.rstrip(".").lower() in _NAME_BLOCKLIST for w in words):
        return False
    return all(_looks_like_name_word(w) for w in words)


def _extract_name(filename: str, text: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[_\-\s]+(resume|cv)\b", "", stem, flags=re.IGNORECASE)
    stem_parts = [p for p in re.split(r"[_\-\s]+", stem) if p.strip()]

    leading_name_parts = []
    for part in stem_parts:
        if part.lower() in _NAME_BLOCKLIST or not _FILENAME_NAME_TOKEN_RE.match(part):
            break
        leading_name_parts.append(part)
    if 2 <= len(leading_name_parts) <= 4:
        return " ".join(leading_name_parts)

    # Lines with contact info stripped out, keeping only ones that still
    # have content afterwards -- e.g. a lone phone number up top shouldn't
    # count as "the first line" for name-detection purposes, and "Name
    # name@example.com" on one line should still expose the name part.
    meaningful = []
    for raw_line in text.split("\n")[:15]:
        line = raw_line.strip()
        if not line or len(line) > 80:
            continue
        cleaned = _EMAIL_RE.sub(" ", line)
        cleaned = _PHONE_CANDIDATE_RE.sub(" ", cleaned).strip()
        if cleaned:
            meaningful.append(cleaned)

    for i, line in enumerate(meaningful):
        if _line_is_full_name(line):
            return " ".join(w.strip(",;:|()[]") for w in line.split() if w.strip(",;:|()[]"))

        if i == 0:
            run = _leading_name_run(line)
            run_word_count = len(run.split()) if run else 0
            if run_word_count >= 2:
                return run
            if run_word_count == 1:
                # Some resumes print "FIRST\nLAST" as two separate header
                # lines -- if the very next line is itself a single bare
                # name-shaped word, treat it as the continuation.
                next_line = meaningful[1] if len(meaningful) > 1 else ""
                next_word = next_line.strip(",;:|()[]")
                if (
                    next_word and " " not in next_word
                    and not re.search(r"\d", next_word)
                    and next_word.rstrip(".").lower() not in _NAME_BLOCKLIST
                    and _looks_like_name_word(next_word)
                ):
                    return f"{run} {next_word}"
                return run

    return "Unknown"


def _extract_skills(text: str, skills_section: str) -> List[str]:
    """Domain-independent: pulls verbatim items straight out of a labeled
    Skills section first (works for any domain, not just what's in the
    taxonomy), unioned with taxonomy cross-references over the full text.
    See domain_taxonomy.extract_skill_phrases for the layered strategy."""
    return domain_taxonomy.extract_skill_phrases(text)


def _parse_list_section(block: str) -> List[str]:
    if not block.strip():
        return []
    items = []
    for line in block.split("\n"):
        line = line.strip(" \t-•*·").strip()
        if line:
            items.append(line)
    return items


def _months_between(y1: int, m1: int, y2: int, m2: int) -> int:
    return max(0, (y2 - y1) * 12 + (m2 - m1))


def _month_to_num(month_str: Optional[str]) -> int:
    if not month_str:
        return 1
    month_str = month_str.lower()[:3]
    order = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    return (order.index(month_str) + 1) if month_str in order else 1


def _extract_experience_entries(experience_section: str) -> List[dict]:
    entries = []
    for match in _DATE_RANGE_RE.finditer(experience_section):
        start_month, start_year, end_month, end_year = match.groups()
        end_label = match.group(0).lower()
        is_current = "present" in end_label or "current" in end_label
        entries.append({
            "raw": match.group(0),
            "start_year": int(start_year),
            "start_month": _month_to_num(start_month),
            "end_year": int(end_year) if end_year else datetime.now().year,
            "end_month": _month_to_num(end_month) if end_month else datetime.now().month,
            "is_current": is_current,
        })
    for match in _NUMERIC_DATE_RANGE_RE.finditer(experience_section):
        start_month, start_year, end_month, end_year = match.groups()
        end_label = match.group(0).lower()
        is_current = "present" in end_label or "current" in end_label
        entries.append({
            "raw": match.group(0),
            "start_year": int(start_year),
            "start_month": int(start_month),
            "end_year": int(end_year) if end_year else datetime.now().year,
            "end_month": int(end_month) if end_month else datetime.now().month,
            "is_current": is_current,
        })
    return entries


def _sum_duration_phrases(text: str) -> float:
    total_months = 0
    for match in _DURATION_PHRASE_RE.finditer(text):
        years, months_combo, months_alone = match.groups()
        if years is not None:
            total_months += int(years) * 12 + (int(months_combo) if months_combo else 0)
        elif months_alone is not None:
            total_months += int(months_alone)
    return round(total_months / 12.0, 1) if total_months else 0.0


def _total_experience_years(text: str, experience_section: str) -> float:
    entries = _extract_experience_entries(experience_section)
    if entries:
        total_months = sum(
            _months_between(e["start_year"], e["start_month"], e["end_year"], e["end_month"])
            for e in entries
        )
        if total_months > 0:
            return round(total_months / 12.0, 1)

    match = _YEARS_EXPERIENCE_RE.search(text)
    if match:
        return float(match.group(1))

    # Last resort: resumes that state a bare duration ("2 months", "(5
    # months)") instead of calendar dates -- scoped to the experience
    # section (falls back to the whole resume if no section was found) to
    # limit false positives from unrelated year/month mentions elsewhere.
    duration_years = _sum_duration_phrases(experience_section)
    if duration_years > 0:
        return duration_years

    return 0.0


def _current_role(experience_section: str) -> Tuple[str, str]:
    """Best-effort (current_company, current_designation) from the first
    non-empty lines of the most recent experience entry (the one containing
    'Present'/'Current', or simply the first entry if none say so -- resumes
    conventionally list the most recent role first)."""
    if not experience_section.strip():
        return "", ""

    blocks = re.split(r"\n\s*\n", experience_section.strip())
    target_block = blocks[0]
    for block in blocks:
        if re.search(r"present|current", block, re.IGNORECASE):
            target_block = block
            break

    lines = [l.strip(" \t-•*·") for l in target_block.split("\n") if l.strip()]
    if not lines:
        return "", ""

    header = lines[0]
    # Common formats: "Title, Company" / "Title at Company" / "Title - Company"
    for sep in [" at ", " @ ", " - ", ", ", " | "]:
        if sep in header:
            parts = [p.strip() for p in header.split(sep, 1)]
            if len(parts) == 2:
                return parts[1], parts[0]

    designation = header
    company = lines[1] if len(lines) > 1 else ""
    return company, designation


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class ParsedResume:
    file_name: str = ""
    name: str = "Unknown"
    email: str = ""
    phone: str = ""
    address: str = ""
    skills: List[str] = field(default_factory=list)
    experience_entries: List[str] = field(default_factory=list)
    education: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)
    linkedin: str = ""
    github: str = ""
    portfolio: str = ""
    total_experience_years: float = 0.0
    current_company: str = ""
    current_designation: str = ""
    raw_text: str = ""
    resume_hash: str = ""
    is_scanned: bool = False
    extraction_ok: bool = True


def compute_resume_hash(raw_text: str) -> str:
    normalized = re.sub(r"\s+", " ", raw_text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def parse_resume(text: str, filename: str, is_scanned: bool = False) -> ParsedResume:
    sections = _split_sections(text)
    skills_section = sections.get("skills", "")
    experience_section = sections.get("experience", text)

    linkedin = _extract_link(text, _LINKEDIN_RE)
    github = _extract_link(text, _GITHUB_RE)

    return ParsedResume(
        file_name=filename,
        name=_extract_name(filename, text),
        email=_extract_email(text),
        phone=_extract_phone(text),
        address=_extract_address(text),
        skills=_extract_skills(text, skills_section),
        experience_entries=_parse_list_section(experience_section),
        education=_parse_list_section(sections.get("education", "")),
        certifications=_parse_list_section(sections.get("certifications", "")),
        projects=_parse_list_section(sections.get("projects", "")),
        languages=_parse_list_section(sections.get("languages", "")),
        achievements=_parse_list_section(sections.get("achievements", "")),
        linkedin=linkedin,
        github=github,
        portfolio=_extract_portfolio(text, linkedin, github),
        total_experience_years=_total_experience_years(text, experience_section),
        current_company=_current_role(experience_section)[0],
        current_designation=_current_role(experience_section)[1],
        raw_text=text,
        resume_hash=compute_resume_hash(text),
        is_scanned=is_scanned,
        extraction_ok=bool(text.strip()),
    )
