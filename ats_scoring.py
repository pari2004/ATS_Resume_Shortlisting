"""ATS scoring engine: parses a job description into structured fields
(domain, required/preferred skills, experience, education, certifications)
and scores a parsed resume against it with a configurable 6-factor weighted
formula, producing a 0-100 score, a Shortlist/Maybe/Reject recommendation,
and a domain-aware recommendation sentence.

Nothing here is hardcoded to a specific job domain -- the Job Description is
the source of truth for what "skills" means on a given import run. See
domain_taxonomy.py for the domain classifier and extraction layers.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from rapidfuzz import fuzz

import domain_taxonomy as dt
from resume_parser import ParsedResume

FUZZY_MATCH_THRESHOLD = 85.0

SHORTLIST_THRESHOLD = 75.0
MAYBE_THRESHOLD = 50.0

_EDUCATION_LEVEL_RANK = {name: i for i, (name, _kw) in enumerate(dt.EDUCATION_LEVEL_KEYWORDS)}


@dataclass
class ScoringWeights:
    required_skills: float = 0.40
    preferred_skills: float = 0.20
    experience: float = 0.15
    education: float = 0.10
    certifications: float = 0.10
    resume_quality: float = 0.05

    def normalized(self) -> "ScoringWeights":
        total = (
            self.required_skills + self.preferred_skills + self.experience
            + self.education + self.certifications + self.resume_quality
        )
        if total <= 0:
            return ScoringWeights()
        return ScoringWeights(
            required_skills=self.required_skills / total,
            preferred_skills=self.preferred_skills / total,
            experience=self.experience / total,
            education=self.education / total,
            certifications=self.certifications / total,
            resume_quality=self.resume_quality / total,
        )


@dataclass
class JobDescription:
    raw_text: str = ""
    title: str = ""
    domain: str = "General / Other"
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    soft_skills: List[str] = field(default_factory=list)
    min_experience_years: float = 0.0
    preferred_experience_years: float = 0.0
    education_requirement: str = ""
    certifications_required: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)


@dataclass
class ATSResult:
    ats_score: float = 0.0
    required_skill_match_pct: float = 0.0
    preferred_skill_match_pct: float = 0.0
    experience_match_pct: float = 0.0
    education_match_pct: float = 0.0
    certifications_match_pct: float = 0.0
    resume_quality_pct: float = 0.0
    matched_required_skills: List[str] = field(default_factory=list)
    missing_required_skills: List[str] = field(default_factory=list)
    matched_preferred_skills: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendation: str = "Reject"
    recommendation_text: str = ""

    # Backward-compatible aliases (older callers/tests used the singular
    # "skill" naming before required/preferred were split out).
    @property
    def skill_match_pct(self) -> float:
        return self.required_skill_match_pct

    @property
    def matched_skills(self) -> List[str]:
        return self.matched_required_skills

    @property
    def missing_skills(self) -> List[str]:
        return self.missing_required_skills


# ---------------------------------------------------------------------------
# Job description analysis
# ---------------------------------------------------------------------------

_TITLE_LABEL_RE = re.compile(r"^(?:job title|position|role)\s*[:\-]\s*(.+)$", re.IGNORECASE)

_PREFERRED_SECTION_ALIASES = {
    "preferred skills", "preferred qualifications", "nice to have",
    "good to have", "bonus points", "preferred",
}
_REQUIRED_SECTION_ALIASES = {
    "required skills", "must have", "must-have", "requirements",
    "required qualifications", "qualifications", "skills", "technical skills",
}

_EXPERIENCE_RANGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*\+?\s*years?", re.IGNORECASE
)
_EXPERIENCE_MIN_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?experience", re.IGNORECASE
)

_CERT_MENTION_RE = re.compile(r"([A-Za-z][A-Za-z0-9 .&/+-]{2,40}?\bcertifi(?:ed|cation)\b)", re.IGNORECASE)

_RESPONSIBILITIES_ALIASES = ["responsibilities", "what you'll do", "what you will do", "role & responsibilities", "key responsibilities"]


def _extract_title(jd_text: str) -> str:
    for line in jd_text.split("\n")[:5]:
        line = line.strip()
        if not line:
            continue
        label_match = _TITLE_LABEL_RE.match(line)
        if label_match:
            return label_match.group(1).strip()
        if len(line) <= 60:
            return line
        break
    return ""


def _split_required_preferred(jd_text: str) -> Tuple[List[str], List[str]]:
    preferred_blocks = dt._find_labeled_sections(jd_text, list(_PREFERRED_SECTION_ALIASES))
    required_blocks = dt._find_labeled_sections(jd_text, list(_REQUIRED_SECTION_ALIASES))

    preferred_items = set()
    for block in preferred_blocks:
        preferred_items.update(dt._split_verbatim_items(block))

    if required_blocks:
        required_items = set()
        for block in required_blocks:
            required_items.update(dt._split_verbatim_items(block))
        required_items -= preferred_items
    else:
        # No explicit required/must-have section -- fall back to the full
        # layered extraction (taxonomy + generic), minus anything already
        # claimed by a preferred section.
        required_items = set(dt.extract_skill_phrases(jd_text)) - preferred_items

    return sorted(required_items), sorted(preferred_items)


def _extract_soft_skills(jd_text: str) -> List[str]:
    lower = jd_text.lower()
    found = []
    for skill in dt.SOFT_SKILLS:
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, lower):
            found.append(skill)
    return sorted(found)


def _extract_experience_range(jd_text: str) -> Tuple[float, float]:
    range_match = _EXPERIENCE_RANGE_RE.search(jd_text)
    if range_match:
        lo, hi = float(range_match.group(1)), float(range_match.group(2))
        return min(lo, hi), max(lo, hi)
    min_match = _EXPERIENCE_MIN_RE.search(jd_text)
    if min_match:
        val = float(min_match.group(1))
        return val, val
    return 0.0, 0.0


def _extract_education_requirement(jd_text: str) -> str:
    return dt.education_level_of(jd_text)


def _extract_certifications(jd_text: str) -> List[str]:
    lower = jd_text.lower()
    found = set()
    for cert in dt.KNOWN_CERTIFICATIONS:
        if cert in lower:
            found.add(cert)
    for match in _CERT_MENTION_RE.finditer(jd_text):
        phrase = match.group(1).strip().lower()
        phrase = re.sub(r"\s+", " ", phrase)
        if len(phrase) <= 45:
            found.add(phrase)
    return sorted(found)


def _extract_responsibilities(jd_text: str) -> List[str]:
    lines: List[str] = []
    for block in dt._find_labeled_sections(jd_text, _RESPONSIBILITIES_ALIASES):
        for line in block.split("\n"):
            line = line.strip(" \t-•*·")
            if line:
                lines.append(line)
    return lines


def analyze_job_description(jd_text: str) -> JobDescription:
    """Single entry point for turning a pasted JD into a structured
    JobDescription. Every field is derived from the JD text itself (or left
    empty/zero) -- nothing here is specific to any one job domain."""
    jd_text = jd_text or ""
    domain = dt.detect_domain(jd_text)
    required_skills, preferred_skills = _split_required_preferred(jd_text)
    soft_skills = _extract_soft_skills(jd_text)
    min_exp, preferred_exp = _extract_experience_range(jd_text)
    education_requirement = _extract_education_requirement(jd_text)
    certifications_required = _extract_certifications(jd_text)
    responsibilities = _extract_responsibilities(jd_text)

    keywords = sorted(set(required_skills) | set(preferred_skills) | set(soft_skills))

    return JobDescription(
        raw_text=jd_text,
        title=_extract_title(jd_text),
        domain=domain,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        soft_skills=soft_skills,
        min_experience_years=min_exp,
        preferred_experience_years=preferred_exp,
        education_requirement=education_requirement,
        certifications_required=certifications_required,
        keywords=keywords,
        responsibilities=responsibilities,
    )


def extract_skills_from_jd(jd_text: str) -> List[str]:
    """Kept for backward compatibility (dashboard's manual "auto-extract
    skills" affordance) -- returns the union of required+preferred skills a
    full analyze_job_description() would find."""
    jd = analyze_job_description(jd_text)
    return sorted(set(jd.required_skills) | set(jd.preferred_skills))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _skill_matches(required_skill: str, candidate_skills: List[str], resume_text: str) -> bool:
    if required_skill in candidate_skills:
        return True
    for skill in candidate_skills:
        if fuzz.ratio(required_skill, skill) >= FUZZY_MATCH_THRESHOLD:
            return True
    pattern = r"(?<![a-z0-9])" + re.escape(required_skill) + r"(?![a-z0-9])"
    return bool(re.search(pattern, resume_text.lower()))


def _score_skill_list(required: List[str], parsed: ParsedResume) -> Tuple[float, List[str], List[str]]:
    if not required:
        return 100.0, [], []
    matched, missing = [], []
    for skill in required:
        skill_norm = skill.strip().lower()
        if not skill_norm:
            continue
        if _skill_matches(skill_norm, parsed.skills, parsed.raw_text):
            matched.append(skill_norm)
        else:
            missing.append(skill_norm)
    total = len(matched) + len(missing)
    pct = (len(matched) / total * 100.0) if total else 100.0
    return round(pct, 1), matched, missing


def _score_experience(parsed: ParsedResume, jd: JobDescription) -> float:
    if jd.min_experience_years <= 0:
        return 100.0
    pct = min(parsed.total_experience_years / jd.min_experience_years, 1.0) * 100.0
    return round(pct, 1)


def _score_education(parsed: ParsedResume, jd: JobDescription) -> float:
    if not jd.education_requirement:
        return 100.0
    required_rank = _EDUCATION_LEVEL_RANK.get(jd.education_requirement, -1)
    candidate_text = " ".join(parsed.education)
    candidate_level = dt.education_level_of(candidate_text)
    candidate_rank = _EDUCATION_LEVEL_RANK.get(candidate_level, -1)
    return 100.0 if candidate_rank >= required_rank else 0.0


def _score_certifications(parsed: ParsedResume, jd: JobDescription) -> Tuple[float, List[str], List[str]]:
    if not jd.certifications_required:
        return 100.0, [], []
    candidate_certs = [c.lower() for c in parsed.certifications]
    candidate_text = parsed.raw_text.lower()
    matched, missing = [], []
    for cert in jd.certifications_required:
        cert_norm = cert.strip().lower()
        if any(fuzz.partial_ratio(cert_norm, c) >= FUZZY_MATCH_THRESHOLD for c in candidate_certs) or cert_norm in candidate_text:
            matched.append(cert_norm)
        else:
            missing.append(cert_norm)
    total = len(matched) + len(missing)
    pct = (len(matched) / total * 100.0) if total else 100.0
    return round(pct, 1), matched, missing


def _score_resume_quality(parsed: ParsedResume) -> float:
    score = 0.0
    if parsed.email:
        score += 20.0
    if parsed.phone:
        score += 20.0
    if parsed.skills:
        score += 20.0
    if parsed.experience_entries:
        score += 20.0
    if parsed.education:
        score += 20.0
    return score


def _recommendation_sentence(domain: str, matched_required: List[str]) -> str:
    (tiers, noun) = dt.phrasing_for_domain(domain)
    top_skills = matched_required[:2]
    if not top_skills:
        return ""
    intensifier = tiers[0]
    if len(top_skills) == 1:
        skills_phrase = top_skills[0].title()
    else:
        skills_phrase = f"{top_skills[0].title()} and {top_skills[1].title()}"
    return f"{intensifier} {skills_phrase} {noun}"


def score_resume(
    parsed: ParsedResume,
    jd: JobDescription,
    weights: ScoringWeights = None,
    shortlist_threshold: float = SHORTLIST_THRESHOLD,
    maybe_threshold: float = MAYBE_THRESHOLD,
) -> ATSResult:
    weights = (weights or ScoringWeights()).normalized()

    required_pct, matched_required, missing_required = _score_skill_list(jd.required_skills, parsed)
    preferred_pct, matched_preferred, _missing_preferred = _score_skill_list(jd.preferred_skills, parsed)
    experience_pct = _score_experience(parsed, jd)
    education_pct = _score_education(parsed, jd)
    certifications_pct, _matched_certs, _missing_certs = _score_certifications(parsed, jd)
    resume_quality_pct = _score_resume_quality(parsed)

    ats_score = round(
        weights.required_skills * required_pct
        + weights.preferred_skills * preferred_pct
        + weights.experience * experience_pct
        + weights.education * education_pct
        + weights.certifications * certifications_pct
        + weights.resume_quality * resume_quality_pct,
        1,
    )

    strengths = [f"Matches required skill: {s}" for s in matched_required]
    strengths += [f"Matches preferred skill: {s}" for s in matched_preferred]
    if parsed.certifications:
        strengths.append(f"{len(parsed.certifications)} certification(s) listed")
    if parsed.total_experience_years >= jd.min_experience_years > 0:
        strengths.append(
            f"Meets minimum experience requirement ({parsed.total_experience_years} yrs)"
        )

    weaknesses = [f"Missing required skill: {s}" for s in missing_required]
    if jd.min_experience_years > 0 and parsed.total_experience_years < jd.min_experience_years:
        shortfall = round(jd.min_experience_years - parsed.total_experience_years, 1)
        weaknesses.append(
            f"{shortfall} year(s) short of the {jd.min_experience_years} required"
        )
    if jd.education_requirement and education_pct < 100:
        weaknesses.append(f"Does not clearly meet the {jd.education_requirement} requirement")

    if ats_score >= shortlist_threshold:
        recommendation = "Shortlist"
    elif ats_score >= maybe_threshold:
        recommendation = "Maybe"
    else:
        recommendation = "Reject"

    recommendation_text = _recommendation_sentence(jd.domain, matched_required + matched_preferred)

    return ATSResult(
        ats_score=ats_score,
        required_skill_match_pct=required_pct,
        preferred_skill_match_pct=preferred_pct,
        experience_match_pct=experience_pct,
        education_match_pct=education_pct,
        certifications_match_pct=certifications_pct,
        resume_quality_pct=resume_quality_pct,
        matched_required_skills=matched_required,
        missing_required_skills=missing_required,
        matched_preferred_skills=matched_preferred,
        strengths=strengths,
        weaknesses=weaknesses,
        recommendation=recommendation,
        recommendation_text=recommendation_text,
    )
