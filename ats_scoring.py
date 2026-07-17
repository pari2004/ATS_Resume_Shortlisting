"""ATS scoring engine: matches a parsed resume against a job description and
produces a 0-100 score, skill/experience match percentages, and a
Shortlist/Maybe/Reject recommendation.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from rapidfuzz import fuzz

from resume_parser import ParsedResume
from skills_taxonomy import SKILLS_TAXONOMY_SORTED

FUZZY_MATCH_THRESHOLD = 85.0

SHORTLIST_THRESHOLD = 75.0
MAYBE_THRESHOLD = 50.0

SKILL_MATCH_WEIGHT = 0.60
EXPERIENCE_MATCH_WEIGHT = 0.25
SEMANTIC_WEIGHT = 0.15


@dataclass
class JobDescription:
    raw_text: str = ""
    required_skills: List[str] = field(default_factory=list)
    min_experience_years: float = 0.0


@dataclass
class ATSResult:
    ats_score: float = 0.0
    skill_match_pct: float = 0.0
    experience_match_pct: float = 0.0
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendation: str = "Reject"


def extract_skills_from_jd(jd_text: str) -> List[str]:
    """Auto-extracts required skills from a pasted job description by
    cross-referencing the shared skills taxonomy -- the same list the resume
    parser uses, so JD skills and resume skills are directly comparable."""
    haystack = jd_text.lower()
    found = []
    for skill in SKILLS_TAXONOMY_SORTED:
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, haystack):
            found.append(skill)
    return sorted(set(found))


def _skill_matches(required_skill: str, candidate_skills: List[str], resume_text: str) -> bool:
    if required_skill in candidate_skills:
        return True
    for skill in candidate_skills:
        if fuzz.ratio(required_skill, skill) >= FUZZY_MATCH_THRESHOLD:
            return True
    pattern = r"(?<![a-z0-9])" + re.escape(required_skill) + r"(?![a-z0-9])"
    return bool(re.search(pattern, resume_text.lower()))


def _score_skills(parsed: ParsedResume, jd: JobDescription) -> tuple:
    if not jd.required_skills:
        return 100.0, [], []

    matched, missing = [], []
    for skill in jd.required_skills:
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


_semantic_model = None


def _semantic_similarity(resume_text: str, jd_text: str) -> Optional[float]:
    """Cosine similarity between resume and JD text via sentence-transformers.
    Lazily loads the model on first use (~90MB download the first time) and
    returns None if it can't be loaded, so callers can fall back gracefully
    instead of failing the whole scoring run."""
    global _semantic_model
    if not resume_text.strip() or not jd_text.strip():
        return None
    try:
        if _semantic_model is None:
            from sentence_transformers import SentenceTransformer, util
            _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            from sentence_transformers import util

        embeddings = _semantic_model.encode([resume_text, jd_text], convert_to_tensor=True)
        score = util.cos_sim(embeddings[0], embeddings[1]).item()
        return max(0.0, min(1.0, score)) * 100.0
    except Exception:
        return None


def score_resume(
    parsed: ParsedResume,
    jd: JobDescription,
    use_semantic: bool = False,
    shortlist_threshold: float = SHORTLIST_THRESHOLD,
    maybe_threshold: float = MAYBE_THRESHOLD,
) -> ATSResult:
    skill_pct, matched, missing = _score_skills(parsed, jd)
    experience_pct = _score_experience(parsed, jd)

    semantic_pct = None
    if use_semantic:
        semantic_pct = _semantic_similarity(parsed.raw_text, jd.raw_text)

    if semantic_pct is not None:
        ats_score = (
            SKILL_MATCH_WEIGHT * skill_pct
            + EXPERIENCE_MATCH_WEIGHT * experience_pct
            + SEMANTIC_WEIGHT * semantic_pct
        )
    else:
        # Redistribute the semantic weight across skills/experience so the
        # score still sums to a full 0-100 scale without it.
        remaining = SKILL_MATCH_WEIGHT + EXPERIENCE_MATCH_WEIGHT
        ats_score = (
            (SKILL_MATCH_WEIGHT / remaining) * skill_pct
            + (EXPERIENCE_MATCH_WEIGHT / remaining) * experience_pct
        )
    ats_score = round(ats_score, 1)

    strengths = [f"Matches required skill: {s}" for s in matched]
    if parsed.certifications:
        strengths.append(f"{len(parsed.certifications)} certification(s) listed")
    if parsed.total_experience_years >= jd.min_experience_years > 0:
        strengths.append(
            f"Meets minimum experience requirement ({parsed.total_experience_years} yrs)"
        )

    weaknesses = [f"Missing required skill: {s}" for s in missing]
    if jd.min_experience_years > 0 and parsed.total_experience_years < jd.min_experience_years:
        shortfall = round(jd.min_experience_years - parsed.total_experience_years, 1)
        weaknesses.append(
            f"{shortfall} year(s) short of the {jd.min_experience_years} required"
        )

    if ats_score >= shortlist_threshold:
        recommendation = "Shortlist"
    elif ats_score >= maybe_threshold:
        recommendation = "Maybe"
    else:
        recommendation = "Reject"

    return ATSResult(
        ats_score=ats_score,
        skill_match_pct=skill_pct,
        experience_match_pct=experience_pct,
        matched_skills=matched,
        missing_skills=missing,
        strengths=strengths,
        weaknesses=weaknesses,
        recommendation=recommendation,
    )
