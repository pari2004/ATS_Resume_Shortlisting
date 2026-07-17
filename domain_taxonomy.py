"""Domain detection and domain-independent skill-phrase extraction.

The rest of the pipeline used to lean on a single fixed SKILLS_TAXONOMY to
decide what counts as a "skill" -- fine for software roles, weak for
anything else. This module makes the Job Description itself the primary
source of truth: `extract_skill_phrases` pulls literal terms straight out of
a labeled section (Skills/Requirements/Tools/...) before ever consulting the
taxonomy, so a Finance JD listing "GST, Tally, Bank Reconciliation" works
even though none of those are in SKILLS_TAXONOMY. The taxonomy and a generic
shape-based fallback only fill in when a text has no clean labeled section
at all (common in unstructured resumes).

No NLP model / network dependency -- consistent with the rest of this
project's offline-first design.
"""

import re
from typing import Dict, List

from skills_taxonomy import SKILLS_TAXONOMY_SORTED

# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "Full Stack Development": ["full stack", "fullstack", "full-stack developer"],
    "Frontend Development": ["frontend developer", "front-end developer", "front end developer", "ui developer"],
    "Backend Development": ["backend developer", "back-end developer", "back end developer", "server-side developer"],
    "Java Developer": ["java developer", "j2ee developer", "spring boot developer"],
    "Python Developer": ["python developer"],
    "React Developer": ["react developer", "reactjs developer"],
    "Node.js Developer": ["node.js developer", "nodejs developer", "node developer"],
    "AI Engineer": ["ai engineer", "artificial intelligence engineer", "genai engineer", "llm engineer"],
    "Machine Learning": ["machine learning engineer", "ml engineer", "machine learning"],
    "Data Scientist": ["data scientist", "data science"],
    "Data Analyst": ["data analyst", "data analytics"],
    "Business Analyst": ["business analyst", "business analysis"],
    "DevOps": ["devops engineer", "devops", "site reliability engineer", "sre"],
    "Cloud Engineer": ["cloud engineer", "cloud architect", "cloud administrator"],
    "Cyber Security": ["cyber security", "cybersecurity", "security analyst", "penetration tester", "soc analyst"],
    "UI/UX Designer": ["ui/ux designer", "ux designer", "ui designer", "product designer"],
    "Product Manager": ["product manager", "product management"],
    "Project Manager": ["project manager", "project management", "scrum master", "program manager"],
    "HR": ["human resources", "hr generalist", "hr manager", "hr executive"],
    "Recruiter": ["recruiter", "talent acquisition", "recruitment specialist"],
    "Sales": ["sales executive", "sales representative", "sales associate", "account executive"],
    "Business Development": ["business development", "bda", "bdm", "business development associate", "business development manager"],
    # "marketing manager" deliberately excluded here -- too ambiguous with
    # Digital Marketing in practice ("Digital Marketing Manager" would
    # otherwise tie on both).
    "Marketing": ["marketing executive", "brand marketing", "product marketing", "marketing communications"],
    "Digital Marketing": ["digital marketing", "performance marketing", "growth marketing"],
    "SEO": ["seo executive", "seo specialist", "search engine optimization"],
    "Finance": ["finance executive", "financial analyst", "finance manager"],
    "Banking": ["banking associate", "bank teller", "relationship manager banking"],
    "Accounting": ["accountant", "accounting executive", "bookkeeper", "accounts payable", "accounts receivable"],
    "Customer Support": ["customer support", "customer service representative", "customer care", "support executive"],
    "Operations": ["operations executive", "operations manager", "operations analyst"],
    "Healthcare": ["registered nurse", "physician", "healthcare", "medical officer", "clinical", "pharmacist"],
    "Mechanical Engineering": ["mechanical engineer", "mechanical design", "cad engineer"],
    "Civil Engineering": ["civil engineer", "site engineer", "structural engineer", "construction engineer"],
    "Electrical Engineering": ["electrical engineer", "electrical design engineer", "power systems engineer"],
}

_GENERAL_DOMAIN = "General / Other"


def detect_domain(jd_text: str) -> str:
    """Returns the best-matching domain name, or "General / Other" if
    nothing scores meaningfully. Hits in the first two non-empty lines
    (where a job title conventionally sits) count double."""
    if not jd_text or not jd_text.strip():
        return _GENERAL_DOMAIN

    lower_text = jd_text.lower()
    title_lines = [l.strip().lower() for l in jd_text.split("\n")[:2] if l.strip()]
    title_text = " ".join(title_lines)

    scores: Dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = 0
        for kw in keywords:
            pattern = r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])"
            if re.search(pattern, title_text):
                score += 2
            elif re.search(pattern, lower_text):
                score += 1
        if score:
            scores[domain] = score

    if not scores:
        return _GENERAL_DOMAIN
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Soft skills / education levels / shared vocabulary
# ---------------------------------------------------------------------------

SOFT_SKILLS = [
    "communication", "leadership", "teamwork", "collaboration", "adaptability",
    "time management", "problem solving", "critical thinking", "creativity",
    "attention to detail", "work ethic", "interpersonal skills",
    "conflict resolution", "decision making", "emotional intelligence",
    "presentation skills", "negotiation", "multitasking", "self motivated",
    "analytical thinking", "flexibility", "stakeholder management",
]

# Ordered lowest -> highest for education-requirement comparison. Every
# keyword is matched with word-boundary regex (see education_level_of()) --
# never substring-check these directly, short abbreviations like "ba"/"ma"
# would false-positive inside ordinary words ("manage", "banking").
EDUCATION_LEVEL_KEYWORDS: List[tuple] = [
    ("high school", ["high school", "hsc", "12th", "secondary school"]),
    ("diploma", ["diploma", "polytechnic"]),
    ("bachelor's", [
        "bachelor", "b.tech", "btech", "b.e", "b.sc", "bsc", "b.com",
        "bcom", "bba", "b.a", "ba", "undergraduate degree", "graduate",
    ]),
    ("master's", [
        "master", "m.tech", "mtech", "m.sc", "msc", "mba", "m.com", "mcom",
        "m.a", "ma", "postgraduate", "pg diploma",
    ]),
    ("phd", ["phd", "ph.d", "doctorate"]),
]


def education_level_of(text: str) -> str:
    """Returns the highest education level whose keyword appears in
    ``text`` (word-boundary matched, case-insensitive), or "" if none do."""
    lower = text.lower()
    best_rank = -1
    best_name = ""
    for rank, (name, keywords) in enumerate(EDUCATION_LEVEL_KEYWORDS):
        for kw in keywords:
            pattern = r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])"
            if re.search(pattern, lower):
                if rank > best_rank:
                    best_rank = rank
                    best_name = name
                break
    return best_name

KNOWN_CERTIFICATIONS = [
    "pmp", "aws certified", "azure certified", "google cloud certified",
    "six sigma", "cfa", "cpa", "shrm-cp", "shrm", "google ads certified",
    "hubspot certified", "salesforce certified", "scrum master", "csm",
    "itil", "ceh", "cissp", "comptia", "google analytics certified",
]

# Per-domain recommendation phrasing: (intensifier-by-tier, noun). Grouped by
# domain category rather than every single domain to keep this maintainable;
# falls back to a neutral default for anything not covered.
_TECH_PHRASING = (("Strong", "Solid", "Basic"), "experience")
_KNOWLEDGE_PHRASING = (("Strong", "Solid", "Basic"), "knowledge")
_SKILLS_PHRASING = (("Excellent", "Good", "Basic"), "skills")

DOMAIN_PHRASING: Dict[str, tuple] = {
    "Full Stack Development": _TECH_PHRASING,
    "Frontend Development": _TECH_PHRASING,
    "Backend Development": _TECH_PHRASING,
    "Java Developer": _TECH_PHRASING,
    "Python Developer": _TECH_PHRASING,
    "React Developer": _TECH_PHRASING,
    "Node.js Developer": _TECH_PHRASING,
    "AI Engineer": _TECH_PHRASING,
    "Machine Learning": _TECH_PHRASING,
    "Data Scientist": _TECH_PHRASING,
    "Data Analyst": _TECH_PHRASING,
    "DevOps": _TECH_PHRASING,
    "Cloud Engineer": _TECH_PHRASING,
    "Cyber Security": _TECH_PHRASING,
    "UI/UX Designer": (("Excellent", "Good", "Basic"), "design experience"),
    "Business Analyst": _SKILLS_PHRASING,
    "Product Manager": _SKILLS_PHRASING,
    "Project Manager": _SKILLS_PHRASING,
    "HR": _SKILLS_PHRASING,
    "Recruiter": _SKILLS_PHRASING,
    "Sales": _TECH_PHRASING,
    "Business Development": _TECH_PHRASING,
    "Marketing": (("Excellent", "Good", "Basic"), "experience"),
    "Digital Marketing": (("Excellent", "Good", "Basic"), "experience"),
    "SEO": (("Excellent", "Good", "Basic"), "experience"),
    "Finance": _KNOWLEDGE_PHRASING,
    "Banking": _KNOWLEDGE_PHRASING,
    "Accounting": _KNOWLEDGE_PHRASING,
    "Customer Support": _SKILLS_PHRASING,
    "Operations": _SKILLS_PHRASING,
    "Healthcare": _KNOWLEDGE_PHRASING,
    "Mechanical Engineering": _TECH_PHRASING,
    "Civil Engineering": _TECH_PHRASING,
    "Electrical Engineering": _TECH_PHRASING,
}
_DEFAULT_PHRASING = (("Strong", "Good", "Basic"), "skills")


def phrasing_for_domain(domain: str) -> tuple:
    return DOMAIN_PHRASING.get(domain, _DEFAULT_PHRASING)


# ---------------------------------------------------------------------------
# Domain-independent skill-phrase extraction
# ---------------------------------------------------------------------------

_SKILL_SECTION_ALIASES = [
    "skills", "required skills", "must have", "must-have", "requirements",
    "required qualifications", "qualifications", "preferred skills",
    "preferred qualifications", "nice to have", "good to have",
    "tools", "technologies", "tools & technologies", "tech stack",
    "core competencies", "key skills", "technical skills",
]

# Every section-header phrase this module (or ats_scoring's JD parser) knows
# how to recognize, used purely to bound *where a section ends* -- so a
# "Required Skills" block correctly stops at the next "Preferred Skills" (or
# "Education", "Certifications", "Responsibilities", ...) header instead of
# sweeping the rest of the document in with it. Kept as one master set that
# grows alongside whatever new categories get added.
_ALL_SECTION_HEADER_ALIASES = set(_SKILL_SECTION_ALIASES) | {
    "education", "education requirement", "qualification",
    "certifications", "certification", "certificates",
    "responsibilities", "what you'll do", "what you will do",
    "role & responsibilities", "key responsibilities",
    "about the role", "about the job", "job summary", "overview",
    "benefits", "perks", "compensation", "salary", "how to apply",
}

_SECTION_SPLIT_RE = re.compile(r"[,;|/••‣◦⁃∙\-\n]+")

_GENERIC_STOPWORDS = {
    "the", "and", "or", "with", "for", "a", "an", "of", "to", "in", "on",
    "we", "you", "our", "is", "are", "will", "be", "as", "this", "that",
    "role", "job", "candidate", "team", "company", "years", "year",
    "experience", "responsibilities", "requirements", "skills", "about",
    "who", "what", "looking", "seeking", "join", "work", "working",
}


def _find_labeled_sections(text: str, aliases: List[str]) -> List[str]:
    """Returns the text blocks following any header line matching one of
    ``aliases`` (case-insensitive, short lines only -- same convention as
    resume_parser._split_sections). Each block is bounded by the next
    recognized header of *any* known category (not just the same one), so a
    "Required Skills" section correctly stops at "Preferred Skills" /
    "Education" / etc. instead of sweeping the rest of the document in."""
    lines = text.split("\n")
    alias_set = {a.lower() for a in aliases}
    boundary_set = _ALL_SECTION_HEADER_ALIASES | alias_set

    blocks = []
    wanted_idx = []
    all_header_idx = []
    for i, line in enumerate(lines):
        stripped = line.strip().lower().strip(":").strip()
        if not stripped or len(stripped) > 40:
            continue
        # Inline "Label: value" lines (e.g. "Education: Bachelor's degree
        # required.") count as a boundary too, even though the label isn't
        # the whole line -- otherwise a following section sweeps the value
        # in as if it were one of its own items.
        inline_label = line.split(":", 1)[0].strip().lower() if ":" in line else ""
        if stripped in boundary_set:
            all_header_idx.append(i)
            if stripped in alias_set:
                wanted_idx.append(i)
        elif inline_label in boundary_set:
            all_header_idx.append(i)

    for line_no in wanted_idx:
        later_headers = [h for h in all_header_idx if h > line_no]
        end = later_headers[0] if later_headers else min(line_no + 15, len(lines))
        blocks.append("\n".join(lines[line_no + 1:end]).strip())
    return blocks


_SENTENCE_LIKE_ITEM_RE = re.compile(
    r"\d+\+?\s*years?\b|\brequired\b|\bpreferred\b|\bresponsib|\bwe are\b|\byou will\b",
    re.IGNORECASE,
)


def _split_verbatim_items(block: str) -> List[str]:
    """Splits a labeled section's content into literal items. Filters out
    "years of experience"/"required"/"preferred" style sentence fragments
    that sometimes sit inside an otherwise clean skills list without a
    clear line break -- these are requirement statements, not skill names."""
    items = []
    for chunk in _SECTION_SPLIT_RE.split(block):
        item = chunk.strip(" \t.").lower()
        item = re.sub(r"\s+", " ", item)
        if 1 < len(item) <= 40 and not item.isdigit() and not _SENTENCE_LIKE_ITEM_RE.search(item):
            items.append(item)
    return items


def _taxonomy_matches(text: str) -> List[str]:
    haystack = text.lower()
    found = []
    for skill in SKILLS_TAXONOMY_SORTED:
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, haystack):
            found.append(skill)
    return found


_GENERIC_PHRASE_RE = re.compile(r"\b[A-Z][A-Za-z0-9+.#]*(?:\s[A-Z][A-Za-z0-9+.#]*){0,2}\b")


def _generic_fallback_phrases(text: str) -> List[str]:
    """Last-resort signal for domains the taxonomy has no coverage for at
    all: pull short Titlecase/ALL-CAPS phrases (tool/technology names are
    conventionally capitalized) and drop generic English words. A lone
    Titlecase word is deliberately excluded -- indistinguishable from
    ordinary sentence-initial capitalization ("Completely unrelated...");
    only multi-word Titlecase phrases or ALL-CAPS/alphanumeric single tokens
    (acronyms like "AWS", "C++") are strong enough signal on their own."""
    found = []
    for match in _GENERIC_PHRASE_RE.finditer(text):
        phrase = match.group(0).strip()
        words = phrase.split()
        if not words or len(words) > 3:
            continue
        if all(w.lower() in _GENERIC_STOPWORDS for w in words):
            continue
        if len(phrase) < 2:
            continue
        if len(words) == 1:
            word = words[0]
            is_acronym = word.isupper() and len(word) >= 2
            has_symbol = bool(re.search(r"[0-9+#.]", word))
            if not (is_acronym or has_symbol):
                continue
        found.append(phrase.lower())
    return found


def extract_skill_phrases(text: str) -> List[str]:
    """Layered, domain-independent phrase extraction: verbatim items from a
    labeled section first, taxonomy cross-reference second, generic
    shape-based fallback only if the first two find nothing at all."""
    if not text or not text.strip():
        return []

    section_items: List[str] = []
    for block in _find_labeled_sections(text, _SKILL_SECTION_ALIASES):
        section_items.extend(_split_verbatim_items(block))

    taxonomy_items = _taxonomy_matches(text)

    combined = sorted(set(section_items) | set(taxonomy_items))
    if combined:
        return combined

    return sorted(set(_generic_fallback_phrases(text)))[:25]
