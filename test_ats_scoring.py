"""Tests for ats_scoring.py: skill/experience match %, weighted ATS score,
and Shortlist/Maybe/Reject threshold boundaries."""

import unittest

import ats_scoring
from resume_parser import ParsedResume


def _resume(skills=None, years=0.0, raw_text="", certifications=None):
    return ParsedResume(
        skills=skills or [],
        total_experience_years=years,
        raw_text=raw_text,
        certifications=certifications or [],
    )


class TestSkillExtraction(unittest.TestCase):
    def test_extract_skills_from_jd(self):
        jd_text = "We need a Python developer with AWS and React experience."
        skills = ats_scoring.extract_skills_from_jd(jd_text)
        self.assertIn("python", skills)
        self.assertIn("aws", skills)
        self.assertIn("react", skills)

    def test_no_skills_found(self):
        self.assertEqual(ats_scoring.extract_skills_from_jd("Completely unrelated text."), [])


class TestScoreResume(unittest.TestCase):
    def test_full_skill_match_full_experience(self):
        jd = ats_scoring.JobDescription(required_skills=["python", "sql"], min_experience_years=2)
        resume = _resume(skills=["python", "sql"], years=5)
        result = ats_scoring.score_resume(resume, jd)
        self.assertEqual(result.skill_match_pct, 100.0)
        self.assertEqual(result.experience_match_pct, 100.0)
        self.assertEqual(result.ats_score, 100.0)
        self.assertEqual(result.missing_skills, [])

    def test_partial_skill_match(self):
        jd = ats_scoring.JobDescription(required_skills=["python", "sql", "aws"], min_experience_years=0)
        resume = _resume(skills=["python"])
        result = ats_scoring.score_resume(resume, jd)
        self.assertAlmostEqual(result.skill_match_pct, 33.3, delta=0.5)
        self.assertIn("sql", result.missing_skills)
        self.assertIn("aws", result.missing_skills)

    def test_no_required_skills_means_full_skill_score(self):
        jd = ats_scoring.JobDescription(required_skills=[], min_experience_years=0)
        resume = _resume(skills=[])
        result = ats_scoring.score_resume(resume, jd)
        self.assertEqual(result.skill_match_pct, 100.0)

    def test_experience_shortfall_scales_down(self):
        jd = ats_scoring.JobDescription(required_skills=[], min_experience_years=10)
        resume = _resume(years=5)
        result = ats_scoring.score_resume(resume, jd)
        self.assertEqual(result.experience_match_pct, 50.0)
        self.assertTrue(any("short of the 10" in w for w in result.weaknesses))

    def test_experience_over_requirement_caps_at_100(self):
        jd = ats_scoring.JobDescription(required_skills=[], min_experience_years=2)
        resume = _resume(years=20)
        result = ats_scoring.score_resume(resume, jd)
        self.assertEqual(result.experience_match_pct, 100.0)

    def test_fuzzy_skill_match(self):
        jd = ats_scoring.JobDescription(required_skills=["kubernetes"], min_experience_years=0)
        resume = _resume(skills=["kuberentes"])  # typo, close fuzzy match
        result = ats_scoring.score_resume(resume, jd)
        self.assertEqual(result.skill_match_pct, 100.0)

    def test_skill_matched_via_raw_text_even_if_not_in_skills_list(self):
        jd = ats_scoring.JobDescription(required_skills=["docker"], min_experience_years=0)
        resume = _resume(skills=[], raw_text="Experienced with Docker containers in production.")
        result = ats_scoring.score_resume(resume, jd)
        self.assertEqual(result.skill_match_pct, 100.0)

    def test_recommendation_thresholds(self):
        jd = ats_scoring.JobDescription(required_skills=["python"], min_experience_years=0)

        shortlisted = ats_scoring.score_resume(_resume(skills=["python"]), jd)
        self.assertEqual(shortlisted.recommendation, "Shortlist")

        rejected = ats_scoring.score_resume(_resume(skills=[]), jd)
        self.assertEqual(rejected.recommendation, "Reject")

    def test_custom_thresholds_override_defaults(self):
        jd = ats_scoring.JobDescription(required_skills=["python"], min_experience_years=0)
        # skill_match=0 -> ats_score 0, but a threshold of 0 should still shortlist
        result = ats_scoring.score_resume(_resume(skills=[]), jd, shortlist_threshold=0.0)
        self.assertEqual(result.recommendation, "Shortlist")

    def test_certifications_add_a_strength(self):
        jd = ats_scoring.JobDescription(required_skills=[], min_experience_years=0)
        resume = _resume(certifications=["AWS Certified Developer"])
        result = ats_scoring.score_resume(resume, jd)
        self.assertTrue(any("certification" in s for s in result.strengths))


if __name__ == "__main__":
    unittest.main(verbosity=2)
