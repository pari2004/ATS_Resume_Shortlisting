"""Tests for ats_scoring.py: JD structured-field extraction, the 6-factor
weighted ATS score, education/certification matching, domain-aware
recommendation phrasing, and Shortlist/Maybe/Reject threshold boundaries."""

import unittest

import ats_scoring
from resume_parser import ParsedResume


def _resume(skills=None, years=0.0, raw_text="", certifications=None,
            education=None, email="a@example.com", phone="1234567890",
            experience_entries=None):
    return ParsedResume(
        skills=skills or [],
        total_experience_years=years,
        raw_text=raw_text,
        certifications=certifications or [],
        education=education or [],
        email=email,
        phone=phone,
        experience_entries=experience_entries or (["some role"] if years else []),
    )


class TestAnalyzeJobDescription(unittest.TestCase):
    def test_domain_detected(self):
        jd = ats_scoring.analyze_job_description("Full Stack Developer\nReact, Node.js, MongoDB required.")
        self.assertEqual(jd.domain, "Full Stack Development")

    def test_required_and_preferred_split(self):
        jd_text = (
            "Finance Executive\n"
            "Required Skills\nGST, Tally, Excel\n"
            "Preferred Skills\nSAP, Power BI\n"
        )
        jd = ats_scoring.analyze_job_description(jd_text)
        self.assertEqual(set(jd.required_skills), {"gst", "tally", "excel"})
        self.assertEqual(set(jd.preferred_skills), {"sap", "power bi"})

    def test_no_required_preferred_split_puts_everything_in_required(self):
        jd = ats_scoring.analyze_job_description("Python Developer\nSkills\nPython, Django, PostgreSQL")
        self.assertIn("python", jd.required_skills)
        self.assertEqual(jd.preferred_skills, [])

    def test_experience_range_extraction(self):
        jd = ats_scoring.analyze_job_description("We need 2-4 years of experience.")
        self.assertEqual(jd.min_experience_years, 2.0)
        self.assertEqual(jd.preferred_experience_years, 4.0)

    def test_single_experience_value_extraction(self):
        jd = ats_scoring.analyze_job_description("Must have 5+ years of experience.")
        self.assertEqual(jd.min_experience_years, 5.0)

    def test_education_requirement_detected(self):
        jd = ats_scoring.analyze_job_description("Education: Bachelor's degree required.")
        self.assertEqual(jd.education_requirement, "bachelor's")

    def test_education_requirement_does_not_false_positive_on_manage(self):
        jd = ats_scoring.analyze_job_description("Responsibilities\n- Manage GST filings")
        self.assertEqual(jd.education_requirement, "")

    def test_certifications_extracted(self):
        jd = ats_scoring.analyze_job_description("Certifications: PMP certified preferred.")
        self.assertIn("pmp", jd.certifications_required)

    def test_title_extracted_from_first_line(self):
        jd = ats_scoring.analyze_job_description("HR Recruiter\nWe are hiring...")
        self.assertEqual(jd.title, "HR Recruiter")

    def test_title_extracted_from_label(self):
        jd = ats_scoring.analyze_job_description("Job Title: Data Scientist\nWe are hiring...")
        self.assertEqual(jd.title, "Data Scientist")

    def test_sections_do_not_leak_into_each_other(self):
        jd_text = (
            "Required Skills\nPython, SQL\n"
            "Preferred Skills\nDocker\n"
            "Education: Master's preferred.\n"
            "Responsibilities\n- Build things\n"
        )
        jd = ats_scoring.analyze_job_description(jd_text)
        self.assertNotIn("education", " ".join(jd.required_skills))
        self.assertNotIn("responsibilities", " ".join(jd.preferred_skills))
        self.assertEqual(jd.responsibilities, ["Build things"])


class TestScoringWeights(unittest.TestCase):
    def test_default_weights_sum_to_one(self):
        w = ats_scoring.ScoringWeights()
        total = (w.required_skills + w.preferred_skills + w.experience
                  + w.education + w.certifications + w.resume_quality)
        self.assertAlmostEqual(total, 1.0)

    def test_normalization_rescales_arbitrary_weights(self):
        w = ats_scoring.ScoringWeights(required_skills=4, preferred_skills=2, experience=2,
                                        education=1, certifications=1, resume_quality=0)
        normalized = w.normalized()
        total = (normalized.required_skills + normalized.preferred_skills + normalized.experience
                  + normalized.education + normalized.certifications + normalized.resume_quality)
        self.assertAlmostEqual(total, 1.0)
        self.assertAlmostEqual(normalized.required_skills, 0.4)


class TestScoreResume(unittest.TestCase):
    def test_full_skill_match_full_experience_and_quality(self):
        jd = ats_scoring.JobDescription(required_skills=["python", "sql"], min_experience_years=2)
        resume = _resume(skills=["python", "sql"], years=5, education=["B.Tech"])
        result = ats_scoring.score_resume(resume, jd)
        self.assertEqual(result.required_skill_match_pct, 100.0)
        self.assertEqual(result.experience_match_pct, 100.0)
        self.assertEqual(result.ats_score, 100.0)
        self.assertEqual(result.missing_required_skills, [])

    def test_partial_skill_match(self):
        jd = ats_scoring.JobDescription(required_skills=["python", "sql", "aws"], min_experience_years=0)
        resume = _resume(skills=["python"])
        result = ats_scoring.score_resume(resume, jd)
        self.assertAlmostEqual(result.required_skill_match_pct, 33.3, delta=0.5)
        self.assertIn("sql", result.missing_required_skills)
        self.assertIn("aws", result.missing_required_skills)

    def test_no_required_skills_means_full_skill_score(self):
        jd = ats_scoring.JobDescription(required_skills=[], min_experience_years=0)
        resume = _resume(skills=[])
        result = ats_scoring.score_resume(resume, jd)
        self.assertEqual(result.required_skill_match_pct, 100.0)

    def test_preferred_skills_scored_separately(self):
        jd = ats_scoring.JobDescription(required_skills=["python"], preferred_skills=["docker", "aws"])
        resume = _resume(skills=["python", "docker"])
        result = ats_scoring.score_resume(resume, jd)
        self.assertEqual(result.required_skill_match_pct, 100.0)
        self.assertAlmostEqual(result.preferred_skill_match_pct, 50.0)

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
        self.assertEqual(result.required_skill_match_pct, 100.0)

    def test_skill_matched_via_raw_text_even_if_not_in_skills_list(self):
        jd = ats_scoring.JobDescription(required_skills=["docker"], min_experience_years=0)
        resume = _resume(skills=[], raw_text="Experienced with Docker containers in production.")
        result = ats_scoring.score_resume(resume, jd)
        self.assertEqual(result.required_skill_match_pct, 100.0)

    def test_education_match_when_requirement_met(self):
        jd = ats_scoring.JobDescription(education_requirement="bachelor's")
        resume = _resume(education=["Bachelor of Science in Computer Science"])
        result = ats_scoring.score_resume(resume, jd)
        self.assertEqual(result.education_match_pct, 100.0)

    def test_education_mismatch_when_requirement_not_met(self):
        jd = ats_scoring.JobDescription(education_requirement="master's")
        resume = _resume(education=["Bachelor of Science"])
        result = ats_scoring.score_resume(resume, jd)
        self.assertEqual(result.education_match_pct, 0.0)
        self.assertTrue(any("master's" in w for w in result.weaknesses))

    def test_no_education_requirement_means_full_credit(self):
        jd = ats_scoring.JobDescription(education_requirement="")
        resume = _resume(education=[])
        result = ats_scoring.score_resume(resume, jd)
        self.assertEqual(result.education_match_pct, 100.0)

    def test_certifications_match(self):
        jd = ats_scoring.JobDescription(certifications_required=["pmp"])
        resume = _resume(certifications=["PMP Certified Project Manager"])
        result = ats_scoring.score_resume(resume, jd)
        self.assertEqual(result.certifications_match_pct, 100.0)

    def test_resume_quality_reflects_completeness(self):
        jd = ats_scoring.JobDescription()
        complete = _resume(skills=["python"], years=2, education=["BS"])
        incomplete = ParsedResume(skills=[], email="", phone="", experience_entries=[], education=[])
        self.assertGreater(
            ats_scoring.score_resume(complete, jd).resume_quality_pct,
            ats_scoring.score_resume(incomplete, jd).resume_quality_pct,
        )

    def test_recommendation_thresholds(self):
        jd = ats_scoring.JobDescription(required_skills=["python"], min_experience_years=0)
        shortlisted = ats_scoring.score_resume(_resume(skills=["python"], education=["BS"]), jd)
        self.assertEqual(shortlisted.recommendation, "Shortlist")

        # A JD with no preferred/education/certification requirements gives
        # every candidate full credit on those 3 factors by design (mirrors
        # the pre-existing "no requirement stated = full credit" rule) --
        # so a genuinely bad-fit test needs a fully-specified JD the
        # candidate clearly fails across the board, not just missing skills.
        demanding_jd = ats_scoring.JobDescription(
            required_skills=["python"], preferred_skills=["docker"],
            education_requirement="master's", certifications_required=["pmp"],
        )
        rejected = ats_scoring.score_resume(
            ParsedResume(skills=[], email="", phone="", experience_entries=[], education=[]), demanding_jd
        )
        self.assertEqual(rejected.recommendation, "Reject")

    def test_custom_thresholds_override_defaults(self):
        jd = ats_scoring.JobDescription(required_skills=["python"], min_experience_years=0)
        result = ats_scoring.score_resume(_resume(skills=[]), jd, shortlist_threshold=0.0)
        self.assertEqual(result.recommendation, "Shortlist")

    def test_certifications_add_a_strength(self):
        jd = ats_scoring.JobDescription(required_skills=[], min_experience_years=0)
        resume = _resume(certifications=["AWS Certified Developer"])
        result = ats_scoring.score_resume(resume, jd)
        self.assertTrue(any("certification" in s for s in result.strengths))

    def test_domain_aware_recommendation_differs_by_domain(self):
        tech_jd = ats_scoring.JobDescription(domain="Full Stack Development", required_skills=["react", "node.js"])
        finance_jd = ats_scoring.JobDescription(domain="Finance", required_skills=["gst", "tally"])

        tech_resume = _resume(skills=["react", "node.js"])
        finance_resume = _resume(skills=["gst", "tally"])

        tech_result = ats_scoring.score_resume(tech_resume, tech_jd)
        finance_result = ats_scoring.score_resume(finance_resume, finance_jd)

        self.assertIn("experience", tech_result.recommendation_text.lower())
        self.assertIn("knowledge", finance_result.recommendation_text.lower())
        self.assertNotEqual(tech_result.recommendation_text, finance_result.recommendation_text)

    def test_unknown_domain_gets_neutral_phrasing(self):
        jd = ats_scoring.JobDescription(domain="General / Other", required_skills=["excel"])
        resume = _resume(skills=["excel"])
        result = ats_scoring.score_resume(resume, jd)
        self.assertTrue(result.recommendation_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
