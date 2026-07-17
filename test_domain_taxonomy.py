"""Tests for domain_taxonomy.py: domain detection across the domains called
out in the spec's testing requirement, the "General / Other" fallback for
vague/unlisted JDs, and the layered skill-phrase extraction (including the
generic fallback for a domain with zero taxonomy coverage)."""

import unittest

import domain_taxonomy as dt

DOMAIN_JDS = {
    "Full Stack Development": "Full Stack Developer\nWe need React, Node.js, MongoDB, and Express experience.",
    "Java Developer": "Java Developer\nRequired Skills\nJava, Spring Boot, Hibernate, Microservices",
    "Python Developer": "Python Developer\nRequired Skills\nPython, Django, Flask, REST APIs",
    "AI Engineer": "AI Engineer\nExperience with LLMs, prompt engineering, and GenAI pipelines required.",
    "Data Scientist": "Data Scientist\nSkills\nPython, Pandas, Machine Learning, Statistics, TensorFlow",
    "Business Analyst": "Business Analyst\nSkills\nRequirements gathering, SQL, Business analysis, Stakeholder management",
    "Recruiter": "HR Recruiter\nSkills\nRecruitment, Talent Acquisition, Onboarding, Employee Engagement",
    "Digital Marketing": "Digital Marketing Manager\nSkills\nSEO, SEM, Google Analytics, Meta Ads, Content Marketing",
    "Finance": "Finance Executive\nSkills\nGST, Tally, SAP, Excel, Accounting",
    "Sales": "Sales Executive\nSkills\nLead Generation, Cold Calling, Negotiation, CRM, Salesforce",
    "Customer Support": "Customer Support Representative\nSkills\nZendesk, Ticketing, Customer Service, Communication",
    "UI/UX Designer": "UI/UX Designer\nSkills\nFigma, Adobe XD, Wireframing, User Research, Prototyping",
}


class TestDetectDomain(unittest.TestCase):
    def test_each_required_domain_is_detected(self):
        for expected_domain, jd_text in DOMAIN_JDS.items():
            with self.subTest(domain=expected_domain):
                self.assertEqual(dt.detect_domain(jd_text), expected_domain)

    def test_vague_jd_falls_back_to_general(self):
        vague = "We are looking for a great team player to join us and help grow the company."
        self.assertEqual(dt.detect_domain(vague), "General / Other")

    def test_empty_jd_falls_back_to_general(self):
        self.assertEqual(dt.detect_domain(""), "General / Other")

    def test_unlisted_niche_domain_falls_back_to_general(self):
        # Deliberately not one of DOMAIN_KEYWORDS' ~25 domains.
        text = "Marine Biologist\nStudy coral reef ecosystems and marine life populations."
        self.assertEqual(dt.detect_domain(text), "General / Other")

    def test_different_jds_produce_different_domains(self):
        domains = {dt.detect_domain(text) for text in DOMAIN_JDS.values()}
        self.assertGreater(len(domains), 8)  # meaningfully differentiated, not collapsing to one bucket


class TestExtractSkillPhrases(unittest.TestCase):
    def test_verbatim_section_extraction_works_with_zero_taxonomy_coverage(self):
        # Mechanical engineering has no taxonomy entries at all -- this must
        # still return real skills via the labeled-section layer.
        text = "Mechanical Design Engineer\nRequirements\nAutoCAD, SolidWorks, GD&T, CNC Machining"
        skills = dt.extract_skill_phrases(text)
        self.assertIn("autocad", skills)
        self.assertIn("solidworks", skills)
        self.assertIn("cnc machining", skills)

    def test_healthcare_domain_gets_real_signal(self):
        text = "Registered Nurse\nSkills\nPatient Care, Vital Signs Monitoring, EHR, Medication Administration"
        skills = dt.extract_skill_phrases(text)
        self.assertTrue(skills)
        self.assertIn("patient care", skills)

    def test_taxonomy_still_used_for_prose_without_a_labeled_section(self):
        text = "We need someone proficient in Python and SQL for data pipelines."
        skills = dt.extract_skill_phrases(text)
        self.assertIn("python", skills)
        self.assertIn("sql", skills)

    def test_generic_stopwords_are_not_returned_as_skills(self):
        text = "We are looking for a great team player to join us."
        skills = dt.extract_skill_phrases(text)
        self.assertEqual(skills, [])

    def test_sentence_initial_capitalization_is_not_mistaken_for_a_skill(self):
        skills = dt.extract_skill_phrases("Completely unrelated text with no structure at all.")
        self.assertEqual(skills, [])


class TestEducationLevelOf(unittest.TestCase):
    def test_bachelor_detected(self):
        self.assertEqual(dt.education_level_of("Bachelor's degree required."), "bachelor's")

    def test_master_detected(self):
        self.assertEqual(dt.education_level_of("MBA preferred."), "master's")

    def test_word_boundary_prevents_false_positive(self):
        # "ma" must not match inside "Manage" -- this was a real bug found
        # against production JD text.
        self.assertEqual(dt.education_level_of("Manage GST filings and bank reconciliation."), "")

    def test_no_match_returns_empty_string(self):
        self.assertEqual(dt.education_level_of("Great communication skills required."), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
