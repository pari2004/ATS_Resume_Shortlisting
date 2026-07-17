"""Tests for excel_store.py: fresh-file creation, append-new, update-on-
duplicate (by hash/email/phone), and never touching unrelated rows."""

import tempfile
import unittest
from pathlib import Path

import excel_store as es


def _record(**overrides):
    base = {
        "Name": "Alice Ng", "Email": "alice@example.com", "Phone": "9998887777",
        "Experience": "3 years", "Skills": "python, sql", "Education": "B.S.",
        "ATS Score": 80, "Skill Match %": 90, "Experience Match %": 70,
        "Missing Skills": "", "Recommendation": "Shortlist", "Status": "New",
        "Resume File Name": "alice.pdf", "Google Drive File ID": "id1",
        "Google Drive URL": "https://drive.google.com/x",
    }
    base.update(overrides)
    return base


class TestExcelStore(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp_dir.name) / "Applicants.xlsx")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_load_missing_file_creates_empty_frame_with_correct_columns(self):
        applicants, dedup = es.load_applicants(self.path)
        self.assertEqual(list(applicants.columns), es.COLUMNS)
        self.assertTrue(applicants.empty)

    def test_insert_new_candidate(self):
        applicants, dedup = es.load_applicants(self.path)
        applicants, dedup, action = es.upsert_candidate(applicants, dedup, _record(), resume_hash="h1")
        self.assertEqual(action, "inserted")
        self.assertEqual(len(applicants), 1)
        self.assertEqual(applicants.iloc[0]["Candidate ID"], "CAND-0001")

    def test_second_distinct_candidate_gets_next_id(self):
        applicants, dedup = es.load_applicants(self.path)
        applicants, dedup, _ = es.upsert_candidate(applicants, dedup, _record(), resume_hash="h1")
        applicants, dedup, action = es.upsert_candidate(
            applicants, dedup,
            _record(Name="Bob Lee", Email="bob@example.com", Phone="1112223333", **{"Resume File Name": "bob.pdf"}),
            resume_hash="h2",
        )
        self.assertEqual(action, "inserted")
        self.assertEqual(len(applicants), 2)
        self.assertEqual(applicants.iloc[1]["Candidate ID"], "CAND-0002")

    def test_exact_resume_hash_duplicate_is_rescored_not_skipped(self):
        """An identical resume file re-imported doesn't get silently
        skipped -- it's refreshed (same candidate row, no duplicate row),
        since a re-run may be scoring against a different job description
        than the first import used."""
        applicants, dedup = es.load_applicants(self.path)
        applicants, dedup, _ = es.upsert_candidate(applicants, dedup, _record(), resume_hash="h1")
        applicants, dedup, action = es.upsert_candidate(
            applicants, dedup, _record(**{"ATS Score": 42}), resume_hash="h1"
        )
        self.assertEqual(action, "rescored")
        self.assertEqual(len(applicants), 1)  # still one row, not a duplicate
        self.assertEqual(applicants.iloc[0]["ATS Score"], "42")  # refreshed, not stale

    def test_same_email_different_hash_updates_existing_row(self):
        applicants, dedup = es.load_applicants(self.path)
        applicants, dedup, _ = es.upsert_candidate(applicants, dedup, _record(), resume_hash="h1")
        applicants, dedup, action = es.upsert_candidate(
            applicants, dedup, _record(**{"ATS Score": 95}), resume_hash="h2"
        )
        self.assertEqual(action, "updated")
        self.assertEqual(len(applicants), 1)
        self.assertEqual(applicants.iloc[0]["Candidate ID"], "CAND-0001")
        self.assertEqual(applicants.iloc[0]["ATS Score"], "95")

    def test_same_phone_different_email_and_hash_updates_existing_row(self):
        applicants, dedup = es.load_applicants(self.path)
        applicants, dedup, _ = es.upsert_candidate(applicants, dedup, _record(), resume_hash="h1")
        applicants, dedup, action = es.upsert_candidate(
            applicants, dedup,
            _record(Email="alice.new@example.com", **{"ATS Score": 88}),
            resume_hash="h2",
        )
        self.assertEqual(action, "updated")
        self.assertEqual(len(applicants), 1)
        self.assertEqual(applicants.iloc[0]["Email"], "alice.new@example.com")

    def test_unrelated_rows_are_never_touched(self):
        applicants, dedup = es.load_applicants(self.path)
        applicants, dedup, _ = es.upsert_candidate(applicants, dedup, _record(), resume_hash="h1")
        applicants, dedup, _ = es.upsert_candidate(
            applicants, dedup,
            _record(Name="Bob Lee", Email="bob@example.com", Phone="1112223333"),
            resume_hash="h2",
        )
        original_bob = applicants[applicants["Email"] == "bob@example.com"].iloc[0].to_dict()

        applicants, dedup, action = es.upsert_candidate(
            applicants, dedup, _record(**{"ATS Score": 99}), resume_hash="h3"
        )
        self.assertEqual(action, "updated")
        bob_after = applicants[applicants["Email"] == "bob@example.com"].iloc[0].to_dict()
        self.assertEqual(original_bob, bob_after)

    def test_save_and_reload_round_trip(self):
        applicants, dedup = es.load_applicants(self.path)
        applicants, dedup, _ = es.upsert_candidate(applicants, dedup, _record(), resume_hash="h1")
        es.save_applicants(applicants, dedup, self.path)

        reloaded, reloaded_dedup = es.load_applicants(self.path)
        self.assertEqual(len(reloaded), 1)
        self.assertEqual(reloaded.iloc[0]["Email"], "alice@example.com")

        # A follow-up run against the same file should still detect the
        # exact-duplicate resume via the persisted dedup index, and rescore
        # it (not silently skip it) without creating a duplicate row.
        reloaded, reloaded_dedup, action = es.upsert_candidate(
            reloaded, reloaded_dedup, _record(), resume_hash="h1"
        )
        self.assertEqual(action, "rescored")
        self.assertEqual(len(reloaded), 1)

    def test_update_after_reload_accepts_numeric_values(self):
        """Regression test: pandas >= 3.0's dtype=str on read_excel produces a
        strict StringDtype that raises on assigning a raw int/float via
        .at[] -- this only shows up once a file has been saved and reloaded
        (a fresh in-memory DataFrame has lenient object dtype), which is
        exactly the situation on a real second import run."""
        applicants, dedup = es.load_applicants(self.path)
        applicants, dedup, _ = es.upsert_candidate(
            applicants, dedup, _record(**{"ATS Score": 71.8, "Skill Match %": 60}), resume_hash="h1"
        )
        es.save_applicants(applicants, dedup, self.path)

        reloaded, reloaded_dedup = es.load_applicants(self.path)
        reloaded, reloaded_dedup, action = es.upsert_candidate(
            reloaded, reloaded_dedup,
            _record(**{"ATS Score": 92.5, "Skill Match %": 100}),
            resume_hash="h2",  # different hash, same email -> update path
        )
        self.assertEqual(action, "updated")
        self.assertEqual(reloaded.iloc[0]["ATS Score"], "92.5")

    def test_insert_after_reload_accepts_numeric_values(self):
        applicants, dedup = es.load_applicants(self.path)
        applicants, dedup, _ = es.upsert_candidate(applicants, dedup, _record(), resume_hash="h1")
        es.save_applicants(applicants, dedup, self.path)

        reloaded, reloaded_dedup = es.load_applicants(self.path)
        reloaded, reloaded_dedup, action = es.upsert_candidate(
            reloaded, reloaded_dedup,
            _record(Name="Bob Lee", Email="bob@example.com", Phone="1112223333", **{"ATS Score": 45.3}),
            resume_hash="h2",
        )
        self.assertEqual(action, "inserted")
        self.assertEqual(reloaded.iloc[1]["ATS Score"], "45.3")

    def test_applicants_sheet_has_exactly_the_spec_columns(self):
        applicants, dedup = es.load_applicants(self.path)
        applicants, dedup, _ = es.upsert_candidate(applicants, dedup, _record(), resume_hash="h1")
        es.save_applicants(applicants, dedup, self.path)

        import openpyxl
        wb = openpyxl.load_workbook(self.path)
        header = [c.value for c in next(wb[es.APPLICANTS_SHEET].iter_rows(max_row=1))]
        self.assertEqual(header, es.COLUMNS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
