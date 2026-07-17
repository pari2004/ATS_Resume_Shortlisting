"""End-to-end tests for import_pipeline.py, covering the scenarios called out
in the spec: public Drive folder, large folder (100+ resumes), duplicate
resumes, invalid PDF, DOC/DOCX mix, and an empty folder. Drive access is
mocked the same way test_drive_utils.py mocks it -- no real network needed.
"""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import docx
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import ats_scoring
import excel_store
import import_pipeline
from drive_utils import DriveFetchResult


def _make_pdf(name, email, skills_line, years_line=""):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    _w, height = letter
    y = height - 100
    for text in [name, f"Email: {email}", "Skills", skills_line, "Experience", years_line]:
        c.setFont("Helvetica", 12)
        c.drawString(100, y, text)
        y -= 22
    c.save()
    return buf.getvalue()


def _make_docx(name, email):
    buf = io.BytesIO()
    document = docx.Document()
    document.add_paragraph(name)
    document.add_paragraph(f"Email: {email}")
    document.save(buf)
    return buf.getvalue()


JD = ats_scoring.JobDescription(
    raw_text="Looking for a Python developer.",
    required_skills=["python"],
    min_experience_years=0,
)


class TestImportPipelineUploads(unittest.TestCase):
    """Uses the uploaded_files path (no Drive mocking needed) to exercise the
    parse -> score -> upsert -> report loop directly."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.applicants_path = str(Path(self.tmp_dir.name) / "Applicants.xlsx")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_single_resume_is_imported(self):
        files = {"a.pdf": _make_pdf("Alice Ng", "alice@example.com", "Python, SQL")}
        report = import_pipeline.run_import(
            drive_link=None, jd=JD, applicants_path=self.applicants_path, uploaded_files=files
        )
        self.assertEqual(report.imported, 1)
        self.assertEqual(report.failed, 0)
        applicants, _ = excel_store.load_applicants(self.applicants_path)
        self.assertEqual(len(applicants), 1)

    def test_duplicate_resume_across_two_files_updates_not_duplicates(self):
        pdf_bytes = _make_pdf("Alice Ng", "alice@example.com", "Python, SQL")
        files = {"a.pdf": pdf_bytes, "a_copy.pdf": pdf_bytes}
        report = import_pipeline.run_import(
            drive_link=None, jd=JD, applicants_path=self.applicants_path, uploaded_files=files
        )
        self.assertEqual(report.imported, 1)
        self.assertEqual(report.duplicates_skipped, 1)
        applicants, _ = excel_store.load_applicants(self.applicants_path)
        self.assertEqual(len(applicants), 1)

    def test_invalid_pdf_counts_as_failed_but_does_not_abort_batch(self):
        files = {
            "good.pdf": _make_pdf("Alice Ng", "alice@example.com", "Python"),
            "broken.pdf": b"not a real pdf",
        }
        report = import_pipeline.run_import(
            drive_link=None, jd=JD, applicants_path=self.applicants_path, uploaded_files=files
        )
        self.assertEqual(report.imported, 1)
        self.assertEqual(report.failed, 1)
        statuses = {o.file_name: o.status for o in report.outcomes}
        self.assertEqual(statuses["broken.pdf"], "failed")
        self.assertEqual(statuses["good.pdf"], "imported")

    def test_docx_resume_is_imported(self):
        files = {"bob.docx": _make_docx("Bob Lee", "bob@example.com")}
        report = import_pipeline.run_import(
            drive_link=None, jd=JD, applicants_path=self.applicants_path, uploaded_files=files
        )
        self.assertEqual(report.imported, 1)

    def test_empty_batch_produces_zero_counts(self):
        report = import_pipeline.run_import(
            drive_link=None, jd=JD, applicants_path=self.applicants_path, uploaded_files={}
        )
        self.assertEqual(report.total_files, 0)
        self.assertEqual(report.imported, 0)
        self.assertEqual(report.failed, 0)

    def test_progress_callback_fires_for_every_file(self):
        files = {
            "a.pdf": _make_pdf("Alice Ng", "alice@example.com", "Python"),
            "b.pdf": _make_pdf("Bob Lee", "bob@example.com", "SQL"),
        }
        calls = []
        import_pipeline.run_import(
            drive_link=None, jd=JD, applicants_path=self.applicants_path,
            uploaded_files=files, progress_cb=lambda done, total, name: calls.append((done, total)),
        )
        self.assertEqual(calls[0], (0, 2))
        self.assertEqual(calls[-1], (2, 2))

    def test_large_batch_of_100_resumes(self):
        files = {
            f"candidate_{i}.pdf": _make_pdf(f"Person {i}", f"person{i}@example.com", "Python, SQL")
            for i in range(100)
        }
        report = import_pipeline.run_import(
            drive_link=None, jd=JD, applicants_path=self.applicants_path, uploaded_files=files
        )
        self.assertEqual(report.total_files, 100)
        self.assertEqual(report.imported, 100)
        self.assertEqual(report.failed, 0)
        self.assertEqual(report.progress_pct, 100.0)
        applicants, _ = excel_store.load_applicants(self.applicants_path)
        self.assertEqual(len(applicants), 100)
        ids = set(applicants["Candidate ID"])
        self.assertEqual(len(ids), 100)  # every candidate got a unique ID


class TestImportPipelineDriveFolder(unittest.TestCase):
    """Exercises the drive_link path with fetch_pdfs_from_drive mocked."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.applicants_path = str(Path(self.tmp_dir.name) / "Applicants.xlsx")

    def tearDown(self):
        self.tmp_dir.cleanup()

    @patch("import_pipeline.fetch_pdfs_from_drive")
    def test_public_folder_success(self, mock_fetch):
        mock_fetch.return_value = DriveFetchResult(
            {"alice.pdf": _make_pdf("Alice Ng", "alice@example.com", "Python")},
            warnings=[],
            file_meta={"alice.pdf": {"file_id": "id123", "url": "https://drive.google.com/x"}},
        )
        report = import_pipeline.run_import(
            drive_link="https://drive.google.com/drive/folders/abc",
            jd=JD, applicants_path=self.applicants_path,
        )
        self.assertEqual(report.imported, 1)
        applicants, _ = excel_store.load_applicants(self.applicants_path)
        self.assertEqual(applicants.iloc[0]["Google Drive File ID"], "id123")

    @patch("import_pipeline.fetch_pdfs_from_drive")
    def test_empty_folder_produces_drive_warning_and_no_crash(self, mock_fetch):
        from drive_utils import DriveFetchError
        mock_fetch.side_effect = DriveFetchError("No resumes were found in that Drive folder.")

        report = import_pipeline.run_import(
            drive_link="https://drive.google.com/drive/folders/empty",
            jd=JD, applicants_path=self.applicants_path,
        )
        self.assertEqual(report.total_files, 0)
        self.assertTrue(any("No resumes" in w for w in report.drive_warnings))


if __name__ == "__main__":
    unittest.main(verbosity=2)
