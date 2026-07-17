"""Tests for drive_utils.py covering URL parsing and the import scenarios
requested for the Google Drive fetch feature: public folder, private folder,
empty folder, nested folders, mixed PDF/DOCX, invalid URL, and a Shared
Drive folder. The Drive API and gdown are mocked since no live Google
credentials/network access are available in this environment.
"""
import io
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import drive_utils
from drive_utils import DriveFetchError, extract_folder_id, fetch_pdfs_from_drive


class TestExtractFolderId(unittest.TestCase):
    def test_basic_folder_link(self):
        self.assertEqual(
            extract_folder_id("https://drive.google.com/drive/folders/1AbCdEfGhIjK"),
            "1AbCdEfGhIjK",
        )

    def test_folder_link_with_sharing_param(self):
        self.assertEqual(
            extract_folder_id("https://drive.google.com/drive/folders/1AbCdEfGhIjK?usp=sharing"),
            "1AbCdEfGhIjK",
        )

    def test_folder_link_with_account_index(self):
        self.assertEqual(
            extract_folder_id("https://drive.google.com/drive/u/0/folders/1AbCdEfGhIjK?usp=drive_link"),
            "1AbCdEfGhIjK",
        )

    def test_open_id_link(self):
        self.assertEqual(
            extract_folder_id("https://drive.google.com/open?id=1AbCdEfGhIjK"),
            "1AbCdEfGhIjK",
        )

    def test_bare_id(self):
        self.assertEqual(extract_folder_id("1AbCdEfGhIjK_bareID123"), "1AbCdEfGhIjK_bareID123")

    def test_link_embedded_in_sentence(self):
        self.assertEqual(
            extract_folder_id("here is the folder: https://drive.google.com/drive/folders/1AbCdEfGhIjK?usp=sharing thanks"),
            "1AbCdEfGhIjK",
        )

    def test_invalid_url_raises(self):
        with self.assertRaises(DriveFetchError):
            extract_folder_id("https://example.com/not-a-drive-link")

    def test_empty_link_raises(self):
        with self.assertRaises(DriveFetchError):
            extract_folder_id("   ")


def _http_error(status, reason_body=b'{"error": {"errors": [{"reason": "notFound"}]}}'):
    from googleapiclient.errors import HttpError

    resp = MagicMock()
    resp.status = status
    return HttpError(resp, reason_body)


class TestDriveApiScenarios(unittest.TestCase):
    """Simulates googleapiclient responses to exercise fetch_pdfs_from_drive's
    Drive-API backend without needing real network access or credentials."""

    def setUp(self):
        # Force the API-key path so _build_drive_service() returns a service.
        self.env_patch = patch.dict("os.environ", {"GOOGLE_DRIVE_API_KEY": "fake-key-for-tests"})
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()

    def _mock_service(self, get_side_effect=None, list_side_effect=None, download_bytes=None):
        service = MagicMock()
        if get_side_effect is not None:
            service.files.return_value.get.return_value.execute.side_effect = get_side_effect
        if list_side_effect is not None:
            service.files.return_value.list.return_value.execute.side_effect = list_side_effect
        return service

    @patch("drive_utils._build_drive_service")
    def test_public_folder_with_pdfs(self, mock_build):
        service = MagicMock()
        service.files().get().execute.return_value = {
            "id": "folder1_id_1234567890", "name": "Resumes", "mimeType": drive_utils.FOLDER_MIME,
        }
        service.files().list().execute.return_value = {
            "files": [
                {"id": "f1", "name": "alice.pdf", "mimeType": "application/pdf"},
                {"id": "f2", "name": "bob.pdf", "mimeType": "application/pdf"},
            ]
        }
        mock_build.return_value = (service, "api_key")

        with patch.object(drive_utils, "_drive_api_download", return_value=b"%PDF-fake-bytes"):
            result = fetch_pdfs_from_drive("https://drive.google.com/drive/folders/folder1_id_1234567890")

        self.assertEqual(set(result.keys()), {"alice.pdf", "bob.pdf"})
        self.assertEqual(result["alice.pdf"], b"%PDF-fake-bytes")
        self.assertEqual(result.warnings, [])

    @patch("drive_utils._build_drive_service")
    def test_private_folder_shows_friendly_message(self, mock_build):
        from googleapiclient.errors import HttpError

        service = MagicMock()
        service.files().get().execute.side_effect = _http_error(404)
        mock_build.return_value = (service, "api_key")

        with self.assertRaises(DriveFetchError) as ctx:
            fetch_pdfs_from_drive("https://drive.google.com/drive/folders/privatefolder_id_1234567890")

        self.assertIn("not shared publicly", str(ctx.exception))
        self.assertNotIn("HttpError", str(ctx.exception))

    @patch("drive_utils._build_drive_service")
    def test_empty_folder(self, mock_build):
        service = MagicMock()
        service.files().get().execute.return_value = {
            "id": "folder1_id_1234567890", "name": "Empty", "mimeType": drive_utils.FOLDER_MIME,
        }
        service.files().list().execute.return_value = {"files": []}
        mock_build.return_value = (service, "api_key")

        with self.assertRaises(DriveFetchError) as ctx:
            fetch_pdfs_from_drive("https://drive.google.com/drive/folders/folder1_id_1234567890")
        self.assertIn("No resumes were found", str(ctx.exception))

    @patch("drive_utils._build_drive_service")
    def test_nested_folders_are_traversed(self, mock_build):
        service = MagicMock()
        service.files().get().execute.return_value = {
            "id": "root_folder_id_1234567890", "name": "Root", "mimeType": drive_utils.FOLDER_MIME,
        }

        def list_side_effect(**kwargs):
            pass  # unused; real call is via .list().execute() chain below

        # files().list() is called repeatedly with different q= per folder;
        # emulate by returning different results based on call count.
        responses = [
            {"files": [
                {"id": "sub1", "name": "2024", "mimeType": drive_utils.FOLDER_MIME},
                {"id": "f1", "name": "top.pdf", "mimeType": "application/pdf"},
            ]},
            {"files": [
                {"id": "f2", "name": "nested.docx",
                 "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
            ]},
        ]
        service.files().list().execute.side_effect = responses
        mock_build.return_value = (service, "api_key")

        with patch.object(drive_utils, "_drive_api_download", return_value=b"bytes"):
            result = fetch_pdfs_from_drive("https://drive.google.com/drive/folders/root_folder_id_1234567890")

        self.assertEqual(set(result.keys()), {"top.pdf", "nested.docx"})

    @patch("drive_utils._build_drive_service")
    def test_mixed_supported_and_unsupported_files(self, mock_build):
        service = MagicMock()
        service.files().get().execute.return_value = {
            "id": "folder1_id_1234567890", "name": "Mixed", "mimeType": drive_utils.FOLDER_MIME,
        }
        service.files().list().execute.return_value = {
            "files": [
                {"id": "f1", "name": "resume.pdf", "mimeType": "application/pdf"},
                {"id": "f2", "name": "resume2.docx",
                 "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
                {"id": "f3", "name": "photo.png", "mimeType": "image/png"},
                {"id": "f4", "name": "notes.txt", "mimeType": "text/plain"},
            ]
        }
        mock_build.return_value = (service, "api_key")

        with patch.object(drive_utils, "_drive_api_download", return_value=b"bytes"):
            result = fetch_pdfs_from_drive("https://drive.google.com/drive/folders/folder1_id_1234567890")

        self.assertEqual(set(result.keys()), {"resume.pdf", "resume2.docx"})
        self.assertTrue(any("2 file(s) skipped" in w for w in result.warnings))

    @patch("drive_utils._build_drive_service")
    def test_partial_download_failure_is_reported_not_fatal(self, mock_build):
        service = MagicMock()
        service.files().get().execute.return_value = {
            "id": "folder1_id_1234567890", "name": "Resumes", "mimeType": drive_utils.FOLDER_MIME,
        }
        service.files().list().execute.return_value = {
            "files": [
                {"id": "f1", "name": "good.pdf", "mimeType": "application/pdf"},
                {"id": "f2", "name": "bad.pdf", "mimeType": "application/pdf"},
            ]
        }
        mock_build.return_value = (service, "api_key")

        def fake_download(service, file_meta):
            return None if file_meta["name"] == "bad.pdf" else b"ok-bytes"

        with patch.object(drive_utils, "_drive_api_download", side_effect=fake_download):
            result = fetch_pdfs_from_drive("https://drive.google.com/drive/folders/folder1_id_1234567890")

        self.assertEqual(list(result.keys()), ["good.pdf"])
        self.assertTrue(any("Failed to download: bad.pdf" in w for w in result.warnings))

    @patch("drive_utils._build_drive_service")
    def test_shared_drive_folder_uses_supports_all_drives(self, mock_build):
        service = MagicMock()
        service.files().get().execute.return_value = {
            "id": "shared1_id_1234567890", "name": "TeamDrive Resumes", "mimeType": drive_utils.FOLDER_MIME,
        }
        service.files().list().execute.return_value = {
            "files": [{"id": "f1", "name": "candidate.pdf", "mimeType": "application/pdf"}]
        }
        mock_build.return_value = (service, "service_account")

        with patch.object(drive_utils, "_drive_api_download", return_value=b"bytes"):
            result = fetch_pdfs_from_drive("https://drive.google.com/drive/folders/shared1_id_1234567890")

        self.assertEqual(set(result.keys()), {"candidate.pdf"})
        _, list_kwargs = service.files().list.call_args
        self.assertTrue(list_kwargs.get("supportsAllDrives"))
        self.assertTrue(list_kwargs.get("includeItemsFromAllDrives"))


class TestGdownFallback(unittest.TestCase):
    """When the Drive API and PyDrive2 backends aren't configured at all
    (no credentials/API key/client secrets present), the tool should fall
    back to gdown."""

    @patch("drive_utils._build_drive_service", return_value=(None, None))
    @patch("drive_utils._fetch_via_pydrive2", return_value=None)
    @patch("drive_utils.gdown")
    def test_gdown_public_folder(self, mock_gdown, _mock_pydrive, _mock_build):
        import collections

        Entry = collections.namedtuple("GoogleDriveFileToDownload", ("id", "path", "local_path"))
        mock_gdown.download_folder.return_value = [
            Entry(id="fid1", path="resume.pdf", local_path="resume.pdf"),
            Entry(id="fid2", path="cover.docx", local_path="cover.docx"),
        ]

        def fake_download(url, output, quiet, use_cookies, user_agent=None):
            Path(output).write_bytes(b"fake-bytes-for-" + url.encode())

        mock_gdown.download.side_effect = fake_download

        result = fetch_pdfs_from_drive("https://drive.google.com/drive/folders/pubfolder_id_1234567890")
        self.assertEqual(set(result.keys()), {"resume.pdf", "cover.docx"})

    @patch("drive_utils._build_drive_service", return_value=(None, None))
    @patch("drive_utils._fetch_via_pydrive2", return_value=None)
    @patch("drive_utils.gdown")
    def test_gdown_private_folder_friendly_message(self, mock_gdown, _mock_pydrive, _mock_build):
        mock_gdown.download_folder.side_effect = Exception(
            "Cannot retrieve the public link of the file. "
            "You may need to change the permission to 'Anyone with the link'"
        )

        with self.assertRaises(DriveFetchError) as ctx:
            fetch_pdfs_from_drive("https://drive.google.com/drive/folders/privfolder_id_1234567890")
        self.assertIn("not shared publicly", str(ctx.exception))

    @patch("drive_utils._build_drive_service", return_value=(None, None))
    @patch("drive_utils._fetch_via_pydrive2", return_value=None)
    @patch("drive_utils.gdown")
    def test_gdown_no_downloadable_paths(self, mock_gdown, _mock_pydrive, _mock_build):
        mock_gdown.download_folder.return_value = []
        with self.assertRaises(DriveFetchError) as ctx:
            fetch_pdfs_from_drive("https://drive.google.com/drive/folders/emptyfolder_id_1234567890")
        self.assertIn("No resumes were found", str(ctx.exception))

    @patch("drive_utils.time.sleep")
    @patch("drive_utils._build_drive_service", return_value=(None, None))
    @patch("drive_utils._fetch_via_pydrive2", return_value=None)
    @patch("drive_utils.gdown")
    def test_gdown_one_bad_file_does_not_abort_whole_import(
        self, mock_gdown, _mock_pydrive, _mock_build, _mock_sleep
    ):
        """Regression test for the reported bug: gdown.download_folder() used
        to be all-or-nothing, so one file tripping its HTML scraping (e.g. a
        large-file virus-scan interstitial) took down the entire import even
        though the folder and every other file were fully public."""
        import collections

        Entry = collections.namedtuple("GoogleDriveFileToDownload", ("id", "path", "local_path"))
        mock_gdown.download_folder.return_value = [
            Entry(id="good1", path="good.pdf", local_path="good.pdf"),
            Entry(id="bad1", path="bad.pdf", local_path="bad.pdf"),
        ]

        def fake_download(url, output, quiet, use_cookies, user_agent=None):
            if "bad1" in url:
                raise Exception("Cannot retrieve the public link of the file.")
            Path(output).write_bytes(b"good-bytes")

        mock_gdown.download.side_effect = fake_download

        result = fetch_pdfs_from_drive("https://drive.google.com/drive/folders/pubfolder_id_1234567890")
        self.assertEqual(list(result.keys()), ["good.pdf"])
        self.assertTrue(any("Failed to download: bad.pdf" in w for w in result.warnings))


if __name__ == "__main__":
    unittest.main(verbosity=2)
