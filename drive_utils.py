"""Google Drive folder import for the ATS Resume Shortlisting Tool.

Resolution order (most reliable first), each one only attempted if the one
before it isn't usable or fails to find anything:

  1. Google Drive API v3 with a service account or a cached OAuth token
     (``GOOGLE_APPLICATION_CREDENTIALS`` / ``GOOGLE_SERVICE_ACCOUNT_FILE`` or
     ``token.json``) -- most reliable, works for private and Shared Drive
     folders the credential has access to.
  2. Google Drive API v3 with a plain API key (``GOOGLE_DRIVE_API_KEY`` /
     ``GOOGLE_API_KEY``) -- reliable for publicly shared folders, no OAuth
     flow required.
  3. PyDrive2, if a previously-saved, non-interactive credential is present.
  4. gdown -- unauthenticated HTML scraping. Works for simple public folders
     but is fragile (Google regularly changes the markup gdown parses, and
     unauthenticated requests get rate-limited). Used only as a last resort.
"""

import io
import logging
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ats_tool.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx"}

FOLDER_MIME = "application/vnd.google-apps.folder"

# Native Google Docs have no file extension and can't be downloaded directly;
# they must be exported to a real document format first.
GOOGLE_DOC_EXPORT = {
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx",
    ),
}

NOT_SHARED_PUBLICLY_MESSAGE = (
    "This folder is not shared publicly. Change Google Drive sharing to "
    "'Anyone with the link -> Viewer' and try again."
)


class DriveFetchError(Exception):
    """Raised when resumes can't be fetched from a Google Drive link.

    Messages on this exception are written to be shown to end users directly
    (no raw library/API error text)."""


class DriveFetchResult(dict):
    """A ``{filename: file_bytes}`` dict that also carries non-fatal import
    warnings (skipped/failed files) alongside the successfully fetched
    resumes. Kept as a plain dict subclass so existing code that just treats
    the return value as ``Dict[str, bytes]`` keeps working unmodified."""

    def __init__(
        self,
        *args,
        warnings: Optional[List[str]] = None,
        file_meta: Optional[Dict[str, Dict[str, str]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.warnings: List[str] = warnings or []
        # filename -> {"file_id": str, "url": str}; used to populate the
        # "Google Drive File ID" / "Google Drive URL" columns downstream.
        self.file_meta: Dict[str, Dict[str, str]] = file_meta or {}


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

_FOLDER_ID_PATTERNS = [
    re.compile(r"/folders/([a-zA-Z0-9_-]{10,})"),
    re.compile(r"[?&]id=([a-zA-Z0-9_-]{10,})"),
]


def extract_folder_id(drive_link: str) -> str:
    """Extracts a folder ID from any common Google Drive folder link shape,
    e.g. .../drive/folders/<id>, .../drive/folders/<id>?usp=sharing,
    .../drive/u/0/folders/<id>, .../open?id=<id>, or a bare folder ID."""
    if not drive_link or not drive_link.strip():
        raise DriveFetchError("Please paste a Google Drive folder link first.")

    link = drive_link.strip().split("#", 1)[0]  # drop URL fragment

    # Tolerate the link being pasted alongside other text.
    url_match = re.search(r"https?://\S*drive\.google\.com\S*", link)
    if url_match:
        link = url_match.group(0)

    for pattern in _FOLDER_ID_PATTERNS:
        match = pattern.search(link)
        if match:
            return match.group(1)

    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", link):
        return link

    raise DriveFetchError(
        "Couldn't find a folder ID in that link. Paste a Google Drive folder "
        "share link, e.g. https://drive.google.com/drive/folders/XXXXXXXXXXXX"
    )


# ---------------------------------------------------------------------------
# Helpers shared by every backend
# ---------------------------------------------------------------------------


def _resolved_filename(name: str, mime_type: str) -> Optional[str]:
    """Returns the filename resumes should be saved under, or None if the
    file's type isn't one we import (unsupported -> caller should skip it)."""
    if mime_type in GOOGLE_DOC_EXPORT:
        _export_mime, ext = GOOGLE_DOC_EXPORT[mime_type]
        return name if name.lower().endswith(ext) else name + ext

    if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS:
        return name

    return None


# ---------------------------------------------------------------------------
# Backend 1 & 2: Google Drive API v3 (service account / OAuth / API key)
# ---------------------------------------------------------------------------


def _load_credentials():
    """Returns (credentials, mode) with mode in {"service_account", "oauth"},
    or (None, None) if no usable credential is configured."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials as UserCredentials
    except ImportError:
        return None, None

    scopes = ["https://www.googleapis.com/auth/drive.readonly"]

    sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get(
        "GOOGLE_SERVICE_ACCOUNT_FILE"
    )
    if sa_path and Path(sa_path).exists():
        try:
            creds = service_account.Credentials.from_service_account_file(sa_path, scopes=scopes)
            return creds, "service_account"
        except Exception as e:
            logger.warning(f"Found service account file at {sa_path} but couldn't load it: {e}")

    token_path = Path(os.environ.get("GOOGLE_OAUTH_TOKEN_FILE", "token.json"))
    if token_path.exists():
        try:
            creds = UserCredentials.from_authorized_user_file(str(token_path), scopes=scopes)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            return creds, "oauth"
        except Exception as e:
            logger.warning(f"Found OAuth token file at {token_path} but couldn't load it: {e}")

    return None, None


def _build_drive_service():
    """Returns (service, auth_mode) or (None, None) if the Drive API v3 path
    isn't usable (library missing, or no credential/API key configured)."""
    try:
        from googleapiclient.discovery import build
    except ImportError:
        logger.info("google-api-python-client not installed; skipping Drive API path.")
        return None, None

    creds, mode = _load_credentials()
    if creds is not None:
        return build("drive", "v3", credentials=creds, cache_discovery=False), mode

    api_key = os.environ.get("GOOGLE_DRIVE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return build("drive", "v3", developerKey=api_key, cache_discovery=False), "api_key"

    return None, None


def _classify_http_error(e) -> str:
    """Turns a googleapiclient HttpError into a message safe to show users,
    distinguishing "your API key/project is misconfigured" (an operator
    problem) from "this folder isn't shared with us" (a data problem)."""
    status = getattr(e.resp, "status", None)
    body = ""
    try:
        body = e.content.decode("utf-8", errors="ignore") if e.content else ""
    except Exception:
        pass
    body_lower = body.lower()

    if any(
        marker in body_lower
        for marker in ("api_key_invalid", "accessnotconfigured", "service_disabled", "api not enabled")
    ):
        return (
            "Google Drive API access isn't configured correctly (invalid API "
            "key or the Drive API isn't enabled for it). Check the "
            "GOOGLE_DRIVE_API_KEY / credential configuration."
        )
    if status == 429 or "rate" in body_lower:
        return "Google Drive is rate-limiting requests right now. Please wait a minute and try again."
    if status in (403, 404):
        return NOT_SHARED_PUBLICLY_MESSAGE
    return f"Google Drive API error while accessing the folder (HTTP {status})."


def _drive_api_list_files(service, folder_id: str) -> Tuple[List[dict], str]:
    """Validates the folder is accessible, then recursively lists every file
    inside it (and its subfolders). Raises DriveFetchError for permission /
    not-found / rate-limit problems."""
    from googleapiclient.errors import HttpError

    try:
        meta = service.files().get(
            fileId=folder_id, fields="id, name, mimeType", supportsAllDrives=True
        ).execute()
    except HttpError as e:
        raise DriveFetchError(_classify_http_error(e))

    if meta.get("mimeType") != FOLDER_MIME:
        raise DriveFetchError(
            "That link points to a single file, not a folder. Paste a link to "
            "the folder containing your resumes."
        )

    all_files: List[dict] = []
    stack = [folder_id]
    seen_folders = set()

    while stack:
        current = stack.pop()
        if current in seen_folders:
            continue
        seen_folders.add(current)

        page_token = None
        while True:
            try:
                response = service.files().list(
                    q=f"'{current}' in parents and trashed = false",
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageSize=1000,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                ).execute()
            except HttpError as e:
                # A subfolder becoming unreadable shouldn't abort the whole
                # import -- skip it and keep whatever else was found.
                logger.warning(f"Couldn't list subfolder {current}: {e}")
                break

            for f in response.get("files", []):
                if f["mimeType"] == FOLDER_MIME:
                    stack.append(f["id"])
                else:
                    all_files.append(f)

            page_token = response.get("nextPageToken")
            if not page_token:
                break

    return all_files, meta.get("name", folder_id)


def _drive_api_download(service, file_meta: dict) -> Optional[bytes]:
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseDownload

    file_id = file_meta["id"]
    mime_type = file_meta["mimeType"]

    try:
        if mime_type in GOOGLE_DOC_EXPORT:
            export_mime, _ext = GOOGLE_DOC_EXPORT[mime_type]
            request = service.files().export_media(fileId=file_id, mimeType=export_mime)
        else:
            request = service.files().get_media(fileId=file_id, supportsAllDrives=True)

        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _status, done = downloader.next_chunk()
        return buffer.getvalue()
    except HttpError as e:
        logger.error(f"Drive API download failed for {file_meta.get('name')}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error downloading {file_meta.get('name')}: {e}")
        return None


_DRIVE_API_MAX_WORKERS = 8
_thread_local_drive = threading.local()


def _thread_local_service(fallback_service):
    """A Drive service instance private to the calling thread. googleapiclient
    Resource objects wrap an httplib2.Http connection that isn't safe to share
    across threads, so each download worker gets its own (built once per
    thread, then reused for every file that thread handles). Falls back to
    the caller's shared service if a fresh one can't be built for some reason
    (e.g. ``_build_drive_service`` mocked to a fixed value in tests, where
    sharing is harmless since nothing real is on the wire)."""
    if not hasattr(_thread_local_drive, "service"):
        fresh_service, _mode = _build_drive_service()
        _thread_local_drive.service = fresh_service if fresh_service is not None else fallback_service
    return _thread_local_drive.service


def _fetch_via_drive_api(folder_id: str) -> Optional[DriveFetchResult]:
    """Returns a DriveFetchResult, or None if the Drive API isn't configured
    at all (so the caller should fall through to the next backend). Raises
    DriveFetchError for permission/not-found problems -- those are definitive
    answers, not a reason to fall back to weaker backends."""
    service, auth_mode = _build_drive_service()
    if service is None:
        return None

    logger.info(f"Using Google Drive API (auth_mode={auth_mode}) for folder {folder_id}")

    files, folder_name = _drive_api_list_files(service, folder_id)
    logger.info(f"Drive API discovered {len(files)} item(s) in '{folder_name}' ({folder_id})")

    if not files:
        raise DriveFetchError(
            "No resumes were found in that Drive folder. Check that it contains "
            "PDF, DOC, or DOCX files."
        )

    resumes: Dict[str, bytes] = {}
    file_meta: Dict[str, Dict[str, str]] = {}
    warnings: List[str] = []
    skipped_unsupported = 0

    download_targets = []
    for f in files:
        filename = _resolved_filename(f["name"], f["mimeType"])
        if filename is None:
            skipped_unsupported += 1
            continue
        download_targets.append((filename, f))

    def _download_one(item):
        filename, f = item
        thread_service = _thread_local_service(service)
        return filename, f, _drive_api_download(thread_service, f)

    # Downloads are I/O-bound (waiting on network/Drive API), so a modest
    # thread pool turns a large folder from "one file at a time, ~1.5s each"
    # into several files in flight at once -- a 146-file folder drops from
    # ~3 minutes to well under 30s.
    with ThreadPoolExecutor(max_workers=_DRIVE_API_MAX_WORKERS) as executor:
        for filename, f, content in executor.map(_download_one, download_targets):
            if content is None:
                warnings.append(f"Failed to download: {f['name']}")
                continue
            resumes[filename] = content
            file_meta[filename] = {
                "file_id": f["id"],
                "url": f"https://drive.google.com/file/d/{f['id']}/view",
            }

    if not resumes and not warnings:
        raise DriveFetchError(
            f"Found {len(files)} file(s) in the folder, but none of them were "
            "PDF, DOC, or DOCX resumes."
        )

    if skipped_unsupported:
        warnings.append(f"{skipped_unsupported} file(s) skipped (unsupported type)")

    logger.info(
        f"Drive API import complete: {len(resumes)} downloaded, {len(warnings)} "
        f"warning(s), {skipped_unsupported} unsupported"
    )
    return DriveFetchResult(resumes, warnings=warnings, file_meta=file_meta)


# ---------------------------------------------------------------------------
# Backend 3: PyDrive2 (only if a non-interactive credential is already saved)
# ---------------------------------------------------------------------------


def _fetch_via_pydrive2(folder_id: str) -> Optional[DriveFetchResult]:
    try:
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
    except ImportError:
        return None

    client_secrets = Path(os.environ.get("PYDRIVE2_CLIENT_SECRETS", "client_secrets.json"))
    saved_creds = Path(os.environ.get("PYDRIVE2_CREDENTIALS", "pydrive2_credentials.json"))
    if not client_secrets.exists() or not saved_creds.exists():
        # No pre-existing, non-interactive credential -- can't do an OAuth
        # consent flow from inside a server process, so skip this backend.
        return None

    try:
        gauth = GoogleAuth()
        gauth.LoadClientConfigFile(str(client_secrets))
        gauth.LoadCredentialsFile(str(saved_creds))
        if gauth.credentials is None:
            return None
        if gauth.access_token_expired:
            gauth.Refresh()
        else:
            gauth.Authorize()
        gauth.SaveCredentialsFile(str(saved_creds))
        drive = GoogleDrive(gauth)
    except Exception as e:
        logger.warning(f"PyDrive2 credentials present but couldn't authorize: {e}")
        return None

    logger.info(f"Using PyDrive2 for folder {folder_id}")

    try:
        file_list = drive.ListFile(
            {"q": f"'{folder_id}' in parents and trashed=false"}
        ).GetList()
    except Exception as e:
        logger.warning(f"PyDrive2 failed to list folder {folder_id}: {e}")
        return None

    if not file_list:
        raise DriveFetchError(
            "No resumes were found in that Drive folder. Check that it contains "
            "PDF, DOC, or DOCX files."
        )

    resumes: Dict[str, bytes] = {}
    file_meta: Dict[str, Dict[str, str]] = {}
    warnings: List[str] = []
    skipped_unsupported = 0

    for entry in file_list:
        name = entry.get("title", "")
        if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped_unsupported += 1
            continue
        try:
            entry.FetchContent()
            resumes[name] = entry.content.read()
            file_id = entry.get("id", "")
            file_meta[name] = {
                "file_id": file_id,
                "url": f"https://drive.google.com/file/d/{file_id}/view" if file_id else "",
            }
        except Exception as e:
            logger.error(f"PyDrive2 download failed for {name}: {e}")
            warnings.append(f"Failed to download: {name}")

    if not resumes and not warnings:
        raise DriveFetchError(
            f"Found {len(file_list)} file(s) in the folder, but none of them "
            "were PDF, DOC, or DOCX resumes."
        )

    if skipped_unsupported:
        warnings.append(f"{skipped_unsupported} file(s) skipped (unsupported type)")

    return DriveFetchResult(resumes, warnings=warnings, file_meta=file_meta)


# ---------------------------------------------------------------------------
# Backend 4: gdown (last resort -- unauthenticated HTML scraping)
# ---------------------------------------------------------------------------

# gdown's own default User-Agent for individual file downloads is a Chrome 39
# string from 2014. Google's anti-automation system flags it and returns a
# 403 "your computer or network may be sending automated queries" page
# instead of the file -- which gdown then can't parse, surfacing as
# "Cannot retrieve the public link of the file" for every single file, even
# on a folder that's fully public. A current, ordinary-looking User-Agent
# avoids that block entirely.
_MODERN_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# gdown is the last-resort, unauthenticated backend -- a couple of retries
# with backoff softens transient failures (a single file briefly rate-limited)
# without meaningfully slowing down a folder where everything just works.
_GDOWN_MAX_ATTEMPTS = 3
_GDOWN_RETRY_DELAY_SECONDS = 2


def _classify_gdown_error(e: Exception) -> str:
    message = str(e).lower()
    if any(word in message for word in ("permission", "private", "access denied", "cannot retrieve")):
        return NOT_SHARED_PUBLICLY_MESSAGE
    if any(word in message for word in ("429", "rate", "too many requests")):
        return (
            "Google Drive is rate-limiting requests right now. Please wait a "
            "minute and try again."
        )
    return (
        "Could not access that Drive folder. Make sure it's shared as 'Anyone "
        "with the link' and that the link points to a folder (not a single file)."
    )


def _fetch_via_gdown(folder_id: str) -> DriveFetchResult:
    """gdown.download_folder() downloads its whole batch as one all-or-nothing
    operation: if a single file trips up its HTML scraping (e.g. Google's
    "can't scan this file for viruses" interstitial on large files), the
    *entire* folder import aborts, even though the folder is public and every
    other file would have downloaded fine. To avoid that, list the folder's
    contents first (skip_download=True, no actual downloading), then download
    each file independently so one bad file just gets skipped and logged
    instead of failing the whole import."""
    logger.info(f"Falling back to gdown for folder {folder_id}")
    url = f"https://drive.google.com/drive/folders/{folder_id}"

    try:
        entries = gdown.download_folder(
            url=url,
            quiet=True,
            use_cookies=False,
            skip_download=True,
            user_agent=_MODERN_USER_AGENT,
        )
    except Exception as e:
        logger.error(f"gdown failed to list folder {folder_id}: {e}")
        raise DriveFetchError(_classify_gdown_error(e))

    if not entries:
        raise DriveFetchError(
            "No resumes were found in that Drive folder. Check that it's "
            "shared publicly ('Anyone with the link can view') and that the "
            "link is correct."
        )

    supported_entries = [e for e in entries if Path(e.path).suffix.lower() in SUPPORTED_EXTENSIONS]
    skipped_unsupported = len(entries) - len(supported_entries)

    if not supported_entries:
        raise DriveFetchError(
            f"Found {len(entries)} file(s) in the folder, but none of them "
            "were PDF, DOC, or DOCX resumes."
        )

    resumes: Dict[str, bytes] = {}
    file_meta: Dict[str, Dict[str, str]] = {}
    warnings: List[str] = []

    with tempfile.TemporaryDirectory() as staging_dir:
        for entry in supported_entries:
            dest = Path(staging_dir, entry.path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            last_error = None
            for attempt in range(_GDOWN_MAX_ATTEMPTS):
                if attempt > 0:
                    time.sleep(_GDOWN_RETRY_DELAY_SECONDS * attempt)
                try:
                    gdown.download(
                        url=f"https://drive.google.com/uc?id={entry.id}",
                        output=str(dest),
                        quiet=True,
                        use_cookies=False,
                        user_agent=_MODERN_USER_AGENT,
                    )
                    resumes[dest.name] = dest.read_bytes()
                    file_meta[dest.name] = {
                        "file_id": entry.id,
                        "url": f"https://drive.google.com/uc?id={entry.id}",
                    }
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
            if last_error is not None:
                logger.error(f"gdown failed to download {entry.path} ({entry.id}): {last_error}")
                warnings.append(f"Failed to download: {entry.path}")

    if not resumes:
        raise DriveFetchError(
            f"Found {len(entries)} file(s) in the folder, but every download "
            "attempt failed. The folder may have hit Google's per-file access "
            "limits -- wait a few minutes and try again."
        )

    if skipped_unsupported:
        warnings.append(f"{skipped_unsupported} file(s) skipped (unsupported type)")

    logger.info(f"gdown import complete: {len(resumes)} downloaded, {skipped_unsupported} unsupported")
    return DriveFetchResult(resumes, warnings=warnings, file_meta=file_meta)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fetch_pdfs_from_drive(drive_link: str) -> DriveFetchResult:
    """Downloads every supported resume (PDF, DOC, DOCX) from a Google Drive
    folder link and returns a ``{filename: file_bytes}`` mapping (as a
    DriveFetchResult, a dict subclass that also exposes a ``.warnings`` list
    for partial failures).

    Tries, in order: the Google Drive API (service account / OAuth / API
    key), PyDrive2, then gdown. The folder must be shared as 'Anyone with the
    link can view' unless a service account or OAuth credential has been
    granted direct access to it."""
    folder_id = extract_folder_id(drive_link)
    logger.info(f"Drive import requested for folder_id={folder_id}")

    result = _fetch_via_drive_api(folder_id)
    if result is not None:
        return result

    result = _fetch_via_pydrive2(folder_id)
    if result is not None:
        return result

    return _fetch_via_gdown(folder_id)
