"""Google Drive integration for Alfred."""

import io
import logging
from pathlib import Path
from typing import Optional

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from core.security.google_oauth import get_credentials

logger = logging.getLogger(__name__)


def _get_service():
    """Get authenticated Drive service."""
    creds = get_credentials()
    if not creds:
        raise Exception("Google not connected. Please authenticate first.")
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_files(
    folder_id: str = None,
    query: str = None,
    max_results: int = 20,
    file_type: str = None,
) -> list[dict]:
    """List files in Drive or a specific folder.

    Args:
        folder_id: Folder ID to list (None for root/all)
        query: Search query string
        max_results: Maximum files to return
        file_type: Filter by type - 'folder', 'document', 'spreadsheet', 'presentation', 'pdf', 'image'
    """
    service = _get_service()

    # Build query
    q_parts = ["trashed = false"]
    if folder_id:
        q_parts.append(f"'{folder_id}' in parents")
    if query:
        q_parts.append(f"name contains '{query}'")
    if file_type:
        mime_map = {
            "folder": "application/vnd.google-apps.folder",
            "document": "application/vnd.google-apps.document",
            "spreadsheet": "application/vnd.google-apps.spreadsheet",
            "presentation": "application/vnd.google-apps.presentation",
            "pdf": "application/pdf",
            "image": "image/",
        }
        if file_type in mime_map:
            if file_type == "image":
                q_parts.append("mimeType contains 'image/'")
            else:
                q_parts.append(f"mimeType = '{mime_map[file_type]}'")

    results = service.files().list(
        q=" and ".join(q_parts),
        pageSize=max_results,
        fields="files(id, name, mimeType, size, modifiedTime, webViewLink, parents)",
        orderBy="modifiedTime desc",
    ).execute()

    files = results.get("files", [])
    return [
        {
            "id": f["id"],
            "name": f["name"],
            "type": _friendly_type(f["mimeType"]),
            "size": f.get("size"),
            "modified": f.get("modifiedTime"),
            "link": f.get("webViewLink"),
        }
        for f in files
    ]


def _friendly_type(mime_type: str) -> str:
    """Convert MIME type to friendly name."""
    type_map = {
        "application/vnd.google-apps.folder": "folder",
        "application/vnd.google-apps.document": "Google Doc",
        "application/vnd.google-apps.spreadsheet": "Google Sheet",
        "application/vnd.google-apps.presentation": "Google Slides",
        "application/pdf": "PDF",
        "text/plain": "Text",
        "application/json": "JSON",
    }
    if mime_type in type_map:
        return type_map[mime_type]
    if mime_type.startswith("image/"):
        return "Image"
    if mime_type.startswith("video/"):
        return "Video"
    if mime_type.startswith("audio/"):
        return "Audio"
    return mime_type.split("/")[-1]


def search_files(query: str, max_results: int = 20) -> list[dict]:
    """Search for files by name or content.

    Args:
        query: Search query
        max_results: Maximum results to return
    """
    service = _get_service()
    results = service.files().list(
        q=f"fullText contains '{query}' and trashed = false",
        pageSize=max_results,
        fields="files(id, name, mimeType, size, modifiedTime, webViewLink)",
        orderBy="modifiedTime desc",
    ).execute()

    files = results.get("files", [])
    return [
        {
            "id": f["id"],
            "name": f["name"],
            "type": _friendly_type(f["mimeType"]),
            "link": f.get("webViewLink"),
        }
        for f in files
    ]


def get_file(file_id: str) -> dict:
    """Get file metadata.

    Args:
        file_id: The file ID
    """
    service = _get_service()
    f = service.files().get(
        fileId=file_id,
        fields="id, name, mimeType, size, modifiedTime, createdTime, webViewLink, owners, shared",
    ).execute()
    return {
        "id": f["id"],
        "name": f["name"],
        "type": _friendly_type(f["mimeType"]),
        "size": f.get("size"),
        "created": f.get("createdTime"),
        "modified": f.get("modifiedTime"),
        "link": f.get("webViewLink"),
        "shared": f.get("shared", False),
    }


def create_folder(name: str, parent_id: str = None) -> dict:
    """Create a folder in Drive.

    Args:
        name: Folder name
        parent_id: Parent folder ID (None for root)
    """
    service = _get_service()
    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_id:
        metadata["parents"] = [parent_id]

    folder = service.files().create(body=metadata, fields="id, name, webViewLink").execute()
    logger.info(f"Created folder: {name}")
    return {
        "id": folder["id"],
        "name": folder["name"],
        "link": folder.get("webViewLink"),
    }


def upload_file(
    local_path: str,
    name: str = None,
    folder_id: str = None,
    mime_type: str = None,
) -> dict:
    """Upload a file to Drive.

    Args:
        local_path: Local file path
        name: Name for the file in Drive (defaults to local filename)
        folder_id: Folder to upload to (None for root)
        mime_type: MIME type (auto-detected if not provided)
    """
    service = _get_service()
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")

    file_name = name or path.name
    metadata = {"name": file_name}
    if folder_id:
        metadata["parents"] = [folder_id]

    media = MediaFileUpload(local_path, mimetype=mime_type, resumable=True)
    f = service.files().create(body=metadata, media_body=media, fields="id, name, webViewLink").execute()

    logger.info(f"Uploaded file: {file_name}")
    return {
        "id": f["id"],
        "name": f["name"],
        "link": f.get("webViewLink"),
    }


def download_file(file_id: str, local_path: str) -> dict:
    """Download a file from Drive.

    Args:
        file_id: The file ID
        local_path: Local path to save the file
    """
    service = _get_service()

    # Get file metadata to check type
    meta = service.files().get(fileId=file_id, fields="name, mimeType").execute()

    # Handle Google Docs types - export them
    export_map = {
        "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
        "application/vnd.google-apps.spreadsheet": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"),
        "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
    }

    if meta["mimeType"] in export_map:
        export_mime, ext = export_map[meta["mimeType"]]
        request = service.files().export_media(fileId=file_id, mimeType=export_mime)
        if not local_path.endswith(ext):
            local_path += ext
    else:
        request = service.files().get_media(fileId=file_id)

    with open(local_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    logger.info(f"Downloaded: {meta['name']} -> {local_path}")
    return {"success": True, "path": local_path, "name": meta["name"]}


def delete_file(file_id: str) -> dict:
    """Move a file to trash.

    Args:
        file_id: The file ID to delete
    """
    service = _get_service()
    service.files().update(fileId=file_id, body={"trashed": True}).execute()
    logger.info(f"Trashed file: {file_id}")
    return {"success": True, "trashed": file_id}


def share_file(file_id: str, email: str, role: str = "reader") -> dict:
    """Share a file with someone.

    Args:
        file_id: The file ID
        email: Email address to share with
        role: Permission role - 'reader', 'writer', 'commenter'
    """
    service = _get_service()
    permission = {
        "type": "user",
        "role": role,
        "emailAddress": email,
    }
    service.permissions().create(fileId=file_id, body=permission, sendNotificationEmail=True).execute()
    logger.info(f"Shared {file_id} with {email} as {role}")
    return {"success": True, "shared_with": email, "role": role}


def move_file(file_id: str, new_folder_id: str) -> dict:
    """Move a file to a different folder.

    Args:
        file_id: The file ID
        new_folder_id: Destination folder ID
    """
    service = _get_service()
    # Get current parents
    f = service.files().get(fileId=file_id, fields="parents").execute()
    previous_parents = ",".join(f.get("parents", []))

    # Move file
    f = service.files().update(
        fileId=file_id,
        addParents=new_folder_id,
        removeParents=previous_parents,
        fields="id, name, parents",
    ).execute()

    logger.info(f"Moved file {file_id} to folder {new_folder_id}")
    return {"success": True, "id": f["id"], "name": f["name"]}
