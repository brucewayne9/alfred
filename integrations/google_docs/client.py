"""Google Docs integration for Alfred."""

import logging

from googleapiclient.discovery import build

from core.security.google_oauth import get_credentials

logger = logging.getLogger(__name__)


def _get_service():
    """Get authenticated Docs service."""
    creds = get_credentials()
    if not creds:
        raise Exception("Google not connected. Please authenticate first.")
    return build("docs", "v1", credentials=creds, cache_discovery=False)


def _get_drive_service():
    """Get authenticated Drive service for file operations."""
    creds = get_credentials()
    if not creds:
        raise Exception("Google not connected. Please authenticate first.")
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def create_document(title: str, content: str = None, folder_id: str = None) -> dict:
    """Create a new Google Doc.

    Args:
        title: Document title
        content: Initial text content (optional)
        folder_id: Folder to create in (optional)
    """
    docs_service = _get_service()
    drive_service = _get_drive_service()

    # Create the document
    doc = docs_service.documents().create(body={"title": title}).execute()
    doc_id = doc["documentId"]

    # Add initial content if provided
    if content:
        requests = [{"insertText": {"location": {"index": 1}, "text": content}}]
        docs_service.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()

    # Move to folder if specified
    if folder_id:
        f = drive_service.files().get(fileId=doc_id, fields="parents").execute()
        previous_parents = ",".join(f.get("parents", []))
        drive_service.files().update(
            fileId=doc_id,
            addParents=folder_id,
            removeParents=previous_parents,
        ).execute()

    # Get the web link
    f = drive_service.files().get(fileId=doc_id, fields="webViewLink").execute()

    logger.info(f"Created Google Doc: {title}")
    return {
        "id": doc_id,
        "title": title,
        "link": f.get("webViewLink"),
    }


def get_document(document_id: str) -> dict:
    """Get document content and metadata.

    Args:
        document_id: The document ID
    """
    service = _get_service()
    doc = service.documents().get(documentId=document_id).execute()

    # Extract text content
    content = ""
    for element in doc.get("body", {}).get("content", []):
        if "paragraph" in element:
            for para_element in element["paragraph"].get("elements", []):
                if "textRun" in para_element:
                    content += para_element["textRun"].get("content", "")

    return {
        "id": doc["documentId"],
        "title": doc.get("title", ""),
        "content": content.strip(),
        "revision_id": doc.get("revisionId"),
    }


def read_document(document_id: str) -> str:
    """Read the text content of a document.

    Args:
        document_id: The document ID
    """
    doc = get_document(document_id)
    return doc["content"]


def append_text(document_id: str, text: str) -> dict:
    """Append text to the end of a document.

    Args:
        document_id: The document ID
        text: Text to append
    """
    service = _get_service()

    # Get document to find end index
    doc = service.documents().get(documentId=document_id).execute()
    end_index = doc["body"]["content"][-1]["endIndex"] - 1

    # Append text
    requests = [
        {"insertText": {"location": {"index": end_index}, "text": "\n" + text}}
    ]
    service.documents().batchUpdate(documentId=document_id, body={"requests": requests}).execute()

    logger.info(f"Appended text to document {document_id}")
    return {"success": True, "document_id": document_id}


def insert_text(document_id: str, text: str, index: int = 1) -> dict:
    """Insert text at a specific position in a document.

    Args:
        document_id: The document ID
        text: Text to insert
        index: Position to insert at (1 = beginning)
    """
    service = _get_service()
    requests = [{"insertText": {"location": {"index": index}, "text": text}}]
    service.documents().batchUpdate(documentId=document_id, body={"requests": requests}).execute()

    logger.info(f"Inserted text at index {index} in document {document_id}")
    return {"success": True, "document_id": document_id}


def replace_text(document_id: str, find: str, replace: str) -> dict:
    """Find and replace text in a document.

    Args:
        document_id: The document ID
        find: Text to find
        replace: Replacement text
    """
    service = _get_service()
    requests = [
        {
            "replaceAllText": {
                "containsText": {"text": find, "matchCase": True},
                "replaceText": replace,
            }
        }
    ]
    result = service.documents().batchUpdate(documentId=document_id, body={"requests": requests}).execute()

    occurrences = result.get("replies", [{}])[0].get("replaceAllText", {}).get("occurrencesChanged", 0)
    logger.info(f"Replaced {occurrences} occurrences in document {document_id}")
    return {"success": True, "occurrences_replaced": occurrences}


def clear_document(document_id: str) -> dict:
    """Clear all content from a document.

    Args:
        document_id: The document ID
    """
    service = _get_service()

    # Get document to find content range
    doc = service.documents().get(documentId=document_id).execute()
    content = doc["body"]["content"]

    if len(content) <= 1:
        return {"success": True, "message": "Document already empty"}

    # Delete all content except the final newline
    end_index = content[-1]["endIndex"] - 1
    if end_index > 1:
        requests = [{"deleteContentRange": {"range": {"startIndex": 1, "endIndex": end_index}}}]
        service.documents().batchUpdate(documentId=document_id, body={"requests": requests}).execute()

    logger.info(f"Cleared document {document_id}")
    return {"success": True, "document_id": document_id}


def list_documents(max_results: int = 20) -> list[dict]:
    """List Google Docs.

    Args:
        max_results: Maximum documents to return
    """
    drive_service = _get_drive_service()
    results = drive_service.files().list(
        q="mimeType = 'application/vnd.google-apps.document' and trashed = false",
        pageSize=max_results,
        fields="files(id, name, modifiedTime, webViewLink)",
        orderBy="modifiedTime desc",
    ).execute()

    return [
        {
            "id": f["id"],
            "name": f["name"],
            "modified": f.get("modifiedTime"),
            "link": f.get("webViewLink"),
        }
        for f in results.get("files", [])
    ]
