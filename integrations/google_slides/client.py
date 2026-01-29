"""Google Slides integration for Alfred."""

import logging

from googleapiclient.discovery import build

from core.security.google_oauth import get_credentials

logger = logging.getLogger(__name__)


def _get_service():
    """Get authenticated Slides service."""
    creds = get_credentials()
    if not creds:
        raise Exception("Google not connected. Please authenticate first.")
    return build("slides", "v1", credentials=creds, cache_discovery=False)


def _get_drive_service():
    """Get authenticated Drive service for file operations."""
    creds = get_credentials()
    if not creds:
        raise Exception("Google not connected. Please authenticate first.")
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def create_presentation(title: str, folder_id: str = None) -> dict:
    """Create a new Google Slides presentation.

    Args:
        title: Presentation title
        folder_id: Folder to create in (optional)
    """
    service = _get_service()
    drive_service = _get_drive_service()

    body = {"title": title}
    presentation = service.presentations().create(body=body).execute()
    presentation_id = presentation["presentationId"]

    # Move to folder if specified
    if folder_id:
        f = drive_service.files().get(fileId=presentation_id, fields="parents").execute()
        previous_parents = ",".join(f.get("parents", []))
        drive_service.files().update(
            fileId=presentation_id,
            addParents=folder_id,
            removeParents=previous_parents,
        ).execute()

    # Get web link
    f = drive_service.files().get(fileId=presentation_id, fields="webViewLink").execute()

    logger.info(f"Created presentation: {title}")
    return {
        "id": presentation_id,
        "title": title,
        "link": f.get("webViewLink"),
    }


def get_presentation(presentation_id: str) -> dict:
    """Get presentation metadata and slide info.

    Args:
        presentation_id: The presentation ID
    """
    service = _get_service()
    presentation = service.presentations().get(presentationId=presentation_id).execute()

    slides = []
    for slide in presentation.get("slides", []):
        slide_info = {
            "id": slide["objectId"],
            "elements": len(slide.get("pageElements", [])),
        }
        # Try to get title from first text box
        for element in slide.get("pageElements", []):
            if "shape" in element and "text" in element.get("shape", {}):
                text_content = element["shape"]["text"].get("textElements", [])
                for text_elem in text_content:
                    if "textRun" in text_elem:
                        slide_info["title_preview"] = text_elem["textRun"]["content"][:50].strip()
                        break
                if "title_preview" in slide_info:
                    break
        slides.append(slide_info)

    return {
        "id": presentation["presentationId"],
        "title": presentation.get("title", ""),
        "slide_count": len(slides),
        "slides": slides,
    }


def add_slide(
    presentation_id: str,
    layout: str = "BLANK",
    index: int = None,
) -> dict:
    """Add a new slide to a presentation.

    Args:
        presentation_id: The presentation ID
        layout: Slide layout - BLANK, TITLE, TITLE_AND_BODY, TITLE_AND_TWO_COLUMNS, etc.
        index: Position to insert (None = end)
    """
    service = _get_service()

    # Map friendly names to layout IDs
    layout_map = {
        "BLANK": "BLANK",
        "TITLE": "TITLE",
        "TITLE_AND_BODY": "TITLE_AND_BODY",
        "TITLE_AND_TWO_COLUMNS": "TITLE_AND_TWO_COLUMNS",
        "TITLE_ONLY": "TITLE_ONLY",
        "SECTION_HEADER": "SECTION_HEADER",
        "SECTION_TITLE_AND_DESCRIPTION": "SECTION_TITLE_AND_DESCRIPTION",
        "ONE_COLUMN_TEXT": "ONE_COLUMN_TEXT",
        "MAIN_POINT": "MAIN_POINT",
        "BIG_NUMBER": "BIG_NUMBER",
    }

    predefined_layout = layout_map.get(layout.upper(), "BLANK")

    request = {
        "createSlide": {
            "slideLayoutReference": {"predefinedLayout": predefined_layout},
        }
    }
    if index is not None:
        request["createSlide"]["insertionIndex"] = index

    result = service.presentations().batchUpdate(
        presentationId=presentation_id,
        body={"requests": [request]},
    ).execute()

    slide_id = result["replies"][0]["createSlide"]["objectId"]
    logger.info(f"Added slide to presentation {presentation_id}")
    return {"success": True, "slide_id": slide_id}


def add_text_to_slide(
    presentation_id: str,
    slide_id: str,
    text: str,
    x: float = 100,
    y: float = 100,
    width: float = 500,
    height: float = 100,
) -> dict:
    """Add a text box to a slide.

    Args:
        presentation_id: The presentation ID
        slide_id: The slide ID
        text: Text content
        x: X position in points
        y: Y position in points
        width: Width in points
        height: Height in points
    """
    service = _get_service()

    # Create unique element ID
    import uuid
    element_id = f"textbox_{uuid.uuid4().hex[:8]}"

    requests = [
        {
            "createShape": {
                "objectId": element_id,
                "shapeType": "TEXT_BOX",
                "elementProperties": {
                    "pageObjectId": slide_id,
                    "size": {
                        "width": {"magnitude": width, "unit": "PT"},
                        "height": {"magnitude": height, "unit": "PT"},
                    },
                    "transform": {
                        "scaleX": 1,
                        "scaleY": 1,
                        "translateX": x,
                        "translateY": y,
                        "unit": "PT",
                    },
                },
            }
        },
        {
            "insertText": {
                "objectId": element_id,
                "text": text,
                "insertionIndex": 0,
            }
        },
    ]

    service.presentations().batchUpdate(
        presentationId=presentation_id,
        body={"requests": requests},
    ).execute()

    logger.info(f"Added text to slide {slide_id}")
    return {"success": True, "element_id": element_id}


def delete_slide(presentation_id: str, slide_id: str) -> dict:
    """Delete a slide from a presentation.

    Args:
        presentation_id: The presentation ID
        slide_id: The slide ID to delete
    """
    service = _get_service()
    requests = [{"deleteObject": {"objectId": slide_id}}]
    service.presentations().batchUpdate(
        presentationId=presentation_id,
        body={"requests": requests},
    ).execute()

    logger.info(f"Deleted slide {slide_id} from presentation {presentation_id}")
    return {"success": True, "deleted_slide": slide_id}


def duplicate_slide(presentation_id: str, slide_id: str) -> dict:
    """Duplicate a slide.

    Args:
        presentation_id: The presentation ID
        slide_id: The slide ID to duplicate
    """
    service = _get_service()
    requests = [{"duplicateObject": {"objectId": slide_id}}]
    result = service.presentations().batchUpdate(
        presentationId=presentation_id,
        body={"requests": requests},
    ).execute()

    new_slide_id = result["replies"][0]["duplicateObject"]["objectId"]
    logger.info(f"Duplicated slide {slide_id}")
    return {"success": True, "new_slide_id": new_slide_id}


def list_presentations(max_results: int = 20) -> list[dict]:
    """List Google Slides presentations.

    Args:
        max_results: Maximum presentations to return
    """
    drive_service = _get_drive_service()
    results = drive_service.files().list(
        q="mimeType = 'application/vnd.google-apps.presentation' and trashed = false",
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


def get_slide_content(presentation_id: str, slide_id: str) -> dict:
    """Get the text content from a specific slide.

    Args:
        presentation_id: The presentation ID
        slide_id: The slide ID
    """
    service = _get_service()
    presentation = service.presentations().get(presentationId=presentation_id).execute()

    for slide in presentation.get("slides", []):
        if slide["objectId"] == slide_id:
            texts = []
            for element in slide.get("pageElements", []):
                if "shape" in element and "text" in element.get("shape", {}):
                    text_content = ""
                    for text_elem in element["shape"]["text"].get("textElements", []):
                        if "textRun" in text_elem:
                            text_content += text_elem["textRun"]["content"]
                    if text_content.strip():
                        texts.append(text_content.strip())
            return {
                "slide_id": slide_id,
                "texts": texts,
            }

    return {"error": "Slide not found"}
