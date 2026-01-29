"""Google Sheets integration for Alfred."""

import logging
from typing import Optional

from googleapiclient.discovery import build

from core.security.google_oauth import get_credentials

logger = logging.getLogger(__name__)


def _get_service():
    """Get authenticated Sheets service."""
    creds = get_credentials()
    if not creds:
        raise Exception("Google not connected. Please authenticate first.")
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def _get_drive_service():
    """Get authenticated Drive service for file operations."""
    creds = get_credentials()
    if not creds:
        raise Exception("Google not connected. Please authenticate first.")
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def create_spreadsheet(title: str, sheet_names: list[str] = None, folder_id: str = None) -> dict:
    """Create a new Google Spreadsheet.

    Args:
        title: Spreadsheet title
        sheet_names: List of sheet names to create (default: one sheet named 'Sheet1')
        folder_id: Folder to create in (optional)
    """
    service = _get_service()
    drive_service = _get_drive_service()

    # Build sheet properties
    sheets = []
    if sheet_names:
        for i, name in enumerate(sheet_names):
            sheets.append({"properties": {"sheetId": i, "title": name}})
    else:
        sheets.append({"properties": {"sheetId": 0, "title": "Sheet1"}})

    body = {"properties": {"title": title}, "sheets": sheets}
    spreadsheet = service.spreadsheets().create(body=body).execute()
    spreadsheet_id = spreadsheet["spreadsheetId"]

    # Move to folder if specified
    if folder_id:
        f = drive_service.files().get(fileId=spreadsheet_id, fields="parents").execute()
        previous_parents = ",".join(f.get("parents", []))
        drive_service.files().update(
            fileId=spreadsheet_id,
            addParents=folder_id,
            removeParents=previous_parents,
        ).execute()

    logger.info(f"Created spreadsheet: {title}")
    return {
        "id": spreadsheet_id,
        "title": title,
        "link": spreadsheet.get("spreadsheetUrl"),
        "sheets": [s["properties"]["title"] for s in spreadsheet.get("sheets", [])],
    }


def get_spreadsheet(spreadsheet_id: str) -> dict:
    """Get spreadsheet metadata.

    Args:
        spreadsheet_id: The spreadsheet ID
    """
    service = _get_service()
    spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()

    return {
        "id": spreadsheet["spreadsheetId"],
        "title": spreadsheet["properties"]["title"],
        "link": spreadsheet.get("spreadsheetUrl"),
        "sheets": [
            {"id": s["properties"]["sheetId"], "title": s["properties"]["title"]}
            for s in spreadsheet.get("sheets", [])
        ],
    }


def read_range(spreadsheet_id: str, range_notation: str) -> list[list]:
    """Read data from a range of cells.

    Args:
        spreadsheet_id: The spreadsheet ID
        range_notation: A1 notation range (e.g., 'Sheet1!A1:D10' or 'A1:D10')
    """
    service = _get_service()
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=range_notation,
    ).execute()

    return result.get("values", [])


def read_sheet(spreadsheet_id: str, sheet_name: str = "Sheet1", max_rows: int = 100) -> list[list]:
    """Read all data from a sheet.

    Args:
        spreadsheet_id: The spreadsheet ID
        sheet_name: Name of the sheet (default: Sheet1)
        max_rows: Maximum rows to read
    """
    return read_range(spreadsheet_id, f"{sheet_name}!A1:ZZ{max_rows}")


def write_range(spreadsheet_id: str, range_notation: str, values: list[list]) -> dict:
    """Write data to a range of cells.

    Args:
        spreadsheet_id: The spreadsheet ID
        range_notation: A1 notation range (e.g., 'Sheet1!A1')
        values: 2D list of values to write
    """
    service = _get_service()
    body = {"values": values}
    result = service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=range_notation,
        valueInputOption="USER_ENTERED",
        body=body,
    ).execute()

    logger.info(f"Wrote {result.get('updatedCells')} cells to {range_notation}")
    return {
        "success": True,
        "updated_cells": result.get("updatedCells"),
        "updated_range": result.get("updatedRange"),
    }


def append_rows(spreadsheet_id: str, values: list[list], sheet_name: str = "Sheet1") -> dict:
    """Append rows to the end of a sheet.

    Args:
        spreadsheet_id: The spreadsheet ID
        values: 2D list of rows to append
        sheet_name: Name of the sheet (default: Sheet1)
    """
    service = _get_service()
    body = {"values": values}
    result = service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=f"{sheet_name}!A1",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body=body,
    ).execute()

    updates = result.get("updates", {})
    logger.info(f"Appended {len(values)} rows to {sheet_name}")
    return {
        "success": True,
        "updated_rows": updates.get("updatedRows"),
        "updated_range": updates.get("updatedRange"),
    }


def append_row(spreadsheet_id: str, values: list, sheet_name: str = "Sheet1") -> dict:
    """Append a single row to the end of a sheet.

    Args:
        spreadsheet_id: The spreadsheet ID
        values: List of values for the row
        sheet_name: Name of the sheet (default: Sheet1)
    """
    return append_rows(spreadsheet_id, [values], sheet_name)


def clear_range(spreadsheet_id: str, range_notation: str) -> dict:
    """Clear values from a range of cells.

    Args:
        spreadsheet_id: The spreadsheet ID
        range_notation: A1 notation range to clear
    """
    service = _get_service()
    service.spreadsheets().values().clear(
        spreadsheetId=spreadsheet_id,
        range=range_notation,
    ).execute()

    logger.info(f"Cleared range {range_notation}")
    return {"success": True, "cleared_range": range_notation}


def add_sheet(spreadsheet_id: str, sheet_name: str) -> dict:
    """Add a new sheet to an existing spreadsheet.

    Args:
        spreadsheet_id: The spreadsheet ID
        sheet_name: Name for the new sheet
    """
    service = _get_service()
    requests = [{"addSheet": {"properties": {"title": sheet_name}}}]
    result = service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": requests},
    ).execute()

    new_sheet = result["replies"][0]["addSheet"]["properties"]
    logger.info(f"Added sheet '{sheet_name}' to spreadsheet {spreadsheet_id}")
    return {
        "success": True,
        "sheet_id": new_sheet["sheetId"],
        "title": new_sheet["title"],
    }


def delete_sheet(spreadsheet_id: str, sheet_id: int) -> dict:
    """Delete a sheet from a spreadsheet.

    Args:
        spreadsheet_id: The spreadsheet ID
        sheet_id: The sheet ID (not name) to delete
    """
    service = _get_service()
    requests = [{"deleteSheet": {"sheetId": sheet_id}}]
    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": requests},
    ).execute()

    logger.info(f"Deleted sheet {sheet_id} from spreadsheet {spreadsheet_id}")
    return {"success": True, "deleted_sheet_id": sheet_id}


def list_spreadsheets(max_results: int = 20) -> list[dict]:
    """List Google Spreadsheets.

    Args:
        max_results: Maximum spreadsheets to return
    """
    drive_service = _get_drive_service()
    results = drive_service.files().list(
        q="mimeType = 'application/vnd.google-apps.spreadsheet' and trashed = false",
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


def find_row(spreadsheet_id: str, sheet_name: str, column: str, value: str) -> Optional[int]:
    """Find the row number where a column contains a specific value.

    Args:
        spreadsheet_id: The spreadsheet ID
        sheet_name: Sheet name
        column: Column letter (e.g., 'A')
        value: Value to search for

    Returns:
        Row number (1-indexed) or None if not found
    """
    data = read_range(spreadsheet_id, f"{sheet_name}!{column}:{column}")
    for i, row in enumerate(data, start=1):
        if row and str(row[0]).strip() == str(value).strip():
            return i
    return None
