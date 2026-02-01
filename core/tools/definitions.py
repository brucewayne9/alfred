"""Tool definitions - registers all available tools for the LLM."""

from core.tools.registry import tool


# ==================== EMAIL TOOLS ====================

@tool(
    name="check_email",
    description="Check inbox for recent emails. Returns subject, sender, and snippet.",
    parameters={"max_results": "int (default 5)", "query": "string - Gmail search query (optional)"},
)
def check_email(max_results: int = 5, query: str = "") -> list[dict]:
    from integrations.gmail.client import get_inbox
    return get_inbox(max_results=max_results, query=query)


@tool(
    name="read_email",
    description="Read the full content of a specific email by its ID.",
    parameters={"message_id": "string - the email message ID"},
)
def read_email(message_id: str) -> dict:
    from integrations.gmail.client import read_email as _read
    return _read(message_id)


@tool(
    name="send_email",
    description="Send an email to someone.",
    parameters={"to": "string - recipient email", "subject": "string", "body": "string"},
)
def send_email(to: str, subject: str, body: str) -> dict:
    from integrations.gmail.client import send_email as _send
    return _send(to, subject, body)


@tool(
    name="unread_email_count",
    description="Get the number of unread emails in inbox.",
    parameters={},
)
def unread_email_count() -> dict:
    from integrations.gmail.client import get_unread_count
    return {"unread": get_unread_count()}


# ==================== CALENDAR TOOLS ====================

@tool(
    name="check_calendar",
    description="Get upcoming calendar events for the next N days.",
    parameters={"days_ahead": "int (default 7)", "max_results": "int (default 10)"},
)
def check_calendar(days_ahead: int = 7, max_results: int = 10) -> list[dict]:
    from integrations.calendar.client import get_upcoming_events
    return get_upcoming_events(max_results=max_results, days_ahead=days_ahead)


@tool(
    name="today_schedule",
    description="Get today's schedule - all events for today.",
    parameters={},
)
def today_schedule() -> list[dict]:
    from integrations.calendar.client import get_today_events
    return get_today_events()


@tool(
    name="create_event",
    description="Create a new calendar event.",
    parameters={
        "summary": "string - event title",
        "start_time": "string - ISO datetime (e.g. 2026-01-28T10:00:00-05:00)",
        "end_time": "string - ISO datetime",
        "description": "string (optional)",
        "location": "string (optional)",
    },
)
def create_event(summary: str, start_time: str, end_time: str, description: str = "", location: str = "") -> dict:
    from integrations.calendar.client import create_event as _create
    return _create(summary, start_time, end_time, description, location)


@tool(
    name="find_free_time",
    description="Find available time slots on a given date.",
    parameters={"date": "string - YYYY-MM-DD", "duration_minutes": "int (default 60)"},
)
def find_free_time(date: str, duration_minutes: int = 60) -> list[dict]:
    from integrations.calendar.client import find_free_time as _find
    return _find(date, duration_minutes)


# ==================== SERVER TOOLS ====================

@tool(
    name="list_servers",
    description="List all registered servers that Alfred can manage.",
    parameters={},
)
def list_servers() -> list[dict]:
    from integrations.servers.manager import list_servers as _list
    return _list()


@tool(
    name="server_status",
    description="Get the status of a specific server (uptime, disk, memory, docker containers).",
    parameters={"server_name": "string - name of the registered server"},
)
def server_status(server_name: str) -> dict:
    from integrations.servers.manager import get_server_status
    return get_server_status(server_name)


@tool(
    name="server_command",
    description="Run a shell command on a remote server. Use with caution.",
    parameters={"server_name": "string", "command": "string - the command to run"},
)
def server_command(server_name: str, command: str) -> dict:
    from integrations.servers.manager import run_command
    return run_command(server_name, command)


@tool(
    name="docker_containers",
    description="List Docker containers on a remote server.",
    parameters={"server_name": "string"},
)
def docker_containers(server_name: str) -> str:
    from integrations.servers.manager import docker_ps
    return docker_ps(server_name)


@tool(
    name="docker_restart",
    description="Restart a Docker container on a remote server.",
    parameters={"server_name": "string", "container": "string - container name"},
)
def docker_restart(server_name: str, container: str) -> str:
    from integrations.servers.manager import docker_restart as _restart
    return _restart(server_name, container)


# ==================== CRM TOOLS (Twenty CRM) ====================

@tool(
    name="crm_list_people",
    description="List contacts/people in the CRM.",
    parameters={"limit": "int (default 20)"},
)
def crm_list_people(limit: int = 20) -> list[dict]:
    from integrations.base_crm.client import list_people
    return list_people(limit=limit)


@tool(
    name="crm_search_people",
    description="Search for a person in the CRM by name or email.",
    parameters={"query": "string - name or email to search for"},
)
def crm_search_people(query: str) -> list[dict]:
    from integrations.base_crm.client import search_people
    return search_people(query)


@tool(
    name="crm_get_person",
    description="Get full details of a specific person/contact in the CRM by their ID.",
    parameters={"person_id": "string - UUID of the person"},
)
def crm_get_person(person_id: str) -> dict:
    from integrations.base_crm.client import get_person
    return get_person(person_id)


@tool(
    name="crm_create_person",
    description="Add a new contact/person to the CRM.",
    parameters={
        "first_name": "string",
        "last_name": "string",
        "email": "string (optional)",
        "phone": "string (optional)",
        "job_title": "string (optional)",
        "city": "string (optional)",
    },
)
def crm_create_person(first_name: str, last_name: str, email: str = "",
                       phone: str = "", job_title: str = "", city: str = "") -> dict:
    from integrations.base_crm.client import create_person
    return create_person(first_name, last_name, email, phone, job_title, city)


@tool(
    name="crm_update_person",
    description="Update an existing person/contact in the CRM. Only provide fields you want to change.",
    parameters={
        "person_id": "string - UUID of the person",
        "first_name": "string (optional)",
        "last_name": "string (optional)",
        "email": "string (optional)",
        "phone": "string (optional)",
        "job_title": "string (optional)",
        "city": "string (optional)",
    },
)
def crm_update_person(person_id: str, first_name: str = "", last_name: str = "",
                       email: str = "", phone: str = "", job_title: str = "", city: str = "") -> dict:
    from integrations.base_crm.client import update_person
    return update_person(person_id, first_name, last_name, email, phone, job_title, city)


@tool(
    name="crm_delete_person",
    description="Delete a person/contact from the CRM.",
    parameters={"person_id": "string - UUID of the person to delete"},
)
def crm_delete_person(person_id: str) -> dict:
    from integrations.base_crm.client import delete_person
    return delete_person(person_id)


@tool(
    name="crm_list_companies",
    description="List companies in the CRM.",
    parameters={"limit": "int (default 20)"},
)
def crm_list_companies(limit: int = 20) -> list[dict]:
    from integrations.base_crm.client import list_companies
    return list_companies(limit=limit)


@tool(
    name="crm_search_companies",
    description="Search for a company in the CRM by name.",
    parameters={"query": "string - company name to search for"},
)
def crm_search_companies(query: str) -> list[dict]:
    from integrations.base_crm.client import search_companies
    return search_companies(query)


@tool(
    name="crm_get_company",
    description="Get full details of a specific company in the CRM by its ID.",
    parameters={"company_id": "string - UUID of the company"},
)
def crm_get_company(company_id: str) -> dict:
    from integrations.base_crm.client import get_company
    return get_company(company_id)


@tool(
    name="crm_create_company",
    description="Add a new company to the CRM.",
    parameters={
        "name": "string - company name",
        "domain": "string - website URL (optional)",
        "employees": "int (optional)",
        "city": "string (optional)",
    },
)
def crm_create_company(name: str, domain: str = "", employees: int = 0, city: str = "") -> dict:
    from integrations.base_crm.client import create_company
    return create_company(name, domain, employees, city)


@tool(
    name="crm_update_company",
    description="Update an existing company in the CRM. Only provide fields you want to change.",
    parameters={
        "company_id": "string - UUID of the company",
        "name": "string (optional)",
        "domain": "string - website URL (optional)",
        "employees": "int (optional)",
        "city": "string (optional)",
    },
)
def crm_update_company(company_id: str, name: str = "", domain: str = "",
                        employees: int = 0, city: str = "") -> dict:
    from integrations.base_crm.client import update_company
    return update_company(company_id, name, domain, employees, city)


@tool(
    name="crm_delete_company",
    description="Delete a company from the CRM.",
    parameters={"company_id": "string - UUID of the company to delete"},
)
def crm_delete_company(company_id: str) -> dict:
    from integrations.base_crm.client import delete_company
    return delete_company(company_id)


@tool(
    name="crm_list_opportunities",
    description="List deals/opportunities in the CRM pipeline.",
    parameters={"limit": "int (default 20)"},
)
def crm_list_opportunities(limit: int = 20) -> list[dict]:
    from integrations.base_crm.client import list_opportunities
    return list_opportunities(limit=limit)


@tool(
    name="crm_search_opportunities",
    description="Search deals/opportunities in the CRM by name or stage.",
    parameters={"query": "string - deal name or stage to search for"},
)
def crm_search_opportunities(query: str) -> list[dict]:
    from integrations.base_crm.client import search_opportunities
    return search_opportunities(query)


@tool(
    name="crm_get_opportunity",
    description="Get full details of a specific deal/opportunity by its ID.",
    parameters={"opportunity_id": "string - UUID of the opportunity"},
)
def crm_get_opportunity(opportunity_id: str) -> dict:
    from integrations.base_crm.client import get_opportunity
    return get_opportunity(opportunity_id)


@tool(
    name="crm_create_opportunity",
    description="Create a new deal/opportunity in the CRM.",
    parameters={
        "name": "string - deal name",
        "stage": "string - pipeline stage (MEETING, PROPOSAL, CUSTOMER, etc.)",
        "amount": "float - deal value in dollars (optional)",
        "company_id": "string - UUID of the company (optional)",
        "contact_id": "string - UUID of the point of contact (optional)",
    },
)
def crm_create_opportunity(name: str, stage: str = "MEETING", amount: float = 0,
                            company_id: str = "", contact_id: str = "") -> dict:
    from integrations.base_crm.client import create_opportunity
    return create_opportunity(name, stage, amount, company_id, contact_id)


@tool(
    name="crm_update_deal_stage",
    description="Move a deal/opportunity to a new pipeline stage.",
    parameters={
        "opportunity_id": "string - UUID of the opportunity",
        "stage": "string - new stage (MEETING, PROPOSAL, CUSTOMER, etc.)",
    },
)
def crm_update_deal_stage(opportunity_id: str, stage: str) -> dict:
    from integrations.base_crm.client import update_opportunity_stage
    return update_opportunity_stage(opportunity_id, stage)


@tool(
    name="crm_delete_opportunity",
    description="Delete a deal/opportunity from the CRM.",
    parameters={"opportunity_id": "string - UUID of the opportunity to delete"},
)
def crm_delete_opportunity(opportunity_id: str) -> dict:
    from integrations.base_crm.client import delete_opportunity
    return delete_opportunity(opportunity_id)


@tool(
    name="crm_pipeline_summary",
    description="Get a summary of the deal pipeline: deal count and total dollar value per stage.",
    parameters={},
)
def crm_pipeline_summary() -> dict:
    from integrations.base_crm.client import pipeline_summary
    return pipeline_summary()


@tool(
    name="crm_list_tasks",
    description="List tasks in the CRM.",
    parameters={"limit": "int (default 20)"},
)
def crm_list_tasks(limit: int = 20) -> list[dict]:
    from integrations.base_crm.client import list_tasks
    return list_tasks(limit=limit)


@tool(
    name="crm_create_task",
    description="Create a new task in the CRM.",
    parameters={
        "title": "string - task title",
        "status": "string - TODO or DONE (default TODO)",
        "due_date": "string - ISO datetime (optional)",
    },
)
def crm_create_task(title: str, status: str = "TODO", due_date: str = "") -> dict:
    from integrations.base_crm.client import create_task
    return create_task(title, status, due_date)


@tool(
    name="crm_update_task",
    description="Update a CRM task. Change its title, status (TODO/DONE), or due date.",
    parameters={
        "task_id": "string - UUID of the task",
        "title": "string (optional)",
        "status": "string - TODO or DONE (optional)",
        "due_date": "string - ISO datetime (optional)",
    },
)
def crm_update_task(task_id: str, title: str = "", status: str = "", due_date: str = "") -> dict:
    from integrations.base_crm.client import update_task
    return update_task(task_id, title, status, due_date)


@tool(
    name="crm_delete_task",
    description="Delete a task from the CRM.",
    parameters={"task_id": "string - UUID of the task to delete"},
)
def crm_delete_task(task_id: str) -> dict:
    from integrations.base_crm.client import delete_task
    return delete_task(task_id)


@tool(
    name="crm_add_note_to_person",
    description="Add a note linked to a specific person/contact in the CRM.",
    parameters={
        "title": "string - note title",
        "person_id": "string - UUID of the person",
        "body": "string - note content (optional)",
    },
)
def crm_add_note_to_person(title: str, person_id: str, body: str = "") -> dict:
    from integrations.base_crm.client import create_note_for_person
    return create_note_for_person(title, person_id, body)


@tool(
    name="crm_add_note_to_company",
    description="Add a note linked to a specific company in the CRM.",
    parameters={
        "title": "string - note title",
        "company_id": "string - UUID of the company",
        "body": "string - note content (optional)",
    },
)
def crm_add_note_to_company(title: str, company_id: str, body: str = "") -> dict:
    from integrations.base_crm.client import create_note_for_company
    return create_note_for_company(title, company_id, body)


@tool(
    name="crm_add_note_to_deal",
    description="Add a note linked to a specific deal/opportunity in the CRM.",
    parameters={
        "title": "string - note title",
        "opportunity_id": "string - UUID of the opportunity",
        "body": "string - note content (optional)",
    },
)
def crm_add_note_to_deal(title: str, opportunity_id: str, body: str = "") -> dict:
    from integrations.base_crm.client import create_note_for_opportunity
    return create_note_for_opportunity(title, opportunity_id, body)


@tool(
    name="crm_add_task_to_person",
    description="Create a task linked to a specific person/contact in the CRM.",
    parameters={
        "title": "string - task title",
        "person_id": "string - UUID of the person",
        "status": "string - TODO or DONE (default TODO)",
        "due_date": "string - ISO datetime (optional)",
    },
)
def crm_add_task_to_person(title: str, person_id: str, status: str = "TODO", due_date: str = "") -> dict:
    from integrations.base_crm.client import create_task_for_person
    return create_task_for_person(title, person_id, status, due_date)


@tool(
    name="crm_add_task_to_company",
    description="Create a task linked to a specific company in the CRM.",
    parameters={
        "title": "string - task title",
        "company_id": "string - UUID of the company",
        "status": "string - TODO or DONE (default TODO)",
        "due_date": "string - ISO datetime (optional)",
    },
)
def crm_add_task_to_company(title: str, company_id: str, status: str = "TODO", due_date: str = "") -> dict:
    from integrations.base_crm.client import create_task_for_company
    return create_task_for_company(title, company_id, status, due_date)


@tool(
    name="crm_add_task_to_deal",
    description="Create a task linked to a specific deal/opportunity in the CRM.",
    parameters={
        "title": "string - task title",
        "opportunity_id": "string - UUID of the opportunity",
        "status": "string - TODO or DONE (default TODO)",
        "due_date": "string - ISO datetime (optional)",
    },
)
def crm_add_task_to_deal(title: str, opportunity_id: str, status: str = "TODO", due_date: str = "") -> dict:
    from integrations.base_crm.client import create_task_for_opportunity
    return create_task_for_opportunity(title, opportunity_id, status, due_date)


# ==================== MEMORY TOOLS ====================

@tool(
    name="remember",
    description="Store a piece of information in long-term memory for later recall.",
    parameters={"text": "string - the information to remember", "category": "string - general/business/personal/financial"},
)
def remember(text: str, category: str = "general") -> dict:
    from core.memory.store import store_memory
    doc_id = store_memory(text, category)
    return {"stored": True, "id": doc_id}


@tool(
    name="recall",
    description="Search memory for relevant information based on a query.",
    parameters={"query": "string - what to search for", "category": "string (optional)"},
)
def recall_memory(query: str, category: str = "general") -> list[dict]:
    from core.memory.store import recall
    return recall(query, category)


# ==================== DOCUMENT TOOLS ====================

@tool(
    name="analyze_document",
    description="Analyze an uploaded document (PDF, Word, Excel, CSV, TXT, etc). Returns the extracted text content.",
    parameters={"file_path": "string - path to the uploaded document"},
)
def analyze_document(file_path: str) -> dict:
    from core.tools.files import parse_document
    return parse_document(file_path)


@tool(
    name="create_document",
    description="Create a document file that the user can download. Supports txt, md, csv, pdf, docx, xlsx, json formats.",
    parameters={
        "content": "string - the content to put in the document",
        "filename": "string - base name for the file (no extension)",
        "format": "string - output format: txt, md, csv, pdf, docx, xlsx, or json",
    },
)
def create_document_tool(content: str, filename: str, format: str = "txt") -> dict:
    from core.tools.files import create_document
    result = create_document(content, filename, format)
    if result["error"]:
        return {"success": False, "error": result["error"]}
    return {
        "success": True,
        "filename": result["filename"],
        "download_url": f"/download/{result['filename']}",
        "message": f"Document created: {result['filename']}",
    }


# ==================== IMAGE GENERATION ====================

@tool(
    name="generate_image",
    description="Generate an image from a text description using AI (SDXL Turbo). Creates high-quality images in seconds.",
    parameters={
        "prompt": "string - detailed description of the image to generate",
        "width": "int - image width in pixels (default 1024, max 1536)",
        "height": "int - image height in pixels (default 1024, max 1536)",
    },
)
async def generate_image_tool(prompt: str, width: int = 1024, height: int = 1024) -> dict:
    from integrations.comfyui.client import generate_image

    # Clamp dimensions
    width = min(max(width, 512), 1536)
    height = min(max(height, 512), 1536)

    result = await generate_image(prompt, width, height)

    if not result["success"]:
        return {"success": False, "error": result["error"]}

    return {
        "success": True,
        "message": f"Image generated successfully",
        "filename": result["filename"],
        "download_url": result["download_url"],
        "base64": result["base64"],
    }


# ==================== KNOWLEDGE BASE (LightRAG) ====================

@tool(
    name="search_knowledge",
    description="Search the knowledge base for information from uploaded documents, contracts, emails, and notes. Use this when the user asks about something that might be in their documents.",
    parameters={
        "query": "string - the question or search query",
        "top_k": "int - number of results to return (default 5)",
    },
)
async def search_knowledge(query: str, top_k: int = 5) -> dict:
    from integrations.lightrag.client import query_context
    result = await query_context(query, top_k=top_k)
    if not result["success"]:
        return {"success": False, "error": result["error"]}
    return {"success": True, "context": result["result"]}


@tool(
    name="ask_knowledge",
    description="Ask a question and get an answer from the knowledge base with full LLM reasoning. Use for complex questions requiring synthesis across multiple documents.",
    parameters={
        "question": "string - the question to answer",
        "mode": "string - search mode: 'hybrid' (default), 'local', 'global', or 'naive'",
    },
)
async def ask_knowledge(question: str, mode: str = "hybrid") -> dict:
    from integrations.lightrag.client import query
    result = await query(question, mode=mode, only_need_context=False)
    if not result["success"]:
        return {"success": False, "error": result["error"]}
    return {"success": True, "answer": result["result"]}


@tool(
    name="store_to_knowledge",
    description="Store text content in the knowledge base for future retrieval. Use for important information, notes, or summaries that should be remembered.",
    parameters={
        "text": "string - the content to store",
        "description": "string - brief description of what this content is about",
    },
)
async def store_to_knowledge(text: str, description: str = "") -> dict:
    from integrations.lightrag.client import upload_text
    result = await upload_text(text, description)
    if not result["success"]:
        return {"success": False, "error": result["error"]}
    return {"success": True, "message": "Content stored in knowledge base"}


@tool(
    name="list_knowledge_documents",
    description="List documents stored in the knowledge base.",
    parameters={"limit": "int - max documents to return (default 20)"},
)
async def list_knowledge_documents(limit: int = 20) -> dict:
    from integrations.lightrag.client import list_documents
    result = await list_documents(limit=limit)
    if not result["success"]:
        return {"success": False, "error": result["error"]}
    return {"success": True, "documents": result["documents"]}


# ==================== GOOGLE DRIVE ====================

@tool(
    name="drive_list_files",
    description="List files in Google Drive or a specific folder.",
    parameters={"folder_id": "str (optional)", "query": "str - search term (optional)", "file_type": "str - folder/document/spreadsheet/presentation/pdf/image (optional)"},
)
def drive_list_files(folder_id: str = None, query: str = None, file_type: str = None) -> list[dict]:
    from integrations.google_drive.client import list_files
    return list_files(folder_id=folder_id, query=query, file_type=file_type)


@tool(
    name="drive_search",
    description="Search for files in Google Drive by name or content.",
    parameters={"query": "str - search query"},
)
def drive_search(query: str) -> list[dict]:
    from integrations.google_drive.client import search_files
    return search_files(query)


@tool(
    name="drive_create_folder",
    description="Create a folder in Google Drive.",
    parameters={"name": "str - folder name", "parent_id": "str - parent folder ID (optional)"},
)
def drive_create_folder(name: str, parent_id: str = None) -> dict:
    from integrations.google_drive.client import create_folder
    return create_folder(name, parent_id)


@tool(
    name="drive_upload",
    description="Upload a file to Google Drive.",
    parameters={"local_path": "str - path to local file", "name": "str - name in Drive (optional)", "folder_id": "str - destination folder (optional)"},
)
def drive_upload(local_path: str, name: str = None, folder_id: str = None) -> dict:
    from integrations.google_drive.client import upload_file
    return upload_file(local_path, name, folder_id)


@tool(
    name="drive_download",
    description="Download a file from Google Drive.",
    parameters={"file_id": "str - Drive file ID", "local_path": "str - where to save locally"},
)
def drive_download(file_id: str, local_path: str) -> dict:
    from integrations.google_drive.client import download_file
    return download_file(file_id, local_path)


@tool(
    name="drive_share",
    description="Share a file with someone.",
    parameters={"file_id": "str - file ID", "email": "str - email to share with", "role": "str - reader/writer/commenter (default reader)"},
)
def drive_share(file_id: str, email: str, role: str = "reader") -> dict:
    from integrations.google_drive.client import share_file
    return share_file(file_id, email, role)


@tool(
    name="drive_delete",
    description="Move a file to trash.",
    parameters={"file_id": "str - file ID to delete"},
)
def drive_delete(file_id: str) -> dict:
    from integrations.google_drive.client import delete_file
    return delete_file(file_id)


# ==================== GOOGLE DOCS ====================

@tool(
    name="docs_create",
    description="Create a new Google Doc.",
    parameters={"title": "str - document title", "content": "str - initial text (optional)", "folder_id": "str - folder to create in (optional)"},
)
def docs_create(title: str, content: str = None, folder_id: str = None) -> dict:
    from integrations.google_docs.client import create_document
    return create_document(title, content, folder_id)


@tool(
    name="docs_read",
    description="Read the text content of a Google Doc.",
    parameters={"document_id": "str - the document ID"},
)
def docs_read(document_id: str) -> str:
    from integrations.google_docs.client import read_document
    return read_document(document_id)


@tool(
    name="docs_append",
    description="Append text to the end of a Google Doc.",
    parameters={"document_id": "str - the document ID", "text": "str - text to append"},
)
def docs_append(document_id: str, text: str) -> dict:
    from integrations.google_docs.client import append_text
    return append_text(document_id, text)


@tool(
    name="docs_replace",
    description="Find and replace text in a Google Doc.",
    parameters={"document_id": "str - the document ID", "find": "str - text to find", "replace": "str - replacement text"},
)
def docs_replace(document_id: str, find: str, replace: str) -> dict:
    from integrations.google_docs.client import replace_text
    return replace_text(document_id, find, replace)


@tool(
    name="docs_list",
    description="List Google Docs.",
    parameters={"max_results": "int (default 20)"},
)
def docs_list(max_results: int = 20) -> list[dict]:
    from integrations.google_docs.client import list_documents
    return list_documents(max_results)


# ==================== GOOGLE SHEETS ====================

@tool(
    name="sheets_create",
    description="Create a new Google Spreadsheet.",
    parameters={"title": "str - spreadsheet title", "sheet_names": "list of str - sheet names (optional)", "folder_id": "str - folder (optional)"},
)
def sheets_create(title: str, sheet_names: list = None, folder_id: str = None) -> dict:
    from integrations.google_sheets.client import create_spreadsheet
    return create_spreadsheet(title, sheet_names, folder_id)


@tool(
    name="sheets_read",
    description="Read data from a Google Sheet range.",
    parameters={"spreadsheet_id": "str", "range_notation": "str - A1 notation like 'Sheet1!A1:D10'"},
)
def sheets_read(spreadsheet_id: str, range_notation: str) -> list[list]:
    from integrations.google_sheets.client import read_range
    return read_range(spreadsheet_id, range_notation)


@tool(
    name="sheets_write",
    description="Write data to a Google Sheet range.",
    parameters={"spreadsheet_id": "str", "range_notation": "str - A1 notation", "values": "2D list of values"},
)
def sheets_write(spreadsheet_id: str, range_notation: str, values: list[list]) -> dict:
    from integrations.google_sheets.client import write_range
    return write_range(spreadsheet_id, range_notation, values)


@tool(
    name="sheets_append_row",
    description="Append a row to a Google Sheet.",
    parameters={"spreadsheet_id": "str", "values": "list of values for the row", "sheet_name": "str (default Sheet1)"},
)
def sheets_append_row(spreadsheet_id: str, values: list, sheet_name: str = "Sheet1") -> dict:
    from integrations.google_sheets.client import append_row
    return append_row(spreadsheet_id, values, sheet_name)


@tool(
    name="sheets_list",
    description="List Google Spreadsheets.",
    parameters={"max_results": "int (default 20)"},
)
def sheets_list(max_results: int = 20) -> list[dict]:
    from integrations.google_sheets.client import list_spreadsheets
    return list_spreadsheets(max_results)


@tool(
    name="sheets_get",
    description="Get spreadsheet metadata including sheet names.",
    parameters={"spreadsheet_id": "str"},
)
def sheets_get(spreadsheet_id: str) -> dict:
    from integrations.google_sheets.client import get_spreadsheet
    return get_spreadsheet(spreadsheet_id)


# ==================== GOOGLE SLIDES ====================

@tool(
    name="slides_create",
    description="Create a new Google Slides presentation.",
    parameters={"title": "str - presentation title", "folder_id": "str - folder (optional)"},
)
def slides_create(title: str, folder_id: str = None) -> dict:
    from integrations.google_slides.client import create_presentation
    return create_presentation(title, folder_id)


@tool(
    name="slides_get",
    description="Get presentation metadata and slide info.",
    parameters={"presentation_id": "str"},
)
def slides_get(presentation_id: str) -> dict:
    from integrations.google_slides.client import get_presentation
    return get_presentation(presentation_id)


@tool(
    name="slides_add_slide",
    description="Add a slide to a presentation.",
    parameters={"presentation_id": "str", "layout": "str - BLANK/TITLE/TITLE_AND_BODY (default BLANK)"},
)
def slides_add_slide(presentation_id: str, layout: str = "BLANK") -> dict:
    from integrations.google_slides.client import add_slide
    return add_slide(presentation_id, layout)


@tool(
    name="slides_add_text",
    description="Add text to a slide.",
    parameters={"presentation_id": "str", "slide_id": "str", "text": "str", "x": "float (default 100)", "y": "float (default 100)"},
)
def slides_add_text(presentation_id: str, slide_id: str, text: str, x: float = 100, y: float = 100) -> dict:
    from integrations.google_slides.client import add_text_to_slide
    return add_text_to_slide(presentation_id, slide_id, text, x, y)


@tool(
    name="slides_list",
    description="List Google Slides presentations.",
    parameters={"max_results": "int (default 20)"},
)
def slides_list(max_results: int = 20) -> list[dict]:
    from integrations.google_slides.client import list_presentations
    return list_presentations(max_results)


# ==================== N8N WORKFLOW AUTOMATION ====================

@tool(
    name="n8n_list_workflows",
    description="List all automation workflows in n8n.",
    parameters={"limit": "int (default 50)", "active_only": "bool - only show active workflows (default false)"},
)
def n8n_list_workflows(limit: int = 50, active_only: bool = False) -> list[dict]:
    from integrations.n8n.client import list_workflows
    return list_workflows(limit, active_only)


@tool(
    name="n8n_get_workflow",
    description="Get details of a specific workflow including its nodes and structure.",
    parameters={"workflow_id": "string - the workflow ID"},
)
def n8n_get_workflow(workflow_id: str) -> dict:
    from integrations.n8n.client import get_workflow_summary
    return get_workflow_summary(workflow_id)


@tool(
    name="n8n_create_workflow",
    description="Create a new automation workflow from a description. Describe what you want the workflow to do and it will generate the appropriate structure.",
    parameters={
        "name": "string - workflow name",
        "description": "string - describe what the workflow should do (e.g., 'send a Slack message every day at 9am')",
    },
)
def n8n_create_workflow_from_desc(name: str, description: str) -> dict:
    from integrations.n8n.client import create_workflow_from_description, create_workflow
    spec = create_workflow_from_description(name, description)
    result = create_workflow(spec["name"], spec["nodes"], spec["connections"])
    return {
        "success": True,
        "workflow": result,
        "message": f"Workflow '{name}' created. Use n8n_activate_workflow to enable it.",
        "nodes_created": len(spec["nodes"]),
    }


@tool(
    name="n8n_create_webhook_slack_workflow",
    description="Create a workflow that listens for webhooks and sends notifications to Slack.",
    parameters={
        "name": "string - workflow name",
        "webhook_path": "string - URL path for the webhook (e.g., 'my-webhook')",
        "slack_channel": "string - Slack channel (e.g., '#general')",
        "message_template": "string - message to send (can include {{$json.field}} placeholders)",
    },
)
def n8n_create_webhook_slack(name: str, webhook_path: str, slack_channel: str, message_template: str) -> dict:
    from integrations.n8n.client import create_webhook_to_slack_workflow
    result = create_webhook_to_slack_workflow(name, webhook_path, slack_channel, message_template)
    return {"success": True, "workflow": result}


@tool(
    name="n8n_create_scheduled_workflow",
    description="Create a workflow that runs on a schedule and makes an HTTP request.",
    parameters={
        "name": "string - workflow name",
        "cron": "string - cron expression (e.g., '0 9 * * *' for 9 AM daily)",
        "url": "string - URL to call",
        "method": "string - HTTP method (GET, POST, etc.)",
    },
)
def n8n_create_scheduled(name: str, cron: str, url: str, method: str = "GET") -> dict:
    from integrations.n8n.client import create_scheduled_http_workflow
    result = create_scheduled_http_workflow(name, cron, url, method)
    return {"success": True, "workflow": result}


@tool(
    name="n8n_activate_workflow",
    description="Activate a workflow so it starts running.",
    parameters={"workflow_id": "string - the workflow ID to activate"},
)
def n8n_activate_workflow(workflow_id: str) -> dict:
    from integrations.n8n.client import activate_workflow
    result = activate_workflow(workflow_id)
    return {"success": True, "workflow": result, "message": "Workflow activated"}


@tool(
    name="n8n_deactivate_workflow",
    description="Deactivate a workflow to stop it from running.",
    parameters={"workflow_id": "string - the workflow ID to deactivate"},
)
def n8n_deactivate_workflow(workflow_id: str) -> dict:
    from integrations.n8n.client import deactivate_workflow
    result = deactivate_workflow(workflow_id)
    return {"success": True, "workflow": result, "message": "Workflow deactivated"}


@tool(
    name="n8n_delete_workflow",
    description="Delete a workflow permanently.",
    parameters={"workflow_id": "string - the workflow ID to delete"},
)
def n8n_delete_workflow(workflow_id: str) -> dict:
    from integrations.n8n.client import delete_workflow
    return delete_workflow(workflow_id)


@tool(
    name="n8n_execute_workflow",
    description="Execute/run a workflow manually, optionally with input data.",
    parameters={
        "workflow_id": "string - the workflow ID to execute",
        "data": "dict - optional input data to pass to the workflow",
    },
)
def n8n_execute_workflow(workflow_id: str, data: dict = None) -> dict:
    from integrations.n8n.client import execute_workflow
    return execute_workflow(workflow_id, data)


@tool(
    name="n8n_get_executions",
    description="Get execution history for workflows.",
    parameters={
        "workflow_id": "string - optional workflow ID to filter by",
        "limit": "int - max executions to return (default 20)",
    },
)
def n8n_get_executions(workflow_id: str = "", limit: int = 20) -> list[dict]:
    from integrations.n8n.client import get_executions
    return get_executions(workflow_id if workflow_id else None, limit)


# ==================== NEXTCLOUD ====================

@tool(
    name="nextcloud_list_files",
    description="List files and folders in Nextcloud at a given path.",
    parameters={"path": "string - folder path (default '/')"},
)
def nextcloud_list_files(path: str = "/") -> list[dict]:
    from integrations.nextcloud.client import list_files
    return list_files(path)


@tool(
    name="nextcloud_search_files",
    description="Search for files by name in Nextcloud.",
    parameters={"query": "string - search query", "path": "string - folder to search in (default '/')"},
)
def nextcloud_search_files(query: str, path: str = "/") -> list[dict]:
    from integrations.nextcloud.client import search_files
    return search_files(query, path)


@tool(
    name="nextcloud_read_file",
    description="Read the contents of a text file from Nextcloud.",
    parameters={"path": "string - file path"},
)
def nextcloud_read_file(path: str) -> dict:
    from integrations.nextcloud.client import download_file_text
    try:
        content = download_file_text(path)
        return {"success": True, "content": content}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool(
    name="nextcloud_upload_file",
    description="Upload or create a file in Nextcloud.",
    parameters={"path": "string - destination path", "content": "string - file content"},
)
def nextcloud_upload_file(path: str, content: str) -> dict:
    from integrations.nextcloud.client import upload_file
    return upload_file(path, content)


@tool(
    name="nextcloud_create_folder",
    description="Create a folder in Nextcloud.",
    parameters={"path": "string - folder path to create"},
)
def nextcloud_create_folder(path: str) -> dict:
    from integrations.nextcloud.client import create_folder
    return create_folder(path)


@tool(
    name="nextcloud_delete_file",
    description="Delete a file or folder from Nextcloud.",
    parameters={"path": "string - path to delete"},
)
def nextcloud_delete_file(path: str) -> dict:
    from integrations.nextcloud.client import delete_file
    return delete_file(path)


@tool(
    name="nextcloud_move_file",
    description="Move or rename a file/folder in Nextcloud.",
    parameters={"source": "string - current path", "destination": "string - new path"},
)
def nextcloud_move_file(source: str, destination: str) -> dict:
    from integrations.nextcloud.client import move_file
    return move_file(source, destination)


@tool(
    name="nextcloud_storage_info",
    description="Get Nextcloud storage quota usage.",
    parameters={},
)
def nextcloud_storage_info() -> dict:
    from integrations.nextcloud.client import get_storage_info
    return get_storage_info()


# Nextcloud Notes
@tool(
    name="nextcloud_list_notes",
    description="List all notes in Nextcloud Notes app.",
    parameters={},
)
def nextcloud_list_notes() -> list[dict]:
    from integrations.nextcloud.client import list_notes
    return list_notes()


@tool(
    name="nextcloud_get_note",
    description="Get a specific note from Nextcloud.",
    parameters={"note_id": "int - note ID"},
)
def nextcloud_get_note(note_id: int) -> dict:
    from integrations.nextcloud.client import get_note
    return get_note(note_id)


@tool(
    name="nextcloud_create_note",
    description="Create a new note in Nextcloud.",
    parameters={"title": "string", "content": "string (optional)", "category": "string (optional)"},
)
def nextcloud_create_note(title: str, content: str = "", category: str = "") -> dict:
    from integrations.nextcloud.client import create_note
    return create_note(title, content, category)


@tool(
    name="nextcloud_update_note",
    description="Update a note in Nextcloud.",
    parameters={"note_id": "int", "title": "string (optional)", "content": "string (optional)"},
)
def nextcloud_update_note(note_id: int, title: str = None, content: str = None) -> dict:
    from integrations.nextcloud.client import update_note
    return update_note(note_id, title, content)


@tool(
    name="nextcloud_delete_note",
    description="Delete a note from Nextcloud.",
    parameters={"note_id": "int"},
)
def nextcloud_delete_note(note_id: int) -> dict:
    from integrations.nextcloud.client import delete_note
    return delete_note(note_id)


# Nextcloud Talk
@tool(
    name="nextcloud_list_conversations",
    description="List all Nextcloud Talk conversations/chats.",
    parameters={},
)
def nextcloud_list_conversations() -> list[dict]:
    from integrations.nextcloud.client import list_conversations
    return list_conversations()


@tool(
    name="nextcloud_get_messages",
    description="Get messages from a Nextcloud Talk conversation.",
    parameters={"token": "string - conversation token", "limit": "int (default 50)"},
)
def nextcloud_get_messages(token: str, limit: int = 50) -> list[dict]:
    from integrations.nextcloud.client import get_messages
    return get_messages(token, limit)


@tool(
    name="nextcloud_send_message",
    description="Send a message to a Nextcloud Talk conversation.",
    parameters={"token": "string - conversation token", "message": "string - message to send"},
)
def nextcloud_send_message(token: str, message: str) -> dict:
    from integrations.nextcloud.client import send_message
    return send_message(token, message)


@tool(
    name="nextcloud_create_conversation",
    description="Create a new Nextcloud Talk group conversation.",
    parameters={"name": "string - conversation name", "invite_users": "list of user IDs to invite (optional)"},
)
def nextcloud_create_conversation(name: str, invite_users: list[str] = None) -> dict:
    from integrations.nextcloud.client import create_conversation
    return create_conversation(name, 2, invite_users)


@tool(
    name="nextcloud_add_participant",
    description="Add a user to a Nextcloud Talk conversation.",
    parameters={"token": "string - conversation token", "user_id": "string - user to add"},
)
def nextcloud_add_participant(token: str, user_id: str) -> dict:
    from integrations.nextcloud.client import add_participant
    return add_participant(token, user_id)


# Nextcloud User Management
@tool(
    name="nextcloud_list_users",
    description="List users in Nextcloud (requires admin).",
    parameters={"search": "string - search query (optional)", "limit": "int (default 50)"},
)
def nextcloud_list_users(search: str = "", limit: int = 50) -> list[dict]:
    from integrations.nextcloud.client import list_users
    return list_users(search, limit)


@tool(
    name="nextcloud_get_user",
    description="Get details about a Nextcloud user.",
    parameters={"user_id": "string - username"},
)
def nextcloud_get_user(user_id: str) -> dict:
    from integrations.nextcloud.client import get_user
    return get_user(user_id)


@tool(
    name="nextcloud_create_user",
    description="Create a new Nextcloud user (requires admin).",
    parameters={
        "user_id": "string - username",
        "password": "string - initial password",
        "email": "string (optional)",
        "display_name": "string (optional)",
        "groups": "list of group names (optional)",
    },
)
def nextcloud_create_user(user_id: str, password: str, email: str = "",
                          display_name: str = "", groups: list[str] = None) -> dict:
    from integrations.nextcloud.client import create_user
    return create_user(user_id, password, email, display_name, groups)


@tool(
    name="nextcloud_delete_user",
    description="Delete a Nextcloud user (requires admin).",
    parameters={"user_id": "string - username to delete"},
)
def nextcloud_delete_user(user_id: str) -> dict:
    from integrations.nextcloud.client import delete_user
    return delete_user(user_id)


@tool(
    name="nextcloud_enable_user",
    description="Enable a disabled Nextcloud user.",
    parameters={"user_id": "string - username"},
)
def nextcloud_enable_user(user_id: str) -> dict:
    from integrations.nextcloud.client import enable_user
    return enable_user(user_id)


@tool(
    name="nextcloud_disable_user",
    description="Disable a Nextcloud user.",
    parameters={"user_id": "string - username"},
)
def nextcloud_disable_user(user_id: str) -> dict:
    from integrations.nextcloud.client import disable_user
    return disable_user(user_id)


@tool(
    name="nextcloud_list_groups",
    description="List all Nextcloud groups.",
    parameters={},
)
def nextcloud_list_groups() -> list[dict]:
    from integrations.nextcloud.client import list_groups
    return list_groups()


@tool(
    name="nextcloud_add_user_to_group",
    description="Add a user to a Nextcloud group.",
    parameters={"user_id": "string - username", "group_id": "string - group name"},
)
def nextcloud_add_user_to_group(user_id: str, group_id: str) -> dict:
    from integrations.nextcloud.client import add_user_to_group
    return add_user_to_group(user_id, group_id)


# Nextcloud Calendar
@tool(
    name="nextcloud_list_calendars",
    description="List Nextcloud calendars.",
    parameters={},
)
def nextcloud_list_calendars() -> list[dict]:
    from integrations.nextcloud.client import list_calendars
    return list_calendars()


@tool(
    name="nextcloud_get_calendar_events",
    description="Get events from a Nextcloud calendar.",
    parameters={"calendar_id": "string - calendar ID", "days_ahead": "int - days to look ahead (default 30)"},
)
def nextcloud_get_calendar_events(calendar_id: str, days_ahead: int = 30) -> list[dict]:
    from integrations.nextcloud.client import get_calendar_events
    return get_calendar_events(calendar_id, days_ahead)


# Nextcloud Tasks
@tool(
    name="nextcloud_get_tasks",
    description="Get tasks from a Nextcloud task list.",
    parameters={"list_id": "string - task list/calendar ID"},
)
def nextcloud_get_tasks(list_id: str) -> list[dict]:
    from integrations.nextcloud.client import get_tasks
    return get_tasks(list_id)


# ==================== STRIPE ====================

@tool(
    name="stripe_get_balance",
    description="Get Stripe account balance (available and pending funds).",
    parameters={},
)
def stripe_get_balance() -> dict:
    from integrations.stripe.client import get_balance
    return get_balance()


@tool(
    name="stripe_list_payouts",
    description="List recent Stripe payouts to your bank account.",
    parameters={"limit": "int (default 20)"},
)
def stripe_list_payouts(limit: int = 20) -> list[dict]:
    from integrations.stripe.client import list_payouts
    return list_payouts(limit)


@tool(
    name="stripe_revenue_summary",
    description="Get a summary of recent revenue, active subscriptions, and estimated MRR.",
    parameters={},
)
def stripe_revenue_summary() -> dict:
    from integrations.stripe.client import get_revenue_summary
    return get_revenue_summary()


# Payments
@tool(
    name="stripe_list_payments",
    description="List recent Stripe payments/charges.",
    parameters={"limit": "int (default 20)", "customer": "string - customer ID (optional)"},
)
def stripe_list_payments(limit: int = 20, customer: str = None) -> list[dict]:
    from integrations.stripe.client import list_payments
    return list_payments(limit, customer)


@tool(
    name="stripe_get_payment",
    description="Get details of a specific Stripe payment/charge.",
    parameters={"charge_id": "string - charge ID (ch_...)"},
)
def stripe_get_payment(charge_id: str) -> dict:
    from integrations.stripe.client import get_payment
    return get_payment(charge_id)


@tool(
    name="stripe_search_payments",
    description="Search Stripe payments. Query examples: 'amount>1000', 'status:succeeded', 'customer:cus_xxx'.",
    parameters={"query": "string - Stripe search query", "limit": "int (default 20)"},
)
def stripe_search_payments(query: str, limit: int = 20) -> list[dict]:
    from integrations.stripe.client import search_payments
    return search_payments(query, limit)


@tool(
    name="stripe_refund_payment",
    description="Issue a refund for a Stripe payment. Omit amount for full refund.",
    parameters={
        "charge_id": "string - charge ID to refund",
        "amount": "float - amount in dollars (optional, omit for full refund)",
        "reason": "string - 'duplicate', 'fraudulent', or 'requested_by_customer' (optional)",
    },
)
def stripe_refund_payment(charge_id: str, amount: float = None, reason: str = None) -> dict:
    from integrations.stripe.client import refund_payment
    return refund_payment(charge_id, amount, reason)


@tool(
    name="stripe_list_refunds",
    description="List Stripe refunds.",
    parameters={"limit": "int (default 20)", "charge": "string - charge ID to filter by (optional)"},
)
def stripe_list_refunds(limit: int = 20, charge: str = None) -> list[dict]:
    from integrations.stripe.client import list_refunds
    return list_refunds(limit, charge)


# Customers
@tool(
    name="stripe_list_customers",
    description="List Stripe customers.",
    parameters={"limit": "int (default 20)", "email": "string - filter by email (optional)"},
)
def stripe_list_customers(limit: int = 20, email: str = None) -> list[dict]:
    from integrations.stripe.client import list_customers
    return list_customers(limit, email)


@tool(
    name="stripe_search_customers",
    description="Search Stripe customers. Query examples: 'email:john@example.com', 'name:John'.",
    parameters={"query": "string - Stripe search query", "limit": "int (default 20)"},
)
def stripe_search_customers(query: str, limit: int = 20) -> list[dict]:
    from integrations.stripe.client import search_customers
    return search_customers(query, limit)


@tool(
    name="stripe_get_customer",
    description="Get details of a specific Stripe customer.",
    parameters={"customer_id": "string - customer ID (cus_...)"},
)
def stripe_get_customer(customer_id: str) -> dict:
    from integrations.stripe.client import get_customer
    return get_customer(customer_id)


@tool(
    name="stripe_create_customer",
    description="Create a new Stripe customer.",
    parameters={
        "email": "string - customer email",
        "name": "string (optional)",
        "phone": "string (optional)",
        "description": "string (optional)",
    },
)
def stripe_create_customer(email: str, name: str = None, phone: str = None, description: str = None) -> dict:
    from integrations.stripe.client import create_customer
    return create_customer(email, name, phone, description)


@tool(
    name="stripe_update_customer",
    description="Update a Stripe customer.",
    parameters={
        "customer_id": "string - customer ID",
        "email": "string (optional)",
        "name": "string (optional)",
        "phone": "string (optional)",
    },
)
def stripe_update_customer(customer_id: str, email: str = None, name: str = None, phone: str = None) -> dict:
    from integrations.stripe.client import update_customer
    return update_customer(customer_id, email, name, phone)


@tool(
    name="stripe_delete_customer",
    description="Delete a Stripe customer.",
    parameters={"customer_id": "string - customer ID to delete"},
)
def stripe_delete_customer(customer_id: str) -> dict:
    from integrations.stripe.client import delete_customer
    return delete_customer(customer_id)


# Subscriptions
@tool(
    name="stripe_list_subscriptions",
    description="List Stripe subscriptions.",
    parameters={
        "limit": "int (default 20)",
        "customer": "string - customer ID (optional)",
        "status": "string - 'active', 'past_due', 'canceled', 'all' (optional)",
    },
)
def stripe_list_subscriptions(limit: int = 20, customer: str = None, status: str = None) -> list[dict]:
    from integrations.stripe.client import list_subscriptions
    return list_subscriptions(limit, customer, status)


@tool(
    name="stripe_get_subscription",
    description="Get details of a Stripe subscription.",
    parameters={"subscription_id": "string - subscription ID (sub_...)"},
)
def stripe_get_subscription(subscription_id: str) -> dict:
    from integrations.stripe.client import get_subscription
    return get_subscription(subscription_id)


@tool(
    name="stripe_create_subscription",
    description="Create a subscription for a customer.",
    parameters={
        "customer_id": "string - customer ID",
        "price_id": "string - price ID (price_...)",
        "quantity": "int (default 1)",
    },
)
def stripe_create_subscription(customer_id: str, price_id: str, quantity: int = 1) -> dict:
    from integrations.stripe.client import create_subscription
    return create_subscription(customer_id, price_id, quantity)


@tool(
    name="stripe_cancel_subscription",
    description="Cancel a Stripe subscription.",
    parameters={
        "subscription_id": "string - subscription ID",
        "at_period_end": "bool - if true, cancel at end of billing period (default true)",
    },
)
def stripe_cancel_subscription(subscription_id: str, at_period_end: bool = True) -> dict:
    from integrations.stripe.client import cancel_subscription
    return cancel_subscription(subscription_id, at_period_end)


@tool(
    name="stripe_resume_subscription",
    description="Resume a subscription that was scheduled for cancellation.",
    parameters={"subscription_id": "string - subscription ID"},
)
def stripe_resume_subscription(subscription_id: str) -> dict:
    from integrations.stripe.client import resume_subscription
    return resume_subscription(subscription_id)


# Products & Prices
@tool(
    name="stripe_list_products",
    description="List Stripe products.",
    parameters={"limit": "int (default 20)", "active": "bool - filter by active status (optional)"},
)
def stripe_list_products(limit: int = 20, active: bool = None) -> list[dict]:
    from integrations.stripe.client import list_products
    return list_products(limit, active)


@tool(
    name="stripe_get_product",
    description="Get details of a Stripe product.",
    parameters={"product_id": "string - product ID (prod_...)"},
)
def stripe_get_product(product_id: str) -> dict:
    from integrations.stripe.client import get_product
    return get_product(product_id)


@tool(
    name="stripe_create_product",
    description="Create a new Stripe product.",
    parameters={
        "name": "string - product name",
        "description": "string (optional)",
        "active": "bool (default true)",
    },
)
def stripe_create_product(name: str, description: str = None, active: bool = True) -> dict:
    from integrations.stripe.client import create_product
    return create_product(name, description, active)


@tool(
    name="stripe_list_prices",
    description="List Stripe prices.",
    parameters={
        "limit": "int (default 20)",
        "product": "string - product ID (optional)",
        "active": "bool (optional)",
    },
)
def stripe_list_prices(limit: int = 20, product: str = None, active: bool = None) -> list[dict]:
    from integrations.stripe.client import list_prices
    return list_prices(limit, product, active)


@tool(
    name="stripe_create_price",
    description="Create a price for a product.",
    parameters={
        "product_id": "string - product ID",
        "unit_amount": "float - price in dollars",
        "currency": "string (default 'usd')",
        "recurring_interval": "string - 'month' or 'year' for subscriptions (optional)",
    },
)
def stripe_create_price(product_id: str, unit_amount: float, currency: str = "usd",
                        recurring_interval: str = None) -> dict:
    from integrations.stripe.client import create_price
    return create_price(product_id, unit_amount, currency, recurring_interval)


# Invoices
@tool(
    name="stripe_list_invoices",
    description="List Stripe invoices.",
    parameters={
        "limit": "int (default 20)",
        "customer": "string - customer ID (optional)",
        "status": "string - 'draft', 'open', 'paid', 'void' (optional)",
    },
)
def stripe_list_invoices(limit: int = 20, customer: str = None, status: str = None) -> list[dict]:
    from integrations.stripe.client import list_invoices
    return list_invoices(limit, customer, status)


@tool(
    name="stripe_get_invoice",
    description="Get details of a Stripe invoice.",
    parameters={"invoice_id": "string - invoice ID (in_...)"},
)
def stripe_get_invoice(invoice_id: str) -> dict:
    from integrations.stripe.client import get_invoice
    return get_invoice(invoice_id)


@tool(
    name="stripe_create_invoice",
    description="Create a draft invoice for a customer.",
    parameters={
        "customer_id": "string - customer ID",
        "description": "string (optional)",
        "days_until_due": "int (default 30)",
    },
)
def stripe_create_invoice(customer_id: str, description: str = None, days_until_due: int = 30) -> dict:
    from integrations.stripe.client import create_invoice
    return create_invoice(customer_id, description, days_until_due)


@tool(
    name="stripe_add_invoice_item",
    description="Add a line item to a draft invoice.",
    parameters={
        "invoice_id": "string - invoice ID",
        "description": "string - item description",
        "amount": "float - amount in dollars",
        "quantity": "int (default 1)",
    },
)
def stripe_add_invoice_item(invoice_id: str, description: str, amount: float, quantity: int = 1) -> dict:
    from integrations.stripe.client import add_invoice_item
    return add_invoice_item(invoice_id, description, amount, quantity)


@tool(
    name="stripe_finalize_invoice",
    description="Finalize a draft invoice (locks it for payment).",
    parameters={"invoice_id": "string - invoice ID"},
)
def stripe_finalize_invoice(invoice_id: str) -> dict:
    from integrations.stripe.client import finalize_invoice
    return finalize_invoice(invoice_id)


@tool(
    name="stripe_send_invoice",
    description="Send an invoice to the customer via email.",
    parameters={"invoice_id": "string - invoice ID"},
)
def stripe_send_invoice(invoice_id: str) -> dict:
    from integrations.stripe.client import send_invoice
    return send_invoice(invoice_id)


@tool(
    name="stripe_void_invoice",
    description="Void an invoice.",
    parameters={"invoice_id": "string - invoice ID"},
)
def stripe_void_invoice(invoice_id: str) -> dict:
    from integrations.stripe.client import void_invoice
    return void_invoice(invoice_id)


# Payment Links
@tool(
    name="stripe_list_payment_links",
    description="List Stripe payment links.",
    parameters={"limit": "int (default 20)", "active": "bool (optional)"},
)
def stripe_list_payment_links(limit: int = 20, active: bool = None) -> list[dict]:
    from integrations.stripe.client import list_payment_links
    return list_payment_links(limit, active)


@tool(
    name="stripe_create_payment_link",
    description="Create a Stripe payment link for a price.",
    parameters={"price_id": "string - price ID", "quantity": "int (default 1)"},
)
def stripe_create_payment_link(price_id: str, quantity: int = 1) -> dict:
    from integrations.stripe.client import create_payment_link
    return create_payment_link(price_id, quantity)


@tool(
    name="stripe_deactivate_payment_link",
    description="Deactivate a Stripe payment link.",
    parameters={"payment_link_id": "string - payment link ID"},
)
def stripe_deactivate_payment_link(payment_link_id: str) -> dict:
    from integrations.stripe.client import deactivate_payment_link
    return deactivate_payment_link(payment_link_id)


# ==================== UI CONTROL ====================

@tool(
    name="toggle_auto_speak",
    description="Turn auto-speak on or off. Use this when user asks to enable/disable auto-speak, mute, unmute, or stop/start speaking.",
    parameters={"enabled": "bool - true to enable, false to disable"},
)
def toggle_auto_speak(enabled: bool) -> dict:
    action = "enable" if enabled else "disable"
    return {
        "ui_action": "set_auto_speak",
        "value": enabled,
        "message": f"Auto-speak {'enabled' if enabled else 'disabled'}, sir."
    }


@tool(
    name="toggle_hands_free",
    description="Turn hands-free mode on or off. Use this when user asks to enable/disable hands-free or start/stop listening.",
    parameters={"enabled": "bool - true to enable, false to disable"},
)
def toggle_hands_free(enabled: bool) -> dict:
    return {
        "ui_action": "set_hands_free",
        "value": enabled,
        "message": f"Hands-free mode {'enabled' if enabled else 'disabled'}, sir."
    }


# ==================== AZURACAST RADIO TOOLS ====================

@tool(
    name="radio_list_stations",
    description="List all available radio stations. Use this to see station names and IDs for other radio commands.",
    parameters={},
)
def radio_list_stations() -> list[dict]:
    from integrations.azuracast.client import list_stations
    return list_stations()


@tool(
    name="radio_now_playing",
    description="Get what's currently playing on a radio station, including listener count, current song artist/title, and playlist info.",
    parameters={"station": "string (optional) - station name, shortcode, or ID. Defaults to first station if not specified."},
)
def radio_now_playing(station: str = None) -> dict:
    from integrations.azuracast.client import get_now_playing, get_station_id
    station_id = get_station_id(station)
    return get_now_playing(station_id)


@tool(
    name="radio_song_history",
    description="Get recently played songs on a radio station.",
    parameters={
        "station": "string (optional) - station name, shortcode, or ID",
        "limit": "int (default 10) - number of recent songs to return",
    },
)
def radio_song_history(station: str = None, limit: int = 10) -> list[dict]:
    from integrations.azuracast.client import get_song_history, get_station_id
    station_id = get_station_id(station)
    return get_song_history(station_id, limit=limit)


@tool(
    name="radio_playlists",
    description="List all playlists on a radio station with their song counts and enabled status.",
    parameters={"station": "string (optional) - station name, shortcode, or ID"},
)
def radio_playlists(station: str = None) -> list[dict]:
    from integrations.azuracast.client import list_playlists, get_station_id
    station_id = get_station_id(station)
    return list_playlists(station_id)


@tool(
    name="radio_toggle_playlist",
    description="Enable or disable a radio playlist by its ID.",
    parameters={
        "playlist_id": "int - the playlist ID to toggle",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_toggle_playlist(playlist_id: int, station: str = None) -> dict:
    from integrations.azuracast.client import toggle_playlist, get_station_id
    station_id = get_station_id(station)
    return toggle_playlist(station_id, playlist_id)


@tool(
    name="radio_queue",
    description="Get the upcoming song queue on a radio station.",
    parameters={"station": "string (optional) - station name, shortcode, or ID"},
)
def radio_queue(station: str = None) -> list[dict]:
    from integrations.azuracast.client import get_queue, get_station_id
    station_id = get_station_id(station)
    return get_queue(station_id)


@tool(
    name="radio_listeners",
    description="Get current listener details and count for a radio station.",
    parameters={"station": "string (optional) - station name, shortcode, or ID"},
)
def radio_listeners(station: str = None) -> dict:
    from integrations.azuracast.client import get_listener_report, get_station_id
    station_id = get_station_id(station)
    return get_listener_report(station_id)


@tool(
    name="radio_restart",
    description="Restart a radio station's broadcasting services.",
    parameters={"station": "string (optional) - station name, shortcode, or ID"},
)
def radio_restart(station: str = None) -> dict:
    from integrations.azuracast.client import restart_station, get_station_id
    station_id = get_station_id(station)
    return restart_station(station_id)


@tool(
    name="radio_search_media",
    description="Search for songs in a radio station's media library by artist, title, or album.",
    parameters={
        "query": "string - search term for artist, title, or album",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_search_media(query: str, station: str = None) -> list[dict]:
    from integrations.azuracast.client import search_media, get_station_id
    station_id = get_station_id(station)
    return search_media(station_id, query)


# ==================== RADIO ADMIN: STATION MANAGEMENT ====================

@tool(
    name="radio_create_station",
    description="Create a new radio station.",
    parameters={
        "name": "string - the station name",
        "shortcode": "string (optional) - URL-friendly short name",
        "description": "string (optional) - station description",
    },
)
def radio_create_station(name: str, shortcode: str = None, description: str = "") -> dict:
    from integrations.azuracast.client import admin_create_station
    return admin_create_station(name, shortcode, description)


@tool(
    name="radio_update_station",
    description="Update station settings like name, description, or enabled status.",
    parameters={
        "station": "string - station name, shortcode, or ID",
        "name": "string (optional) - new name",
        "description": "string (optional) - new description",
        "is_enabled": "boolean (optional) - enable/disable station",
    },
)
def radio_update_station(station: str, name: str = None, description: str = None,
                         is_enabled: bool = None) -> dict:
    from integrations.azuracast.client import admin_update_station, get_station_id
    station_id = get_station_id(station)
    kwargs = {}
    if name is not None:
        kwargs["name"] = name
    if description is not None:
        kwargs["description"] = description
    if is_enabled is not None:
        kwargs["is_enabled"] = is_enabled
    return admin_update_station(station_id, **kwargs)


@tool(
    name="radio_delete_station",
    description="Permanently delete a radio station. Use with caution!",
    parameters={"station": "string - station name, shortcode, or ID to delete"},
)
def radio_delete_station(station: str) -> dict:
    from integrations.azuracast.client import admin_delete_station, get_station_id
    station_id = get_station_id(station)
    return admin_delete_station(station_id)


@tool(
    name="radio_clone_station",
    description="Clone an existing station with all its settings, playlists, and media.",
    parameters={
        "station": "string - source station to clone",
        "name": "string - name for the new station",
        "shortcode": "string (optional) - shortcode for new station",
    },
)
def radio_clone_station(station: str, name: str, shortcode: str = None) -> dict:
    from integrations.azuracast.client import admin_clone_station, get_station_id
    station_id = get_station_id(station)
    return admin_clone_station(station_id, name, shortcode)


# ==================== RADIO ADMIN: USER MANAGEMENT ====================

@tool(
    name="radio_list_users",
    description="List all AzuraCast user accounts.",
    parameters={},
)
def radio_list_users() -> list[dict]:
    from integrations.azuracast.client import admin_list_users
    return admin_list_users()


@tool(
    name="radio_create_user",
    description="Create a new AzuraCast user account.",
    parameters={
        "email": "string - user's email address (used for login)",
        "name": "string - display name",
        "password": "string - account password",
    },
)
def radio_create_user(email: str, name: str, password: str) -> dict:
    from integrations.azuracast.client import admin_create_user
    return admin_create_user(email, name, password)


@tool(
    name="radio_update_user",
    description="Update a user's details.",
    parameters={
        "user_id": "int - the user ID",
        "email": "string (optional) - new email",
        "name": "string (optional) - new name",
        "password": "string (optional) - new password",
        "is_enabled": "boolean (optional) - enable/disable account",
    },
)
def radio_update_user(user_id: int, email: str = None, name: str = None,
                      password: str = None, is_enabled: bool = None) -> dict:
    from integrations.azuracast.client import admin_update_user
    kwargs = {}
    if email is not None:
        kwargs["email"] = email
    if name is not None:
        kwargs["name"] = name
    if password is not None:
        kwargs["password"] = password
    if is_enabled is not None:
        kwargs["is_enabled"] = is_enabled
    return admin_update_user(user_id, **kwargs)


@tool(
    name="radio_delete_user",
    description="Delete an AzuraCast user account.",
    parameters={"user_id": "int - the user ID to delete"},
)
def radio_delete_user(user_id: int) -> dict:
    from integrations.azuracast.client import admin_delete_user
    return admin_delete_user(user_id)


@tool(
    name="radio_list_roles",
    description="List all roles/permission groups in AzuraCast.",
    parameters={},
)
def radio_list_roles() -> list[dict]:
    from integrations.azuracast.client import admin_list_roles
    return admin_list_roles()


# ==================== RADIO ADMIN: STORAGE ====================

@tool(
    name="radio_storage_locations",
    description="List all storage locations and their usage/quotas.",
    parameters={},
)
def radio_storage_locations() -> list[dict]:
    from integrations.azuracast.client import admin_list_storage_locations
    return admin_list_storage_locations()


@tool(
    name="radio_update_storage_quota",
    description="Update storage quota for a storage location.",
    parameters={
        "location_id": "int - storage location ID",
        "quota_gb": "float - quota in gigabytes (0 for unlimited)",
    },
)
def radio_update_storage_quota(location_id: int, quota_gb: float) -> dict:
    from integrations.azuracast.client import admin_update_storage_quota
    quota_bytes = int(quota_gb * 1024 * 1024 * 1024)
    return admin_update_storage_quota(location_id, quota_bytes)


@tool(
    name="radio_station_quota",
    description="Get storage quota usage for a specific station.",
    parameters={"station": "string (optional) - station name, shortcode, or ID"},
)
def radio_station_quota(station: str = None) -> dict:
    from integrations.azuracast.client import get_station_quota, get_station_id
    station_id = get_station_id(station)
    return get_station_quota(station_id)


# ==================== RADIO: DJ/STREAMER MANAGEMENT ====================

@tool(
    name="radio_list_djs",
    description="List all DJ/streamer accounts for a station.",
    parameters={"station": "string (optional) - station name, shortcode, or ID"},
)
def radio_list_djs(station: str = None) -> list[dict]:
    from integrations.azuracast.client import list_streamers, get_station_id
    station_id = get_station_id(station)
    return list_streamers(station_id)


@tool(
    name="radio_create_dj",
    description="Create a new DJ/streamer account for live broadcasting.",
    parameters={
        "username": "string - login username for the DJ",
        "password": "string - password for streaming",
        "display_name": "string (optional) - name shown when live",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_create_dj(username: str, password: str, display_name: str = None,
                    station: str = None) -> dict:
    from integrations.azuracast.client import create_streamer, get_station_id
    station_id = get_station_id(station)
    return create_streamer(station_id, username, password, display_name)


@tool(
    name="radio_update_dj",
    description="Update a DJ/streamer account.",
    parameters={
        "dj_id": "int - the streamer/DJ ID",
        "username": "string (optional) - new username",
        "password": "string (optional) - new password",
        "display_name": "string (optional) - new display name",
        "is_active": "boolean (optional) - enable/disable account",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_update_dj(dj_id: int, username: str = None, password: str = None,
                    display_name: str = None, is_active: bool = None,
                    station: str = None) -> dict:
    from integrations.azuracast.client import update_streamer, get_station_id
    station_id = get_station_id(station)
    kwargs = {}
    if username is not None:
        kwargs["username"] = username
    if password is not None:
        kwargs["password"] = password
    if display_name is not None:
        kwargs["display_name"] = display_name
    if is_active is not None:
        kwargs["is_active"] = is_active
    return update_streamer(station_id, dj_id, **kwargs)


@tool(
    name="radio_delete_dj",
    description="Delete a DJ/streamer account.",
    parameters={
        "dj_id": "int - the streamer/DJ ID to delete",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_delete_dj(dj_id: int, station: str = None) -> dict:
    from integrations.azuracast.client import delete_streamer, get_station_id
    station_id = get_station_id(station)
    return delete_streamer(station_id, dj_id)


# ==================== RADIO: MEDIA MANAGEMENT ====================

@tool(
    name="radio_upload_song",
    description="Upload a song file to the station's media library.",
    parameters={
        "file_path": "string - local path to the audio file",
        "folder": "string (optional) - destination folder in library",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_upload_song(file_path: str, folder: str = "", station: str = None) -> dict:
    from integrations.azuracast.client import upload_media, get_station_id
    station_id = get_station_id(station)
    return upload_media(station_id, file_path, folder)


@tool(
    name="radio_update_song",
    description="Update song metadata (artist, title, album, etc.).",
    parameters={
        "media_id": "int - the media file ID",
        "artist": "string (optional) - artist name",
        "title": "string (optional) - song title",
        "album": "string (optional) - album name",
        "genre": "string (optional) - genre",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_update_song(media_id: int, artist: str = None, title: str = None,
                      album: str = None, genre: str = None, station: str = None) -> dict:
    from integrations.azuracast.client import update_media, get_station_id
    station_id = get_station_id(station)
    kwargs = {}
    if artist is not None:
        kwargs["artist"] = artist
    if title is not None:
        kwargs["title"] = title
    if album is not None:
        kwargs["album"] = album
    if genre is not None:
        kwargs["genre"] = genre
    return update_media(station_id, media_id, **kwargs)


@tool(
    name="radio_delete_song",
    description="Delete a song from the media library.",
    parameters={
        "media_id": "int - the media file ID to delete",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_delete_song(media_id: int, station: str = None) -> dict:
    from integrations.azuracast.client import delete_media, get_station_id
    station_id = get_station_id(station)
    return delete_media(station_id, media_id)


@tool(
    name="radio_create_folder",
    description="Create a folder in the media library.",
    parameters={
        "folder_path": "string - path for the new folder (e.g., 'Rock/Classic')",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_create_folder(folder_path: str, station: str = None) -> dict:
    from integrations.azuracast.client import create_media_folder, get_station_id
    station_id = get_station_id(station)
    return create_media_folder(station_id, folder_path)


# ==================== RADIO: QUEUE & REQUESTS ====================

@tool(
    name="radio_clear_queue",
    description="Clear all songs from the upcoming queue.",
    parameters={"station": "string (optional) - station name, shortcode, or ID"},
)
def radio_clear_queue(station: str = None) -> dict:
    from integrations.azuracast.client import clear_queue, get_station_id
    station_id = get_station_id(station)
    return clear_queue(station_id)


@tool(
    name="radio_skip_song",
    description="Skip the currently playing song and move to next.",
    parameters={"station": "string (optional) - station name, shortcode, or ID"},
)
def radio_skip_song(station: str = None) -> dict:
    from integrations.azuracast.client import skip_song, get_station_id
    station_id = get_station_id(station)
    return skip_song(station_id)


@tool(
    name="radio_request_song",
    description="Submit a song request to be played.",
    parameters={
        "query": "string - search for the song to request",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_request_song(query: str, station: str = None) -> dict:
    from integrations.azuracast.client import search_requestable_songs, submit_request, get_station_id
    station_id = get_station_id(station)
    # Search for the song
    results = search_requestable_songs(station_id, query)
    if not results:
        return {"error": f"No requestable song found matching '{query}'"}
    # Request the first matching song
    song = results[0]
    result = submit_request(station_id, song["request_id"])
    result["requested_song"] = f"{song['artist']} - {song['title']}"
    return result


# ==================== RADIO: MOUNT POINTS ====================

@tool(
    name="radio_list_mounts",
    description="List all mount points/stream URLs for a station.",
    parameters={"station": "string (optional) - station name, shortcode, or ID"},
)
def radio_list_mounts(station: str = None) -> list[dict]:
    from integrations.azuracast.client import list_mounts, get_station_id
    station_id = get_station_id(station)
    return list_mounts(station_id)


@tool(
    name="radio_create_mount",
    description="Create a new mount point/stream for a station.",
    parameters={
        "name": "string - mount point name (e.g., '/radio.mp3')",
        "display_name": "string (optional) - friendly display name",
        "format": "string (optional) - audio format: mp3, ogg, opus, aac, flac (default: mp3)",
        "bitrate": "int (optional) - stream bitrate in kbps (default: 128)",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_create_mount(name: str, display_name: str = None, format: str = "mp3",
                       bitrate: int = 128, station: str = None) -> dict:
    from integrations.azuracast.client import create_mount, get_station_id
    station_id = get_station_id(station)
    return create_mount(station_id, name, display_name, False, format, bitrate)


@tool(
    name="radio_delete_mount",
    description="Delete a mount point/stream.",
    parameters={
        "mount_id": "int - the mount point ID",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_delete_mount(mount_id: int, station: str = None) -> dict:
    from integrations.azuracast.client import delete_mount, get_station_id
    station_id = get_station_id(station)
    return delete_mount(station_id, mount_id)


# ==================== RADIO: PLAYLISTS (EXPANDED) ====================

@tool(
    name="radio_create_playlist",
    description="Create a new playlist on a station.",
    parameters={
        "name": "string - playlist name",
        "type": "string (optional) - playlist type: default, once_per_x_songs, once_per_x_minutes, once_per_hour, once_per_day, advanced (default: default)",
        "weight": "int (optional) - playlist weight/priority 1-25 (default: 3)",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_create_playlist(name: str, type: str = "default", weight: int = 3,
                          station: str = None) -> dict:
    from integrations.azuracast.client import create_playlist, get_station_id
    station_id = get_station_id(station)
    return create_playlist(station_id, name, type, weight)


@tool(
    name="radio_delete_playlist",
    description="Delete a playlist from a station.",
    parameters={
        "playlist_id": "int - the playlist ID",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_delete_playlist(playlist_id: int, station: str = None) -> dict:
    from integrations.azuracast.client import delete_playlist, get_station_id
    station_id = get_station_id(station)
    return delete_playlist(station_id, playlist_id)


@tool(
    name="radio_reshuffle_playlist",
    description="Reshuffle a playlist's playback order.",
    parameters={
        "playlist_id": "int - the playlist ID",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_reshuffle_playlist(playlist_id: int, station: str = None) -> dict:
    from integrations.azuracast.client import reshuffle_playlist, get_station_id
    station_id = get_station_id(station)
    return reshuffle_playlist(station_id, playlist_id)


# ==================== RADIO: WEBHOOKS ====================

@tool(
    name="radio_list_webhooks",
    description="List all webhooks for a station.",
    parameters={"station": "string (optional) - station name, shortcode, or ID"},
)
def radio_list_webhooks(station: str = None) -> list[dict]:
    from integrations.azuracast.client import list_webhooks, get_station_id
    station_id = get_station_id(station)
    return list_webhooks(station_id)


@tool(
    name="radio_toggle_webhook",
    description="Toggle a webhook on/off.",
    parameters={
        "webhook_id": "int - the webhook ID",
        "station": "string (optional) - station name, shortcode, or ID",
    },
)
def radio_toggle_webhook(webhook_id: int, station: str = None) -> dict:
    from integrations.azuracast.client import toggle_webhook, get_station_id
    station_id = get_station_id(station)
    return toggle_webhook(station_id, webhook_id)


# ==================== RADIO: SYSTEM ====================

@tool(
    name="radio_system_status",
    description="Get AzuraCast server system status (CPU, memory, disk).",
    parameters={},
)
def radio_system_status() -> dict:
    from integrations.azuracast.client import get_system_status
    return get_system_status()


@tool(
    name="radio_services_status",
    description="Get status of all AzuraCast services.",
    parameters={},
)
def radio_services_status() -> list[dict]:
    from integrations.azuracast.client import get_services_status
    return get_services_status()


# ==================== META ADS TOOLS ====================

@tool(
    name="meta_ads_account",
    description="Get Meta (Facebook/Instagram) Ads account info and status.",
    parameters={},
)
def meta_ads_account() -> dict:
    from integrations.meta_ads.client import get_ad_account_info
    return get_ad_account_info()


@tool(
    name="meta_ads_summary",
    description="Get a quick summary of Meta Ads performance (last 7 days).",
    parameters={},
)
def meta_ads_summary() -> dict:
    from integrations.meta_ads.client import get_meta_ads_summary
    return get_meta_ads_summary()


@tool(
    name="meta_ads_performance",
    description="Get overall Meta Ads account performance metrics.",
    parameters={
        "period": "string (optional) - time period: today, yesterday, last_7d, last_14d, last_30d, this_month, last_month (default: last_7d)",
    },
)
def meta_ads_performance(period: str = "last_7d") -> dict:
    from integrations.meta_ads.client import get_account_insights
    return get_account_insights(period)


@tool(
    name="meta_ads_campaigns",
    description="List all Meta Ads campaigns with their status and budgets.",
    parameters={
        "status": "string (optional) - filter by status: ACTIVE, PAUSED, DELETED, ARCHIVED",
    },
)
def meta_ads_campaigns(status: str = None) -> list[dict]:
    from integrations.meta_ads.client import list_campaigns
    return list_campaigns(status)


@tool(
    name="meta_ads_campaign_insights",
    description="Get performance metrics for Meta Ads campaigns.",
    parameters={
        "campaign_id": "string (optional) - specific campaign ID, or all campaigns if not specified",
        "period": "string (optional) - time period: last_7d, last_14d, last_30d, etc. (default: last_7d)",
    },
)
def meta_ads_campaign_insights(campaign_id: str = None, period: str = "last_7d") -> list[dict]:
    from integrations.meta_ads.client import get_campaign_insights
    return get_campaign_insights(campaign_id, period)


@tool(
    name="meta_ads_ad_sets",
    description="List ad sets (audiences/targeting groups) in Meta Ads.",
    parameters={
        "campaign_id": "string (optional) - filter by campaign ID",
    },
)
def meta_ads_ad_sets(campaign_id: str = None) -> list[dict]:
    from integrations.meta_ads.client import list_ad_sets
    return list_ad_sets(campaign_id)


@tool(
    name="meta_ads_ad_set_insights",
    description="Get performance metrics for Meta Ads ad sets.",
    parameters={
        "campaign_id": "string (optional) - specific campaign ID",
        "period": "string (optional) - time period (default: last_7d)",
    },
)
def meta_ads_ad_set_insights(campaign_id: str = None, period: str = "last_7d") -> list[dict]:
    from integrations.meta_ads.client import get_ad_set_insights
    return get_ad_set_insights(campaign_id, period)


@tool(
    name="meta_ads_list",
    description="List individual ads in Meta Ads.",
    parameters={
        "ad_set_id": "string (optional) - filter by ad set ID",
    },
)
def meta_ads_list(ad_set_id: str = None) -> list[dict]:
    from integrations.meta_ads.client import list_ads
    return list_ads(ad_set_id)


@tool(
    name="meta_ads_ad_insights",
    description="Get performance metrics for individual Meta ads.",
    parameters={
        "ad_set_id": "string (optional) - specific ad set ID",
        "period": "string (optional) - time period (default: last_7d)",
    },
)
def meta_ads_ad_insights(ad_set_id: str = None, period: str = "last_7d") -> list[dict]:
    from integrations.meta_ads.client import get_ad_insights
    return get_ad_insights(ad_set_id, period)


@tool(
    name="meta_ads_audience",
    description="Get audience demographic breakdown (age, gender, platform, device).",
    parameters={
        "period": "string (optional) - time period (default: last_7d)",
    },
)
def meta_ads_audience(period: str = "last_7d") -> dict:
    from integrations.meta_ads.client import get_audience_insights
    return get_audience_insights(period)


@tool(
    name="meta_ads_placements",
    description="Get performance by ad placement (feed, stories, reels, etc.).",
    parameters={
        "period": "string (optional) - time period (default: last_7d)",
    },
)
def meta_ads_placements(period: str = "last_7d") -> list[dict]:
    from integrations.meta_ads.client import get_placement_insights
    return get_placement_insights(period)


@tool(
    name="meta_ads_issues",
    description="Check for Meta Ads delivery issues, disapproved ads, or policy violations.",
    parameters={},
)
def meta_ads_issues() -> list[dict]:
    from integrations.meta_ads.client import get_delivery_issues
    return get_delivery_issues()


@tool(
    name="meta_ads_daily_spend",
    description="Get daily spend breakdown for Meta Ads.",
    parameters={
        "days": "int (optional) - number of days to show (default: 7)",
    },
)
def meta_ads_daily_spend(days: int = 7) -> list[dict]:
    from integrations.meta_ads.client import get_spend_by_day
    return get_spend_by_day(days)


def register_all():
    """Import this module to register all tools."""
    pass
