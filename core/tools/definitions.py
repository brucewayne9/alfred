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


def register_all():
    """Import this module to register all tools."""
    pass
