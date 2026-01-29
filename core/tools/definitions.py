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


def register_all():
    """Import this module to register all tools."""
    pass
