"""Agent system for Alfred - spawn and manage specialized agents.

This module enables Alfred to spawn multiple agents with different models
to work on tasks in parallel or sequence. The main Claude orchestrator
decides when to spawn agents and which models to use.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional
import json

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Status of an agent task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentType(str, Enum):
    """Types of specialized agents."""
    CODER = "coder"           # Code generation and review
    RESEARCHER = "researcher"  # Information gathering
    ANALYST = "analyst"       # Data analysis and reasoning
    WRITER = "writer"         # Content creation
    PLANNER = "planner"       # Task planning and breakdown
    EXECUTOR = "executor"     # Tool execution
    GENERAL = "general"       # General purpose


# Map agent types to preferred models
AGENT_MODELS = {
    AgentType.CODER: "cloud:devstral-24b",
    AgentType.RESEARCHER: "cloud:deepseek-v3.2",
    AgentType.ANALYST: "cloud:deepseek-v3.1",
    AgentType.WRITER: "cloud:gpt-oss-120b",
    AgentType.PLANNER: "claude:sonnet",
    AgentType.EXECUTOR: "cloud:nemotron-30b",
    AgentType.GENERAL: "cloud:ministral-8b",
}


@dataclass
class AgentTask:
    """A task to be executed by an agent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal: str = ""
    context: str = ""
    agent_type: AgentType = AgentType.GENERAL
    model_override: Optional[str] = None  # Force specific model
    parent_id: Optional[str] = None  # Parent agent/task ID
    status: AgentStatus = AgentStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "context": self.context[:200] + "..." if len(self.context) > 200 else self.context,
            "agent_type": self.agent_type.value,
            "status": self.status.value,
            "result": self.result[:500] + "..." if isinstance(self.result, str) and len(self.result) > 500 else self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": (self.completed_at - self.started_at).total_seconds() if self.completed_at and self.started_at else None,
        }


@dataclass
class Agent:
    """An agent instance that executes tasks."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_type: AgentType = AgentType.GENERAL
    model: str = ""
    task: Optional[AgentTask] = None
    messages: list = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if not self.model:
            self.model = AGENT_MODELS.get(self.agent_type, "cloud:ministral-8b")


class AgentPool:
    """Pool of agents that can be spawned to work on tasks.

    The AgentPool manages the lifecycle of agents and tasks:
    - Spawn agents with specific capabilities
    - Queue and execute tasks
    - Track results and status
    - Enable inter-agent communication via shared context
    """

    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self._agents: dict[str, Agent] = {}
        self._tasks: dict[str, AgentTask] = {}
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._results: dict[str, Any] = {}
        self._running = False
        self._workers: list[asyncio.Task] = []

    async def start(self):
        """Start the agent pool workers."""
        if self._running:
            return
        self._running = True
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
        logger.info(f"AgentPool started with {self.max_concurrent} workers")

    async def stop(self):
        """Stop the agent pool."""
        self._running = False
        for worker in self._workers:
            worker.cancel()
        self._workers.clear()
        logger.info("AgentPool stopped")

    async def _worker(self, worker_id: str):
        """Worker that processes tasks from the queue."""
        while self._running:
            try:
                task = await asyncio.wait_for(self._task_queue.get(), timeout=1.0)
                await self._execute_task(task, worker_id)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    async def _execute_task(self, task: AgentTask, worker_id: str):
        """Execute a task with the appropriate agent."""
        task.status = AgentStatus.RUNNING
        task.started_at = datetime.now(timezone.utc)

        # Create agent for this task
        agent = Agent(
            agent_type=task.agent_type,
            model=task.model_override or AGENT_MODELS.get(task.agent_type, "cloud:ministral-8b"),
            task=task,
        )
        self._agents[agent.id] = agent

        logger.info(f"[{worker_id}] Executing task {task.id} with agent {agent.id} ({agent.model})")

        try:
            # Build the agent's system prompt
            system_prompt = self._build_agent_prompt(task, agent)

            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task.goal},
            ]

            if task.context:
                messages[1]["content"] = f"Context:\n{task.context}\n\nTask:\n{task.goal}"

            # Execute with the appropriate model
            from core.brain.router import query_smart
            from core.brain.models import MODELS

            model_config = MODELS.get(agent.model)
            if not model_config:
                raise ValueError(f"Unknown model: {agent.model}")

            result = await query_smart(
                task.goal,
                messages=messages,
                force_model=agent.model,
            )

            task.result = result.get("response", "")
            task.status = AgentStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc)

            logger.info(f"Task {task.id} completed successfully")

        except Exception as e:
            task.error = str(e)
            task.status = AgentStatus.FAILED
            task.completed_at = datetime.now(timezone.utc)
            logger.error(f"Task {task.id} failed: {e}")

        finally:
            self._tasks[task.id] = task
            self._results[task.id] = task.to_dict()

    def _build_agent_prompt(self, task: AgentTask, agent: Agent) -> str:
        """Build system prompt for an agent based on its type."""
        base_prompts = {
            AgentType.CODER: """You are a coding specialist agent. Your role is to:
- Write clean, efficient, well-documented code
- Review and improve existing code
- Debug issues and suggest fixes
- Follow best practices for the relevant language/framework
Be precise and provide working code. Include explanations of your approach.""",

            AgentType.RESEARCHER: """You are a research specialist agent. Your role is to:
- Gather and synthesize information
- Identify key facts and insights
- Provide well-sourced, accurate information
- Summarize complex topics clearly
Be thorough but concise. Cite sources when available.""",

            AgentType.ANALYST: """You are an analysis specialist agent. Your role is to:
- Analyze data and information critically
- Identify patterns, trends, and insights
- Provide reasoned conclusions with supporting evidence
- Consider multiple perspectives and trade-offs
Be analytical and objective. Support conclusions with reasoning.""",

            AgentType.WRITER: """You are a writing specialist agent. Your role is to:
- Create clear, engaging content
- Adapt tone and style to the context
- Structure information logically
- Edit and improve existing text
Be creative yet clear. Match the appropriate tone for the task.""",

            AgentType.PLANNER: """You are a planning specialist agent. Your role is to:
- Break down complex tasks into steps
- Identify dependencies and priorities
- Create actionable plans
- Anticipate challenges and suggest mitigations
Be systematic and thorough. Provide clear, actionable steps.""",

            AgentType.EXECUTOR: """You are an execution specialist agent. Your role is to:
- Execute tasks using available tools
- Follow instructions precisely
- Report results clearly
- Handle errors gracefully
Be efficient and accurate. Report what was done and the outcome.""",

            AgentType.GENERAL: """You are a helpful assistant agent. Your role is to:
- Understand and complete the given task
- Provide clear, accurate responses
- Ask for clarification if needed
Be helpful and thorough.""",
        }

        prompt = base_prompts.get(agent.agent_type, base_prompts[AgentType.GENERAL])

        # Add parent context if available
        if task.parent_id and task.parent_id in self._tasks:
            parent = self._tasks[task.parent_id]
            if parent.result:
                prompt += f"\n\nContext from parent task:\n{parent.result[:1000]}"

        return prompt

    async def spawn_agent(
        self,
        goal: str,
        agent_type: AgentType = AgentType.GENERAL,
        context: str = "",
        model_override: str = None,
        parent_id: str = None,
        metadata: dict = None,
    ) -> str:
        """Spawn an agent to work on a task.

        Args:
            goal: What the agent should accomplish
            agent_type: Type of specialist agent
            context: Additional context for the task
            model_override: Force a specific model
            parent_id: Parent task ID for context chaining
            metadata: Additional metadata

        Returns:
            Task ID for tracking
        """
        task = AgentTask(
            goal=goal,
            context=context,
            agent_type=agent_type,
            model_override=model_override,
            parent_id=parent_id,
            metadata=metadata or {},
        )

        self._tasks[task.id] = task
        await self._task_queue.put(task)

        logger.info(f"Spawned {agent_type.value} agent for task {task.id}")
        return task.id

    async def get_task_status(self, task_id: str) -> dict | None:
        """Get the status of a task."""
        task = self._tasks.get(task_id)
        return task.to_dict() if task else None

    async def get_task_result(self, task_id: str, wait: bool = False, timeout: float = 60) -> Any:
        """Get the result of a task.

        Args:
            task_id: The task ID
            wait: If True, wait for completion
            timeout: Max seconds to wait

        Returns:
            Task result or None if not complete
        """
        task = self._tasks.get(task_id)
        if not task:
            return None

        if wait and task.status in (AgentStatus.PENDING, AgentStatus.RUNNING):
            start = asyncio.get_event_loop().time()
            while task.status in (AgentStatus.PENDING, AgentStatus.RUNNING):
                if asyncio.get_event_loop().time() - start > timeout:
                    break
                await asyncio.sleep(0.5)

        return task.to_dict()

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        task = self._tasks.get(task_id)
        if not task:
            return False

        if task.status in (AgentStatus.PENDING, AgentStatus.RUNNING):
            task.status = AgentStatus.CANCELLED
            task.completed_at = datetime.now(timezone.utc)
            return True
        return False

    def list_tasks(self, status: AgentStatus = None) -> list[dict]:
        """List all tasks, optionally filtered by status."""
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return [t.to_dict() for t in sorted(tasks, key=lambda t: t.created_at, reverse=True)]

    def list_agents(self) -> list[dict]:
        """List all active agents."""
        return [
            {
                "id": a.id,
                "type": a.agent_type.value,
                "model": a.model,
                "task_id": a.task.id if a.task else None,
                "created_at": a.created_at.isoformat(),
            }
            for a in self._agents.values()
        ]


# Global agent pool instance
_pool: AgentPool | None = None


def get_agent_pool() -> AgentPool:
    """Get the global agent pool instance."""
    global _pool
    if _pool is None:
        _pool = AgentPool(max_concurrent=3)
    return _pool


async def initialize_agent_pool():
    """Initialize and start the agent pool."""
    pool = get_agent_pool()
    await pool.start()
    return pool


# Convenience functions for spawning agents
async def spawn_coder(goal: str, context: str = "") -> str:
    """Spawn a coder agent."""
    pool = get_agent_pool()
    return await pool.spawn_agent(goal, AgentType.CODER, context)


async def spawn_researcher(goal: str, context: str = "") -> str:
    """Spawn a researcher agent."""
    pool = get_agent_pool()
    return await pool.spawn_agent(goal, AgentType.RESEARCHER, context)


async def spawn_analyst(goal: str, context: str = "") -> str:
    """Spawn an analyst agent."""
    pool = get_agent_pool()
    return await pool.spawn_agent(goal, AgentType.ANALYST, context)


async def spawn_writer(goal: str, context: str = "") -> str:
    """Spawn a writer agent."""
    pool = get_agent_pool()
    return await pool.spawn_agent(goal, AgentType.WRITER, context)


async def spawn_planner(goal: str, context: str = "") -> str:
    """Spawn a planner agent."""
    pool = get_agent_pool()
    return await pool.spawn_agent(goal, AgentType.PLANNER, context)
