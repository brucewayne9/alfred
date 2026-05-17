"""DataForSEO v3 API client.

Wraps a small slice of the DataForSEO REST API used by Alfred's SEO pipeline:
keyword search volume, keyword suggestions, organic SERP, on-page audit,
and backlinks summary.

Credentials come from `config.settings.settings` (NOT `os.environ` at import
time) per [[feedback_never_read_env_at_import]]. The `alfred.service` systemd
unit has no `EnvironmentFile=`, so env-at-import silently breaks production.
Settings are read inside `__init__`, never at module top-level.

All cost-tracking lives on the client instance: `total_cost_usd` accumulates
across every successful call (DataForSEO returns a `cost` field per task).
Each call is logged at INFO with its endpoint path and cost.

No retries. Caller decides retry policy.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx

from config.settings import settings

log = logging.getLogger("dataforseo")

API_BASE = "https://api.dataforseo.com/v3"
DEFAULT_TIMEOUT = 60.0
SUCCESS_STATUS_CODE = 20000
KEYWORD_VOLUME_BATCH = 1000


class DataForSEOError(Exception):
    """Raised when DataForSEO returns a non-20000 status_code or HTTP error."""

    def __init__(self, status_code: int, status_message: str, endpoint: str):
        self.status_code = status_code
        self.status_message = status_message
        self.endpoint = endpoint
        super().__init__(f"DataForSEO {endpoint} failed: {status_code} {status_message}")


class DataForSEOClient:
    """Synchronous DataForSEO v3 client."""

    def __init__(
        self,
        login: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        base_url: str = API_BASE,
    ):
        # Read settings inside __init__, never at module import.
        self.login = login or settings.dataforseo_login
        self.password = password or settings.dataforseo_password
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self.total_cost_usd: float = 0.0

        if not self.login or not self.password:
            raise ValueError(
                "DataForSEO credentials missing — set DATAFORSEO_LOGIN/DATAFORSEO_PASSWORD "
                "in config/.env"
            )

    # ------------------------------------------------------------------ HTTP

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        timeout: Optional[float] = None,
    ) -> dict:
        """Make an authenticated request and return parsed JSON.

        Tracks cost (top-level `cost` field on success), logs per-call,
        raises DataForSEOError on non-20000 status_code or HTTP non-2xx.
        """
        url = f"{self.base_url}{path}"
        auth = (self.login, self.password)
        t = timeout if timeout is not None else self.timeout

        try:
            resp = httpx.request(
                method,
                url,
                auth=auth,
                json=json_body,
                timeout=t,
                headers={"Content-Type": "application/json"},
            )
        except httpx.HTTPError as e:
            log.error("dfs %s transport error: %s", path, e)
            raise

        if resp.status_code >= 300:
            log.error("dfs %s HTTP %s: %s", path, resp.status_code, resp.text[:200])
            raise DataForSEOError(resp.status_code, f"HTTP {resp.status_code}", path)

        data = resp.json()

        status_code = data.get("status_code")
        status_message = data.get("status_message", "")
        if status_code != SUCCESS_STATUS_CODE:
            log.error("dfs %s status=%s msg=%s", path, status_code, status_message)
            raise DataForSEOError(status_code or 0, status_message, path)

        cost = float(data.get("cost") or 0.0)
        if cost:
            self.total_cost_usd += cost
        log.info("dfs %s cost=$%.4f", path, cost)
        return data

    # --------------------------------------------------------- Keyword volume

    def keyword_search_volume(
        self,
        keywords: list[str],
        location_code: int = 2840,
        language_code: str = "en",
    ) -> list[dict]:
        """Google Ads search-volume metrics for up to N keywords.

        Chunks internally at 1000/keyword call. Returns one dict per keyword:
            {keyword, search_volume, cpc, competition, competition_level, monthly_searches}
        """
        if not keywords:
            return []

        path = "/keywords_data/google_ads/search_volume/live"
        results: list[dict] = []

        for i in range(0, len(keywords), KEYWORD_VOLUME_BATCH):
            chunk = keywords[i : i + KEYWORD_VOLUME_BATCH]
            payload = [{
                "keywords": chunk,
                "location_code": location_code,
                "language_code": language_code,
            }]
            data = self._request("POST", path, json_body=payload)
            for task in data.get("tasks") or []:
                for item in task.get("result") or []:
                    if not item:
                        continue
                    # DataForSEO's Google Ads search_volume endpoint puts the
                    # level label (LOW/MEDIUM/HIGH) in `competition` and leaves
                    # `competition_level` unset. Newer Labs endpoints split them
                    # (competition as 0-1 float, competition_level as label).
                    # Normalize so callers always get both: a float when present
                    # and a label string when present.
                    raw_comp = item.get("competition")
                    raw_level = item.get("competition_level")
                    comp_float: Optional[float] = None
                    comp_level: Optional[str] = raw_level
                    if isinstance(raw_comp, (int, float)):
                        comp_float = float(raw_comp)
                    elif isinstance(raw_comp, str) and comp_level is None:
                        comp_level = raw_comp
                    results.append({
                        "keyword": item.get("keyword"),
                        "search_volume": item.get("search_volume"),
                        "cpc": item.get("cpc"),
                        "competition": comp_float,
                        "competition_level": comp_level,
                        "monthly_searches": item.get("monthly_searches") or [],
                    })
        return results

    # ----------------------------------------------------- Keyword suggestions

    def keyword_suggestions(
        self,
        seed_keyword: str,
        location_code: int = 2840,
        language_code: str = "en",
        limit: int = 100,
    ) -> list[dict]:
        """DataForSEO Labs keyword_suggestions for one seed term."""
        path = "/dataforseo_labs/google/keyword_suggestions/live"
        payload = [{
            "keyword": seed_keyword,
            "location_code": location_code,
            "language_code": language_code,
            "limit": limit,
        }]
        data = self._request("POST", path, json_body=payload)

        results: list[dict] = []
        for task in data.get("tasks") or []:
            for r in task.get("result") or []:
                for item in r.get("items") or []:
                    if not item:
                        continue
                    info = item.get("keyword_info") or {}
                    props = item.get("keyword_properties") or {}
                    intent = item.get("search_intent_info") or {}
                    results.append({
                        "keyword": item.get("keyword"),
                        "search_volume": info.get("search_volume"),
                        "competition": info.get("competition"),
                        "competition_level": info.get("competition_level"),
                        "cpc": info.get("cpc"),
                        "keyword_difficulty": props.get("keyword_difficulty"),
                        "search_intent": intent.get("main_intent"),
                    })
        return results

    # -------------------------------------------------------------- SERP live

    def serp_organic(
        self,
        keyword: str,
        location_code: int = 2840,
        language_code: str = "en",
        depth: int = 100,
    ) -> dict:
        """Google organic SERP for a keyword. Returns ranked results list."""
        path = "/serp/google/organic/live/regular"
        payload = [{
            "keyword": keyword,
            "location_code": location_code,
            "language_code": language_code,
            "depth": depth,
        }]
        data = self._request("POST", path, json_body=payload)

        results: list[dict] = []
        total_count = 0
        for task in data.get("tasks") or []:
            for r in task.get("result") or []:
                total_count = r.get("total_count") or total_count
                for item in r.get("items") or []:
                    if (item or {}).get("type") != "organic":
                        continue
                    results.append({
                        "rank_group": item.get("rank_group"),
                        "rank_absolute": item.get("rank_absolute"),
                        "domain": item.get("domain"),
                        "url": item.get("url"),
                        "title": item.get("title"),
                        "snippet": item.get("description") or item.get("snippet"),
                    })
        return {"keyword": keyword, "results": results, "total_count": total_count}

    # --------------------------------------------------------- On-page audit
    # Task-based: post → poll tasks_ready → fetch summary. Block by default.

    def onpage_task_post(self, target: str, max_crawl_pages: int = 100) -> str:
        """Create an on-page audit task. Returns task_id."""
        path = "/on_page/task_post"
        payload = [{
            "target": target,
            "max_crawl_pages": max_crawl_pages,
        }]
        data = self._request("POST", path, json_body=payload)
        tasks = data.get("tasks") or []
        if not tasks or not tasks[0].get("id"):
            raise DataForSEOError(0, "no task id returned", path)
        return tasks[0]["id"]

    def onpage_tasks_ready(self) -> list[str]:
        """Return list of task_ids that are ready for retrieval."""
        path = "/on_page/tasks_ready"
        data = self._request("GET", path)
        ready: list[str] = []
        for task in data.get("tasks") or []:
            for r in task.get("result") or []:
                if (r or {}).get("id"):
                    ready.append(r["id"])
        return ready

    def onpage_summary_get(self, task_id: str) -> Optional[dict]:
        """Fetch on-page summary for a completed task. None if not ready."""
        path = f"/on_page/summary/{task_id}"
        try:
            data = self._request("GET", path)
        except DataForSEOError as e:
            # 40602 = task in queue / not ready yet
            if e.status_code in (40602, 40400):
                return None
            raise
        tasks = data.get("tasks") or []
        if not tasks:
            return None
        result = (tasks[0].get("result") or [None])[0]
        return result

    def onpage_summary(
        self,
        target: str,
        max_crawl_pages: int = 100,
        timeout_s: float = 600.0,
        poll_interval_s: float = 15.0,
    ) -> tuple[str, dict]:
        """Blocking convenience: post audit, poll until ready, return (task_id, summary).

        Returns the task_id alongside the summary so callers can fetch per-page
        detail via /on_page/pages without re-discovering it from /tasks_ready
        (which returns empty once a task has been read).
        """
        task_id = self.onpage_task_post(target, max_crawl_pages=max_crawl_pages)
        log.info("dfs on_page task posted id=%s target=%s", task_id, target)

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ready_ids = set(self.onpage_tasks_ready())
            if task_id in ready_ids:
                summary = self.onpage_summary_get(task_id)
                if summary is not None:
                    return task_id, summary
            time.sleep(poll_interval_s)

        raise DataForSEOError(
            0,
            f"on_page audit for {target} timed out after {timeout_s:.0f}s (task_id={task_id})",
            "/on_page/summary",
        )

    # -------------------------------------------------------- Backlinks

    def backlinks_summary(self, target: str) -> dict:
        """Backlink-profile summary (rank, referring domains, etc.)."""
        path = "/backlinks/summary/live"
        payload = [{"target": target}]
        data = self._request("POST", path, json_body=payload)
        for task in data.get("tasks") or []:
            result = (task.get("result") or [None])[0]
            if result:
                return result
        return {}
