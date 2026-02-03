"""Firecrawl API client for web scraping and crawling.

Firecrawl provides:
- Single page scraping (URL -> markdown/HTML)
- Website crawling (crawl entire sites)
- Google search scraping
- Structured data extraction

Docs: https://docs.firecrawl.dev/
"""

import logging
import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv("/home/aialfred/alfred/config/.env")

logger = logging.getLogger(__name__)

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
FIRECRAWL_BASE_URL = "https://api.firecrawl.dev/v1"


def is_configured() -> bool:
    """Check if Firecrawl is configured."""
    return bool(FIRECRAWL_API_KEY)


def _headers() -> dict:
    """Get API headers."""
    return {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json",
    }


def scrape_url(
    url: str,
    formats: list[str] = None,
    only_main_content: bool = True,
    include_tags: list[str] = None,
    exclude_tags: list[str] = None,
    wait_for: int = 0,
) -> dict:
    """Scrape a single URL and return its content.

    Args:
        url: The URL to scrape
        formats: Output formats - ['markdown', 'html', 'rawHtml', 'links', 'screenshot']
                 Default is ['markdown']
        only_main_content: Remove headers/footers/navs (default True)
        include_tags: Only include these HTML tags
        exclude_tags: Exclude these HTML tags
        wait_for: Wait ms for JS to load (for dynamic sites)

    Returns:
        dict with markdown/html content, metadata, links
    """
    if not is_configured():
        return {"success": False, "error": "Firecrawl API key not configured"}

    formats = formats or ["markdown"]

    payload = {
        "url": url,
        "formats": formats,
        "onlyMainContent": only_main_content,
    }

    if include_tags:
        payload["includeTags"] = include_tags
    if exclude_tags:
        payload["excludeTags"] = exclude_tags
    if wait_for:
        payload["waitFor"] = wait_for

    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{FIRECRAWL_BASE_URL}/scrape",
                headers=_headers(),
                json=payload,
            )

            if resp.status_code == 200:
                data = resp.json()
                return {
                    "success": True,
                    "url": url,
                    "markdown": data.get("data", {}).get("markdown", ""),
                    "html": data.get("data", {}).get("html", ""),
                    "metadata": data.get("data", {}).get("metadata", {}),
                    "links": data.get("data", {}).get("links", []),
                }
            else:
                return {"success": False, "error": f"{resp.status_code}: {resp.text}"}

    except Exception as e:
        logger.error(f"Firecrawl scrape failed: {e}")
        return {"success": False, "error": str(e)}


def crawl_website(
    url: str,
    max_depth: int = 2,
    limit: int = 10,
    include_paths: list[str] = None,
    exclude_paths: list[str] = None,
    allow_subdomains: bool = False,
    wait: bool = True,
    poll_interval: int = 5,
    max_wait: int = 300,
) -> dict:
    """Crawl an entire website starting from a URL.

    Args:
        url: Starting URL
        max_depth: How deep to crawl (default 2)
        limit: Max pages to crawl (default 10)
        include_paths: Only crawl paths matching these patterns
        exclude_paths: Skip paths matching these patterns
        allow_subdomains: Include subdomains in crawl
        wait: Wait for crawl to complete (default True)
        poll_interval: Seconds between status checks
        max_wait: Max seconds to wait for completion

    Returns:
        dict with crawled pages content
    """
    if not is_configured():
        return {"success": False, "error": "Firecrawl API key not configured"}

    payload = {
        "url": url,
        "maxDepth": max_depth,
        "limit": limit,
        "allowSubdomains": allow_subdomains,
        "scrapeOptions": {
            "formats": ["markdown"],
            "onlyMainContent": True,
        },
    }

    if include_paths:
        payload["includePaths"] = include_paths
    if exclude_paths:
        payload["excludePaths"] = exclude_paths

    try:
        with httpx.Client(timeout=60) as client:
            # Start the crawl
            resp = client.post(
                f"{FIRECRAWL_BASE_URL}/crawl",
                headers=_headers(),
                json=payload,
            )

            if resp.status_code != 200:
                return {"success": False, "error": f"{resp.status_code}: {resp.text}"}

            data = resp.json()
            crawl_id = data.get("id")

            if not crawl_id:
                return {"success": False, "error": "No crawl ID returned"}

            if not wait:
                return {
                    "success": True,
                    "crawl_id": crawl_id,
                    "status": "started",
                    "message": f"Crawl started. Check status with crawl_id: {crawl_id}",
                }

            # Poll for completion
            elapsed = 0
            while elapsed < max_wait:
                status = get_crawl_status(crawl_id)

                if status.get("status") == "completed":
                    return {
                        "success": True,
                        "crawl_id": crawl_id,
                        "status": "completed",
                        "pages_crawled": status.get("total", 0),
                        "pages": status.get("data", []),
                    }
                elif status.get("status") == "failed":
                    return {"success": False, "error": "Crawl failed", "details": status}

                time.sleep(poll_interval)
                elapsed += poll_interval

            return {
                "success": False,
                "error": f"Crawl timed out after {max_wait}s",
                "crawl_id": crawl_id,
            }

    except Exception as e:
        logger.error(f"Firecrawl crawl failed: {e}")
        return {"success": False, "error": str(e)}


def get_crawl_status(crawl_id: str) -> dict:
    """Check the status of a crawl job.

    Args:
        crawl_id: The crawl job ID

    Returns:
        dict with status and crawled data
    """
    if not is_configured():
        return {"success": False, "error": "Firecrawl API key not configured"}

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"{FIRECRAWL_BASE_URL}/crawl/{crawl_id}",
                headers=_headers(),
            )

            if resp.status_code == 200:
                return resp.json()
            else:
                return {"success": False, "error": f"{resp.status_code}: {resp.text}"}

    except Exception as e:
        logger.error(f"Firecrawl status check failed: {e}")
        return {"success": False, "error": str(e)}


def search_google(
    query: str,
    limit: int = 5,
    scrape_results: bool = True,
) -> dict:
    """Search Google and optionally scrape the result pages.

    Args:
        query: Search query
        limit: Number of results (default 5)
        scrape_results: Also scrape the content of each result page

    Returns:
        dict with search results and their content
    """
    if not is_configured():
        return {"success": False, "error": "Firecrawl API key not configured"}

    payload = {
        "query": query,
        "limit": limit,
    }

    if scrape_results:
        payload["scrapeOptions"] = {
            "formats": ["markdown"],
            "onlyMainContent": True,
        }

    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{FIRECRAWL_BASE_URL}/search",
                headers=_headers(),
                json=payload,
            )

            if resp.status_code == 200:
                data = resp.json()
                results = data.get("data", [])

                return {
                    "success": True,
                    "query": query,
                    "result_count": len(results),
                    "results": [
                        {
                            "title": r.get("metadata", {}).get("title", ""),
                            "url": r.get("url", ""),
                            "description": r.get("metadata", {}).get("description", ""),
                            "markdown": r.get("markdown", "")[:5000] if scrape_results else None,
                        }
                        for r in results
                    ],
                }
            else:
                return {"success": False, "error": f"{resp.status_code}: {resp.text}"}

    except Exception as e:
        logger.error(f"Firecrawl search failed: {e}")
        return {"success": False, "error": str(e)}


def extract_data(
    url: str,
    schema: dict,
    prompt: str = None,
) -> dict:
    """Extract structured data from a page using LLM.

    Args:
        url: URL to extract from
        schema: JSON schema defining what to extract
        prompt: Optional prompt to guide extraction

    Returns:
        dict with extracted structured data

    Example schema:
        {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "price": {"type": "number"},
                "features": {"type": "array", "items": {"type": "string"}}
            }
        }
    """
    if not is_configured():
        return {"success": False, "error": "Firecrawl API key not configured"}

    payload = {
        "url": url,
        "formats": ["extract"],
        "extract": {
            "schema": schema,
        },
    }

    if prompt:
        payload["extract"]["prompt"] = prompt

    try:
        with httpx.Client(timeout=90) as client:
            resp = client.post(
                f"{FIRECRAWL_BASE_URL}/scrape",
                headers=_headers(),
                json=payload,
            )

            if resp.status_code == 200:
                data = resp.json()
                return {
                    "success": True,
                    "url": url,
                    "extracted": data.get("data", {}).get("extract", {}),
                    "metadata": data.get("data", {}).get("metadata", {}),
                }
            else:
                return {"success": False, "error": f"{resp.status_code}: {resp.text}"}

    except Exception as e:
        logger.error(f"Firecrawl extract failed: {e}")
        return {"success": False, "error": str(e)}


def scrape_to_lightrag(
    url: str,
    description: str = "",
) -> dict:
    """Scrape a URL and add it directly to LightRAG knowledge base.

    Args:
        url: URL to scrape
        description: Description for LightRAG indexing

    Returns:
        dict with scrape and upload status
    """
    # First scrape the URL
    scrape_result = scrape_url(url)

    if not scrape_result.get("success"):
        return scrape_result

    content = scrape_result.get("markdown", "")
    if not content:
        return {"success": False, "error": "No content extracted from URL"}

    # Upload to LightRAG
    try:
        import asyncio
        from integrations.lightrag.client import upload_text

        # Add URL and metadata to content
        title = scrape_result.get("metadata", {}).get("title", url)
        full_content = f"# {title}\n\nSource: {url}\n\n{content}"

        # Run async upload
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            upload_result = loop.run_until_complete(
                upload_text(full_content, description or f"Web content from {url}")
            )
        finally:
            loop.close()

        return {
            "success": upload_result.get("success", False),
            "url": url,
            "title": title,
            "content_length": len(content),
            "lightrag_result": upload_result,
        }

    except Exception as e:
        logger.error(f"Failed to upload to LightRAG: {e}")
        return {
            "success": False,
            "error": f"Scraped successfully but LightRAG upload failed: {e}",
            "content": content[:1000],
        }


def crawl_to_lightrag(
    url: str,
    max_depth: int = 2,
    limit: int = 10,
    description: str = "",
) -> dict:
    """Crawl a website and add all pages to LightRAG.

    Args:
        url: Starting URL
        max_depth: Crawl depth
        limit: Max pages
        description: Description for LightRAG

    Returns:
        dict with crawl and upload status
    """
    # First crawl the site
    crawl_result = crawl_website(url, max_depth=max_depth, limit=limit)

    if not crawl_result.get("success"):
        return crawl_result

    pages = crawl_result.get("pages", [])
    if not pages:
        return {"success": False, "error": "No pages crawled"}

    # Upload each page to LightRAG
    try:
        import asyncio
        from integrations.lightrag.client import upload_text

        uploaded = 0
        failed = 0

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            for page in pages:
                content = page.get("markdown", "")
                page_url = page.get("metadata", {}).get("url", "")
                title = page.get("metadata", {}).get("title", page_url)

                if not content:
                    continue

                full_content = f"# {title}\n\nSource: {page_url}\n\n{content}"

                result = loop.run_until_complete(
                    upload_text(full_content, description or f"Crawled from {url}")
                )

                if result.get("success"):
                    uploaded += 1
                else:
                    failed += 1
        finally:
            loop.close()

        return {
            "success": True,
            "url": url,
            "pages_crawled": len(pages),
            "pages_uploaded": uploaded,
            "pages_failed": failed,
        }

    except Exception as e:
        logger.error(f"Failed to upload crawl to LightRAG: {e}")
        return {"success": False, "error": str(e)}
