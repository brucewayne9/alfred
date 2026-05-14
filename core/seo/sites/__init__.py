# core/seo/sites/__init__.py
from core.seo.sites.registry import (
    register_site, get_site_by_slug, get_site_by_id, list_sites, update_site, deactivate_site,
)
__all__ = ["register_site", "get_site_by_slug", "get_site_by_id", "list_sites", "update_site", "deactivate_site"]
