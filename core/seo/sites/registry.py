# core/seo/sites/registry.py
"""CRUD for seo_sites. The orchestrator's source of truth on which sites exist."""
from __future__ import annotations

from typing import Optional

from sqlalchemy import select

from core.seo.db import SessionLocal
from core.seo.models import SeoSite


def register_site(
    slug: str,
    domain: str,
    display_name: str,
    wp_rest_url: str,
    *,
    wp_username: Optional[str] = None,
    wp_app_password: Optional[str] = None,  # caller should pass already-encrypted
    gsc_property: Optional[str] = None,
    ga4_property_id: Optional[str] = None,
    brand_profile_path: Optional[str] = None,
    business_type: str = "Organization",
) -> SeoSite:
    """Insert a new site. Raises ValueError on duplicate slug."""
    with SessionLocal() as s:
        existing = s.scalar(select(SeoSite).where(SeoSite.slug == slug))
        if existing:
            raise ValueError(f"site already registered: {slug}")
        site = SeoSite(
            slug=slug,
            domain=domain,
            display_name=display_name,
            wp_rest_url=wp_rest_url,
            wp_username=wp_username,
            wp_app_password=wp_app_password,
            gsc_property=gsc_property,
            ga4_property_id=ga4_property_id,
            brand_profile_path=brand_profile_path,
            business_type=business_type,
            status="active",
        )
        s.add(site)
        s.commit()
        s.refresh(site)
        return site


def get_site_by_slug(slug: str) -> Optional[SeoSite]:
    with SessionLocal() as s:
        return s.scalar(select(SeoSite).where(SeoSite.slug == slug))


def get_site_by_id(site_id: int) -> Optional[SeoSite]:
    with SessionLocal() as s:
        return s.get(SeoSite, site_id)


def list_sites(include_inactive: bool = False) -> list[SeoSite]:
    with SessionLocal() as s:
        q = select(SeoSite).order_by(SeoSite.id)
        if not include_inactive:
            q = q.where(SeoSite.status == "active")
        return list(s.scalars(q).all())


def update_site(slug: str, **patch) -> SeoSite:
    allowed = {
        "domain", "display_name", "wp_rest_url", "wp_username", "wp_app_password",
        "gsc_property", "ga4_property_id", "brand_profile_path", "business_type", "status",
    }
    bad = set(patch.keys()) - allowed
    if bad:
        raise ValueError(f"cannot update fields: {bad}")
    with SessionLocal() as s:
        site = s.scalar(select(SeoSite).where(SeoSite.slug == slug))
        if not site:
            raise ValueError(f"site not found: {slug}")
        for k, v in patch.items():
            setattr(site, k, v)
        s.commit()
        s.refresh(site)
        return site


def deactivate_site(slug: str) -> None:
    update_site(slug, status="inactive")
