"""Forge tenancy scope — the single source of truth for 'what can this viewer
see and write'. Built once at the API boundary from the authenticated identity
and threaded into every data-layer call.

A member is pinned to their own org. A super_admin defaults to view_all (every
org merged) but may focus one org via the dashboard switcher. requested_org is
ONLY honored for super_admin — a member can never escape their org by passing it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scope:
    org: str                # the org to stamp writes with / focus reads on
    role: str               # member | org_admin | super_admin
    view_all: bool          # super_admin seeing every org at once

    @property
    def is_super(self) -> bool:
        return self.role == "super_admin"

    def can_write_org(self, org: str) -> bool:
        """May this viewer create/modify rows in `org`?"""
        return self.is_super or org == self.org

    def can_read_org(self, org: str) -> bool:
        return self.view_all or self.is_super or org == self.org


def scope_from_user(user: dict, requested_org: str | None = None) -> Scope:
    role = (user or {}).get("role", "member")
    org = (user or {}).get("org", "mainstay")
    if role == "super_admin":
        focus = (requested_org or "").strip().lower()
        if focus and focus != "*" and focus != "all":
            return Scope(org=focus, role=role, view_all=False)
        return Scope(org="*", role=role, view_all=True)
    # member / org_admin — pinned to their own org; requested_org ignored.
    return Scope(org=org, role=role, view_all=False)
