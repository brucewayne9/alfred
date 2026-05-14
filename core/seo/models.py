# core/seo/models.py
"""SQLAlchemy table definitions matching migrations/2026-05-14-seo-initial-schema.sql."""
from __future__ import annotations

import datetime as dt
from sqlalchemy import (
    BigInteger, Column, Date, DateTime, ForeignKey, Integer, Numeric,
    String, Text, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from core.seo.db import Base


class SeoSite(Base):
    __tablename__ = "seo_sites"
    id                 = Column(Integer, primary_key=True)
    slug               = Column(String(64), unique=True, nullable=False)
    domain             = Column(String(255), nullable=False)
    display_name       = Column(String(255), nullable=False)
    wp_rest_url        = Column(String(255), nullable=False)
    wp_username        = Column(String(255))
    wp_app_password    = Column(Text)  # encrypted at app layer
    gsc_property       = Column(String(255))
    ga4_property_id    = Column(String(64))
    brand_profile_path = Column(String(255))
    business_type      = Column(String(32), default="Organization")
    status             = Column(String(32), default="active")
    created_at         = Column(DateTime(timezone=True), server_default=func.now())
    updated_at         = Column(DateTime(timezone=True), server_default=func.now())

    queries     = relationship("SeoQuery", back_populates="site", cascade="all, delete-orphan")
    pages       = relationship("SeoPage", back_populates="site", cascade="all, delete-orphan")
    briefs      = relationship("SeoBrief", back_populates="site", cascade="all, delete-orphan")
    pending     = relationship("SeoPending", back_populates="site", cascade="all, delete-orphan")
    decided     = relationship("SeoDecided", back_populates="site", cascade="all, delete-orphan")
    backlinks   = relationship("SeoBacklink", back_populates="site", cascade="all, delete-orphan")
    haro_opps   = relationship("SeoHaroOpp", back_populates="site", cascade="all, delete-orphan")
    rankings    = relationship("SeoRankingDaily", back_populates="site", cascade="all, delete-orphan")


class SeoQuery(Base):
    __tablename__ = "seo_queries"
    id          = Column(BigInteger, primary_key=True)
    site_id     = Column(Integer, ForeignKey("seo_sites.id", ondelete="CASCADE"), nullable=False)
    query       = Column(Text, nullable=False)
    position    = Column(Numeric(6, 2))
    impressions = Column(Integer, default=0)
    clicks      = Column(Integer, default=0)
    ctr         = Column(Numeric(6, 4))
    captured_at = Column(Date, nullable=False)
    __table_args__ = (UniqueConstraint("site_id", "query", "captured_at"),)
    site = relationship("SeoSite", back_populates="queries")


class SeoPage(Base):
    __tablename__ = "seo_pages"
    id                = Column(BigInteger, primary_key=True)
    site_id           = Column(Integer, ForeignKey("seo_sites.id", ondelete="CASCADE"), nullable=False)
    url               = Column(Text, nullable=False)
    page_type         = Column(String(64))
    indexed_at        = Column(DateTime(timezone=True))
    last_audit_at     = Column(DateTime(timezone=True))
    schema_status     = Column(String(32))
    meta_status       = Column(String(32))
    cwv_lcp_ms        = Column(Integer)
    cwv_cls           = Column(Numeric(6, 3))
    cwv_inp_ms        = Column(Integer)
    organic_sessions  = Column(Integer, default=0)
    conversions       = Column(Integer, default=0)
    last_seen_at      = Column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (UniqueConstraint("site_id", "url"),)
    site = relationship("SeoSite", back_populates="pages")


class SeoBrief(Base):
    __tablename__ = "seo_briefs"
    id              = Column(BigInteger, primary_key=True)
    site_id         = Column(Integer, ForeignKey("seo_sites.id", ondelete="CASCADE"), nullable=False)
    topic           = Column(Text, nullable=False)
    content_type    = Column(String(32))
    target_keywords = Column(JSONB)
    audience        = Column(Text)
    status          = Column(String(32), default="queued")
    brief_payload   = Column(JSONB)
    source_signal   = Column(JSONB)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    site = relationship("SeoSite", back_populates="briefs")


class SeoPending(Base):
    __tablename__ = "seo_pending"
    id            = Column(BigInteger, primary_key=True)
    site_id       = Column(Integer, ForeignKey("seo_sites.id", ondelete="CASCADE"), nullable=False)
    brief_id      = Column(BigInteger, ForeignKey("seo_briefs.id", ondelete="SET NULL"))
    content_type  = Column(String(32))
    title         = Column(Text)
    body_payload  = Column(JSONB)
    source_signal = Column(JSONB)
    status        = Column(String(32), default="pending")
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    site = relationship("SeoSite", back_populates="pending")


class SeoDecided(Base):
    __tablename__ = "seo_decided"
    id           = Column(BigInteger, primary_key=True)
    site_id      = Column(Integer, ForeignKey("seo_sites.id", ondelete="CASCADE"), nullable=False)
    brief_id     = Column(BigInteger, ForeignKey("seo_briefs.id", ondelete="SET NULL"))
    content_type = Column(String(32))
    title        = Column(Text)
    body_payload = Column(JSONB)
    decided_at   = Column(DateTime(timezone=True), server_default=func.now())
    decided_by   = Column(String(64))
    outcome      = Column(String(32))
    wp_post_id   = Column(BigInteger)
    error        = Column(Text)
    site = relationship("SeoSite", back_populates="decided")


class SeoBacklink(Base):
    __tablename__ = "seo_backlinks"
    id          = Column(BigInteger, primary_key=True)
    site_id     = Column(Integer, ForeignKey("seo_sites.id", ondelete="CASCADE"), nullable=False)
    source_url  = Column(Text, nullable=False)
    target_url  = Column(Text)
    anchor_text = Column(Text)
    first_seen  = Column(DateTime(timezone=True), server_default=func.now())
    last_seen   = Column(DateTime(timezone=True), server_default=func.now())
    lost_at     = Column(DateTime(timezone=True))
    __table_args__ = (UniqueConstraint("site_id", "source_url", "target_url"),)
    site = relationship("SeoSite", back_populates="backlinks")


class SeoHaroOpp(Base):
    __tablename__ = "seo_haro_opps"
    id                  = Column(BigInteger, primary_key=True)
    site_id             = Column(Integer, ForeignKey("seo_sites.id", ondelete="CASCADE"), nullable=False)
    source_email_id     = Column(String(255))
    query_text          = Column(Text)
    deadline            = Column(DateTime(timezone=True))
    draft_pitch_payload = Column(JSONB)
    status              = Column(String(32), default="pending")
    response_sent_at    = Column(DateTime(timezone=True))
    created_at          = Column(DateTime(timezone=True), server_default=func.now())
    site = relationship("SeoSite", back_populates="haro_opps")


class SeoRankingDaily(Base):
    __tablename__ = "seo_rankings_daily"
    id          = Column(BigInteger, primary_key=True)
    site_id     = Column(Integer, ForeignKey("seo_sites.id", ondelete="CASCADE"), nullable=False)
    query       = Column(Text, nullable=False)
    position    = Column(Numeric(6, 2))
    captured_at = Column(Date, nullable=False)
    __table_args__ = (UniqueConstraint("site_id", "query", "captured_at"),)
    site = relationship("SeoSite", back_populates="rankings")
