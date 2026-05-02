"""
Pipeline orchestrator: takes a completed intake (photos + price) and runs
it through vision -> copywriter -> WooCommerce draft creation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

from core.jewelry import db, vision, copywriter, woocommerce

logger = logging.getLogger(__name__)


def process_intake(
    intake_id: int,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Run the full pipeline for one intake. Returns a result dict.

    progress_callback receives short status strings ("describing piece...",
    "uploading photos...") that the bot can relay back to the user.
    """
    def _say(msg: str) -> None:
        logger.info("intake %d: %s", intake_id, msg)
        if progress_callback:
            try:
                progress_callback(msg)
            except Exception:
                logger.exception("progress_callback raised; continuing")

    intake = db.get_intake(intake_id)
    if not intake:
        raise ValueError(f"intake {intake_id} not found")

    import json as _json
    photos = _json.loads(intake["photos_json"])
    if not photos:
        raise ValueError("intake has no photos")
    if not intake["price_cents"]:
        raise ValueError("intake has no price")

    photo_paths: List[Path] = [Path(p["path"]) for p in photos]

    # Step 1: vision describe (skip if a description is already on the intake — supports retries)
    if intake["description"]:
        description = intake["description"]
        _say("re-using existing description (resume)...")
    else:
        db.set_status(intake_id, "describing")
        _say("looking at the piece...")
        description = vision.describe_piece(photo_paths)
        db.set_description(intake_id, description)

    # Step 2: copywriter
    db.set_status(intake_id, "drafting")
    _say("writing the listing...")
    copy = copywriter.write_copy(description, intake["price_cents"])
    db.set_copy(
        intake_id,
        seo_title=copy["name"],
        sku=copy["sku"],
        short_description=copy["short_description"],
        long_description=copy["long_description"],
    )

    # Step 3: upload photos to WP media library
    _say("uploading photos...")
    attachment_ids: List[int] = []
    for p in photo_paths:
        try:
            attachment_ids.append(woocommerce.upload_image(p))
        except Exception as e:
            logger.exception("photo upload failed for %s", p)
            # Soft-fail on individual photo upload — keep going if at least one succeeded.
    if not attachment_ids:
        db.set_status(intake_id, "error", error="no photos uploaded")
        raise RuntimeError("all photo uploads failed")

    # Step 4: create the draft product
    _say("creating the draft...")
    post_id = woocommerce.create_draft_product(
        name=copy["name"],
        sku=copy["sku"],
        price_cents=intake["price_cents"],
        short_description=copy["short_description"],
        long_description=copy["long_description"],
        category_slug=copy["category"],
        tags=copy["tags"],
        image_attachment_ids=attachment_ids,
    )
    db.set_woocommerce_id(intake_id, post_id)
    db.set_status(intake_id, "done")

    return {
        "intake_id": intake_id,
        "post_id": post_id,
        "name": copy["name"],
        "sku": copy["sku"],
        "preview_url": woocommerce.preview_url(post_id),
        "edit_url": woocommerce.admin_edit_url(post_id),
    }
