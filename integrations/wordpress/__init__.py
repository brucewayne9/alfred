"""WordPress integration - Multi-site management with SEO, Elementor, and full admin capabilities."""

from integrations.wordpress.client import (
    # Site management
    list_sites,
    get_site,
    test_connection,

    # Posts & Pages
    get_posts,
    get_post,
    create_post,
    update_post,
    delete_post,
    get_pages,
    get_page,
    create_page,
    update_page,

    # Media
    get_media,
    upload_media,

    # SEO (RankMath)
    get_seo_meta,
    update_seo_meta,
    get_seo_score,

    # Plugins
    get_plugins,
    activate_plugin,
    deactivate_plugin,
    update_plugin,

    # Themes
    get_themes,
    activate_theme,

    # Users
    get_users,

    # Site Health
    get_site_health,
)

__all__ = [
    "list_sites",
    "get_site",
    "test_connection",
    "get_posts",
    "get_post",
    "create_post",
    "update_post",
    "delete_post",
    "get_pages",
    "get_page",
    "create_page",
    "update_page",
    "get_media",
    "upload_media",
    "get_seo_meta",
    "update_seo_meta",
    "get_seo_score",
    "get_plugins",
    "activate_plugin",
    "deactivate_plugin",
    "update_plugin",
    "get_themes",
    "activate_theme",
    "get_users",
    "get_site_health",
]
