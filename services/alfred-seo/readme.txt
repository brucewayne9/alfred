=== Alfred SEO ===
Contributors: alfredlabs
Tags: seo, schema, sitemap, open-graph
Requires at least: 6.0
Tested up to: 6.7
Stable tag: 0.1.0
Requires PHP: 8.0
License: Proprietary

SEO foundation controlled remotely by Alfred orchestrator on 105.

== Description ==

Renders schema (JSON-LD), Open Graph, Twitter Cards, meta descriptions, sitemaps,
image alt text, and internal linking. All decisions come from the Alfred
orchestrator via WP REST API. Plugin has sane local fallbacks for when the
orchestrator is unavailable — the site never breaks.

== Local Testing ==

Run the WP test suite installer once:

    bash bin/install-wp-tests.sh wordpress_test root '' localhost latest

Then run tests:

    cd services/alfred-seo && phpunit

== Changelog ==

= 0.1.0 =
* Initial release. Phase 1 foundation.
