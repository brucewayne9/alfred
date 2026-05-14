# Google OAuth setup for SEO ingest

One-time setup. Required before GSC and GA4 sync jobs can run.

## 1. Create OAuth client at Google Cloud Console

1. Open https://console.cloud.google.com/apis/credentials
2. Select (or create) a project: "Alfred SEO"
3. Configure OAuth consent screen:
   - User type: External (no domain restriction needed for read-only flows)
   - App name: "Alfred SEO"
   - User support email: mjohnson@groundrushinc.com
   - Scopes: `webmasters.readonly`, `analytics.readonly`
   - Test users: mjohnson@groundrushlabs.com (the GSC + GA4 owner)
4. Create OAuth client ID:
   - Application type: **Desktop app**
   - Name: "Alfred SEO Desktop"
5. Copy the Client ID + Client Secret into `config/.env`:
   ```
   SEO_GOOGLE_OAUTH_CLIENT_ID=...apps.googleusercontent.com
   SEO_GOOGLE_OAUTH_CLIENT_SECRET=...
   ```

## 2. Enable the APIs

In the same project: APIs & Services → Library → enable:
- Search Console API
- Google Analytics Data API
- PageSpeed Insights API (also generate a separate API key)

For PageSpeed Insights API key:
1. APIs & Services → Credentials → Create credentials → API key
2. Restrict key to "PageSpeed Insights API"
3. Add to `config/.env`:
   ```
   SEO_PSI_API_KEY=AIza...
   ```

## 3. One-time consent

From a desktop/terminal where you can open a browser (NOT the 105 server directly — use SSH tunnel + tmux on your Mac, OR run locally):

```
ssh -L 8080:localhost:8080 server-105
cd /home/aialfred/alfred
./venv/bin/python -m integrations.google_seo.oauth authorize
```

Browser opens → consent screen → click through. Token file lands at
`/home/aialfred/alfred/data/seo/google_oauth_token.json`.

## 4. Verify

```
./venv/bin/python -c "from integrations.google_seo import get_credentials; print(get_credentials().valid)"
```

Expected: `True`.
