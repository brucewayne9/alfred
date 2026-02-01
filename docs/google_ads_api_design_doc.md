# Alfred AI Assistant - Google Ads API Integration
## Design Documentation for API Access Application

---

## 1. Application Overview

**Application Name:** Alfred AI Assistant
**Developer/Company:** GroundRush Inc | GroundRush Labs
**Contact:** Mike Johnson
**Application Type:** Internal Business Tool (Self-use)
**Website:** Private/Internal deployment

### Description

Alfred is a private, self-hosted AI assistant designed for personal and business management. The Google Ads API integration will enable the business owner to interact with their Google Ads account(s) using natural language queries through a conversational interface.

**Primary Use Case:** Allow the business owner to monitor campaign performance, receive insights, and get optimization recommendations through voice or text conversations with Alfred.

---

## 2. Intended API Usage

### 2.1 Features to be Implemented

| Feature | API Resources Used | Purpose |
|---------|-------------------|---------|
| Campaign Performance Reports | `GoogleAdsService.SearchStream` | Retrieve metrics (impressions, clicks, conversions, cost, ROAS) |
| Quality Score Analysis | `GoogleAdsService.Search` | Monitor keyword quality scores and components |
| Budget Monitoring | `Campaign`, `Budget` resources | Track spend pacing and budget utilization |
| Recommendations | `Recommendation` resource | Surface Google's optimization suggestions |
| Search Terms Report | `SearchTermView` | Analyze actual search queries triggering ads |
| Conversion Tracking Status | `ConversionAction` | Verify conversion tracking health |
| Account Hierarchy | `CustomerClient` | List and navigate managed accounts |

### 2.2 Operations

- **READ ONLY**: This integration will only perform read operations
- No campaigns, ads, or settings will be created, modified, or deleted via the API
- All optimization actions will be recommendations that the user implements manually in the Google Ads UI

### 2.3 API Methods Used

```
GoogleAdsService.Search
GoogleAdsService.SearchStream
CustomerService.ListAccessibleCustomers
```

---

## 3. User Flow

### 3.1 Authentication Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. User initiates Google Ads connection in Alfred          │
│                          ↓                                  │
│  2. Alfred redirects to Google OAuth 2.0 consent screen     │
│                          ↓                                  │
│  3. User authenticates with Google account                  │
│                          ↓                                  │
│  4. User grants read-only access to Google Ads data         │
│                          ↓                                  │
│  5. OAuth tokens stored securely (encrypted at rest)        │
│                          ↓                                  │
│  6. Alfred confirms connection and lists accessible accounts│
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Typical User Interaction

```
User: "How are my Google Ads campaigns performing this week?"

Alfred: [Queries GoogleAdsService.SearchStream for campaign metrics]

Alfred: "Here's your Google Ads performance for this week:

         Campaign: Summer Sale 2024
         • Impressions: 45,230
         • Clicks: 1,847 (4.08% CTR)
         • Conversions: 89
         • Cost: $923.45
         • ROAS: 3.2x

         Campaign: Brand Awareness
         • Impressions: 128,400
         • Clicks: 3,201 (2.49% CTR)
         • Conversions: 45
         • Cost: $1,245.00
         • ROAS: 1.8x

         Recommendation: Your 'Summer Sale' campaign is
         outperforming. Consider increasing its budget."
```

---

## 4. Technical Architecture

### 4.1 System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Alfred AI Assistant                       │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │   Web UI    │    │  Voice UI   │    │   Mobile App    │  │
│  │  (Browser)  │    │ (Hey Alfred)│    │  (iOS/Android)  │  │
│  └──────┬──────┘    └──────┬──────┘    └────────┬────────┘  │
│         │                  │                     │           │
│         └──────────────────┼─────────────────────┘           │
│                            ↓                                 │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Core API Layer                        ││
│  │         (FastAPI + Authentication + Rate Limiting)       ││
│  └─────────────────────────┬───────────────────────────────┘│
│                            ↓                                 │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                   AI Brain / Router                      ││
│  │              (Natural Language Processing)               ││
│  └─────────────────────────┬───────────────────────────────┘│
│                            ↓                                 │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                 Integration Layer                        ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ ││
│  │  │  Gmail   │ │ Calendar │ │   CRM    │ │ Google Ads │ ││
│  │  └──────────┘ └──────────┘ └──────────┘ └─────┬──────┘ ││
│  └───────────────────────────────────────────────┼─────────┘│
└──────────────────────────────────────────────────┼──────────┘
                                                   ↓
                               ┌───────────────────────────────┐
                               │      Google Ads API           │
                               │   (OAuth 2.0 + REST/gRPC)     │
                               └───────────────────────────────┘
```

### 4.2 Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.11 + FastAPI |
| Google Ads Client | google-ads Python library |
| Authentication | OAuth 2.0 with refresh tokens |
| Token Storage | Encrypted file storage (AES-256) |
| Hosting | Self-hosted (private server) |

---

## 5. Data Handling & Privacy

### 5.1 Data Access

- **Scope**: Read-only access to Google Ads account data
- **Accounts**: Only accounts the authenticated user has access to
- **Storage**: No Google Ads data is permanently stored; queries are made in real-time

### 5.2 Data Security

| Security Measure | Implementation |
|------------------|----------------|
| Token Encryption | AES-256 encryption at rest |
| Transport Security | HTTPS/TLS 1.3 only |
| Access Control | Single-user authentication required |
| Token Refresh | Automatic refresh token rotation |
| Audit Logging | All API calls logged with timestamps |

### 5.3 Data Retention

- **API Responses**: Not stored permanently; used only for immediate response generation
- **OAuth Tokens**: Stored encrypted; deleted upon user disconnection
- **Logs**: API call logs retained for 30 days for debugging purposes

---

## 6. Compliance

### 6.1 Google Ads API Terms of Service

This application complies with:
- ✅ Google Ads API Terms and Conditions
- ✅ Required Minimum Functionality (read-only reporting)
- ✅ Prohibited Practices Policy (no automated bidding/campaign changes)
- ✅ Data Protection requirements

### 6.2 Usage Restrictions

- **No Automated Changes**: All campaign modifications require manual user action in Google Ads UI
- **No Data Resale**: Data is not shared, sold, or exposed to third parties
- **No Bulk Operations**: Tool is designed for single-account owner use
- **Rate Limiting**: Respects Google Ads API rate limits

### 6.3 User Consent

- Users explicitly grant OAuth permissions before any data access
- Clear disclosure of what data will be accessed
- Easy revocation through Google Account settings or within Alfred

---

## 7. Access Level Requested

**Requested Access Level:** Basic Access

**Justification:**
- Single business owner use case
- Read-only operations only
- Low API call volume (estimated <1,000 requests/day)
- No management of client accounts (not an agency tool)

---

## 8. Sample API Queries

### 8.1 Campaign Performance Query

```sql
SELECT
  campaign.id,
  campaign.name,
  campaign.status,
  metrics.impressions,
  metrics.clicks,
  metrics.ctr,
  metrics.conversions,
  metrics.cost_micros,
  metrics.conversions_value
FROM campaign
WHERE segments.date DURING LAST_7_DAYS
  AND campaign.status = 'ENABLED'
ORDER BY metrics.cost_micros DESC
```

### 8.2 Quality Score Query

```sql
SELECT
  ad_group_criterion.keyword.text,
  ad_group_criterion.quality_info.quality_score,
  ad_group_criterion.quality_info.creative_quality_score,
  ad_group_criterion.quality_info.post_click_quality_score,
  ad_group_criterion.quality_info.search_predicted_ctr
FROM keyword_view
WHERE ad_group_criterion.status = 'ENABLED'
ORDER BY ad_group_criterion.quality_info.quality_score ASC
LIMIT 50
```

### 8.3 Recommendations Query

```sql
SELECT
  recommendation.type,
  recommendation.impact.base_metrics.impressions,
  recommendation.impact.potential_metrics.impressions,
  recommendation.campaign_budget_recommendation,
  recommendation.keyword_recommendation
FROM recommendation
WHERE recommendation.type IN (
  'CAMPAIGN_BUDGET',
  'KEYWORD',
  'TEXT_AD',
  'TARGET_CPA_OPT_IN'
)
```

---

## 9. Support & Contact

**Developer Contact:**
Name: Mike Johnson
Company: GroundRush Inc | GroundRush Labs
Email: [Your email]
Phone: [Your phone]

**Technical Support:**
Application issues are handled internally by the development team.

---

## 10. Screenshots / UI Mockups

*Note: Alfred uses a conversational interface. Below is an example interaction.*

```
┌─────────────────────────────────────────────────────────────┐
│  Alfred AI Assistant                              [≡] [×]   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  You: Show me my Google Ads performance for this month      │
│                                                             │
│  ┌────────────────────────────────────────────────────────┐│
│  │ Alfred:                                                 ││
│  │                                                         ││
│  │ Here's your Google Ads performance for January 2026:    ││
│  │                                                         ││
│  │ 📊 Overall Performance                                  ││
│  │ ────────────────────────────────────                   ││
│  │ Total Spend: $4,523.87                                 ││
│  │ Impressions: 234,567                                   ││
│  │ Clicks: 8,934 (3.81% CTR)                             ││
│  │ Conversions: 267                                       ││
│  │ ROAS: 2.8x                                            ││
│  │                                                         ││
│  │ 🏆 Top Performing Campaign                              ││
│  │ "Spring Collection" - 4.2x ROAS                        ││
│  │                                                         ││
│  │ ⚠️ Needs Attention                                      ││
│  │ "Brand Terms" - Quality Score dropped to 6/10          ││
│  │                                                         ││
│  │ 💡 Google Recommends                                    ││
│  │ • Increase "Spring Collection" budget by 20%           ││
│  │ • Add 5 suggested keywords to "Summer Prep"            ││
│  └────────────────────────────────────────────────────────┘│
│                                                             │
│  [────────────────────────────────────] [🎤] [Send]        │
└─────────────────────────────────────────────────────────────┘
```

---

**Document Version:** 1.0
**Last Updated:** February 2026
**Status:** Submitted for Google Ads API Access Review
