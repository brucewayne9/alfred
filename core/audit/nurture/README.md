# AI Savings Audit — 5-Email Nurture Sequence

Drafted templates for the post-audit nurture flow. **Drafts only** — review and tune before activating in Brevo.

## Files

| Day | File                      | Subject (default)                               | Send trigger |
|-----|---------------------------|-------------------------------------------------|--------------|
| 0   | `01_day_0_pdf_resend.html`| Re-sending your AI Savings Audit PDF            | 2 hrs after submit if no PDF email open |
| 2   | `02_day_2_case_study.html`| How a 6-person agency captured $84k/yr with AI  | 2 days after submit |
| 5   | `03_day_5_tactical_tip.html`| The 7-minute audit you can run yourself         | 5 days after submit |
| 9   | `04_day_9_book_call.html` | Last 3 audit-call slots this week               | 9 days after submit, only if not yet booked |
| 14  | `05_day_14_reengage.html` | Quick check — still leaking $X/mo?              | 14 days after submit, only if no engagement |

## Brevo workflow setup (one-time, in Brevo dashboard)

1. **Create the list** — Contacts → Lists → New List → "AI Audit Leads"
2. **Grab the integer list ID** from the URL or list properties → paste into `config/.env`
   as `BREVO_AI_AUDIT_LIST_ID=<int>` → restart `alfred.service`
3. **Define custom contact attributes** (Settings → Attributes → Add attribute):
   - `INDUSTRY` (text)
   - `AI_AUDIT_ANNUAL` (number)
   - `AI_AUDIT_MONTHLY` (number)
   - `AI_AUDIT_FIT` (text — "low" / "mid" / "high")
   - `AI_AUDIT_PAYBACK_MO` (number)
   - `UTM_SOURCE` (text)
   - `UTM_CAMPAIGN` (text)
   - `BOOKED_CALL` (boolean — set to true by the Phase 2 #13 Gmail-booking sync)

   Once these exist in Brevo, re-enable them in `core/audit/lead_router.py:_brevo_upsert`
   (currently commented out for safety while running with default attrs only).

4. **Build the automation workflow** (Automations → New workflow → "AI Audit Nurture"):
   - **Trigger:** Contact added to "AI Audit Leads" list
   - Step 1: Wait 2 hours → if `LAST_OPEN` is null → send `01_day_0_pdf_resend.html`
   - Step 2: Wait until +2 days → send `02_day_2_case_study.html`
   - Step 3: Wait until +5 days → send `03_day_5_tactical_tip.html`
   - Step 4: Wait until +9 days → if `BOOKED_CALL` is not true → send `04_day_9_book_call.html`
   - Step 5: Wait until +14 days → if no email click in last 7 days → send `05_day_14_reengage.html`
   - End

5. **Activate** → "Set live" in the Brevo workflow dashboard.

## Personalization

All templates use Brevo's `{{ contact.FIRSTNAME }}`, `{{ contact.COMPANY }}`,
`{{ contact.AI_AUDIT_ANNUAL }}`, etc. The Brevo template engine replaces these
on send. **Test-send each template to your own address before activating** —
Brevo will substitute the values from your contact card.

## Tone

- Keep it conversational. Mike-the-operator voice, not corporate marketing.
- Mention the prospect's actual savings figure when sensible (`AI_AUDIT_ANNUAL`).
- Single CTA per email — book a call. The PDF resend is the only exception.
- No "limited time only" gimmicks. The Day 9 email mentions slot scarcity
  ("only 3 audit-call slots left this week") but that should be true when sent
  — Mike checks his Google Calendar before activating.

## Sender / reply-to

All emails should send from `alfred@groundrushinc.com` (Mike's deliverable
inbox), reply-to `mjohnson@groundrushinc.com`. Configure in Brevo workflow
sender settings.
