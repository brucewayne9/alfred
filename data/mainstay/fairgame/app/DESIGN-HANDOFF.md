# Fair Game — Cosmetic Redesign Hand-off (for Travis)

## TL;DR
Fair Game is a **static front-end**: 5 HTML pages + 1 CSS file. The brains (accounts,
payments, ticket transfers, pricing rules) live in a **separate backend API that Travis
never sees or touches**. So you can restyle **100% of the look** with zero risk to how it
works — *as long as you follow one rule:*

> **Change how things LOOK. Never change an element's `id`, never rename the CSS classes
> the code uses, and never edit anything inside `<script>` tags.**

That's the whole contract. Repaint every room; don't move the plumbing.

---

## What's in the package (`app/` folder)
| File | What it is |
|------|-----------|
| `index.html` | Shows list / landing — *"Real tickets. Fair prices. Straight from Rod."* |
| `show.html` | One show: primary seats + fan resale + buy bar + checkout |
| `sell.html` | Resell a ticket (face-value price-cap calculator) |
| `account.html` | Get verified / account (SMS + email verification) |
| `admin.html` | Tour console (Rod's team — ops/dashboard) |
| `fairgame.css` | **The entire skin** — design tokens + every style rule |

Only external dependency: **Google Fonts** (Anton + Sora) — swap for anything.
No build step, no framework. Double-click any `.html` to open it.

---

## ✅ FREE REIN — redesign anything here
- **All of `fairgame.css`.** Rewrite the whole token block at the top (colors, gradients,
  radii, shadows, fonts) and every rule below it. This is where most of the magic happens.
- **Layout, spacing, typography, imagery, logos, backgrounds, hero sections, card designs,
  button shapes, icons, animations, micro-interactions, hover states.**
- **Visible copy / labels** (headlines, button text) — fine to wordsmith.
- **Add new image/asset files** into the folder and reference them.

## 🚫 FROZEN — do not change (this is the "wiring")
1. **Element `id="..."` attributes.** The code finds elements by `id` to fill them with
   live data (seats, prices, your account). Rename one and that section goes blank.
2. **CSS class names the code generates** — `fg-btn`, `fg-btn-ghost`, `fg-btn-sm`,
   `fg-badge`, `priority`, `verified`, `dot`, `fg-empty`, `fg-toast`, `fg-hidden`, etc.
   **You can restyle these classes however you want — just don't rename or delete them.**
3. **Everything inside `<script> … </script>`.** Leave the JavaScript alone.
4. **Form input `name` / `id` attributes.**

### Frozen IDs by page (keep these exact)
- **index.html:** `navRight, showsGrid, showsMeta, statShows, toast`
- **show.html:** `navRight, showHead, inventory, listings, buybar, qMinus, qPlus, qVal,
  getBtn, payBtn, doneBtn, gateOverlay, gateClose, orderOverlay, orderBody, orderClose,
  retryClose, toast`
- **sell.html:** `navRight, gatePane, formPane, donePane, sellForm, show, viewShow, face,
  faceEcho, calc, amtBuyer, amtRod, amtYou, listBtn, listAnother, doneMsg, doneSummary,
  section, loadHint, toast`
- **account.html:** `navRight, registerForm, registerBtn, phone, email, smsTarget,
  smsVerifyBtn, smsResend, smsDevCode, emailTarget, emailVerifyBtn, emailDevCode, steps,
  heroTitle, heroLede, acctRows, doneBadges, signOut, startOver, loadHint, toast`
- **admin.html:** `kpis, showsBox, showsMeta, ordersBox, ordersMeta, reloadBtn, keyBtn,
  stamp, toast`

---

## 🗺️ Page map — "what is where"
- **index** = front door. Hero pitch + grid of tour shows. Sets the whole brand tone.
- **show** = the money page. Primary inventory at the top, capped **fan resale** below,
  sticky **buy bar**, checkout overlay. Trust + clarity matter most here.
- **sell** = a fan listing their extra ticket. The price-cap calculator shows the buyer
  price and the split — this is the anti-scalper story made visible.
- **account** = verification flow (becoming a "Verified Fan").
- **admin** = internal console for Rod's team. Can stay utilitarian.

---

## 🎯 Design direction (the brief)
Fair Game is **Rod Wave's fan-first, anti-scalper ticket marketplace** — the opposite of
a cold StubHub/Ticketmaster. Tone: **premium but warm, trustworthy, "straight from Rod."**
- Lean into the trust signals: **Verified Fan / Priority Fan** badges, the **face-value
  price cap**, "fair price" messaging. These are the brand.
- **Mobile-first** — most fans are on phones.
- Current palette (yours to change): black + fire-orange + gold + cream, condensed display
  type (Anton). Feel free to take it somewhere richer / more Rod.

## ↩️ How to hand it back
Just return the edited **`app/` folder** — the HTML files, `fairgame.css`, and any new
image assets. **Keep the file names the same.** We drop it straight back in.
