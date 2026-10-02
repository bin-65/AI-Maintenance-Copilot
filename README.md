# AI Maintenance Copilot

### Multi-Agent Industrial Maintenance Automation Platform

> From maintenance complaint to an AI-assisted work order in minutes.

AI Maintenance Copilot is a hackathon MVP built entirely in **Python + Streamlit** with the **Groq API**.
An operator or maintenance engineer describes an equipment problem, for example:

> *"Pump P-102 has abnormal vibration and bearing temperature has increased to 92 °C."*

Seven specialised AI agents then work through the problem in sequence: triage, diagnosis, safety assessment,
maintenance planning and the engineering decision. The workflow then **automatically** creates a work order,
writes a PDF maintenance report, saves everything to SQLite and updates the dashboard KPIs.

> **Demo environment using synthetic equipment and maintenance data.** There is no SCADA, historian, CMMS or
> plant-control integration. This is a decision-support and workflow-automation prototype.

---

## Features

| Area | What it does |
|---|---|
| **Dashboard** | Landing section, 8 KPI cards, 6 Plotly charts, recent work orders |
| **New Maintenance Request** | Equipment form, sensor readings, CSV/Excel/PDF/TXT upload, demo cases, deterministic pre-checks, live agent execution |
| **Agent outputs** | Tabs for Triage, Diagnostic, Safety, Maintenance Plan, Senior Engineer, Work Order, Report & RCA |
| **Agent trace** | "View Agent Reasoning / Trace": a short explanation and the evidence for each step (no hidden chain-of-thought) |
| **Business automation** | Request → work order (`WO-2026-0001` style IDs) → spare-parts matching → cost and downtime estimate → PDF → DB → KPIs |
| **Notification** | Writes the email subject, recipients and body, with copy buttons (nothing is sent) |
| **Machines** | Fleet table, equipment details, overdue PM check, failure stats, **Start AI Analysis** button |
| **Maintenance History** | Search and filters, downtime and failure charts, previous recommendations |
| **Work Orders** | Filters by priority, equipment, status and category; click a row to see full details, update status and download the PDF |
| **Analytics** | Failure categories, top equipment, open vs closed, cost, resolution time, agent execution telemetry |
| **Demo Mode** | 3 one-click demo cases and a reset tool for the demo database |

---

## Architecture

```
ai-maintenance-copilot/
├── app.py                       # Streamlit UI (all pages)
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── .streamlit/
│   ├── config.toml              # dark navy industrial theme
│   └── secrets.toml.example     # template only - never commit real secrets
├── agents/
│   ├── llm.py                   # shared GroqClient: generate(), generate_json(), retries, cache, errors
│   ├── triage_agent.py
│   ├── diagnostic_agent.py
│   ├── safety_agent.py
│   ├── maintenance_agent.py
│   ├── senior_engineer_agent.py
│   ├── work_order_agent.py
│   └── report_agent.py
├── workflow/
│   └── maintenance_workflow.py  # orchestrator + notification builder
├── database/
│   └── database.py              # SQLite schema, seeding, queries, KPIs
├── utils/
│   ├── calculations.py          # deterministic engineering calculations (demo thresholds)
│   ├── pdf_generator.py         # ReportLab PDF report
│   └── sample_data.py           # demo cases + synthetic seed work orders
└── data/
    ├── machines.csv
    ├── maintenance_history.csv
    └── spare_parts.csv
```

### Agent architecture

```
Complaint + readings
   │
   ▼
Input Validation ── Python: vibration zone, temperature/pressure deviation, overdue PM, severity floor
   │                SQLite: create maintenance request, load history + compatible spare parts
   ▼
1. Triage Agent ─────────── category, severity, priority, department, shutdown flag
   │  (guard-rail: severity can never be below the deterministic floor; priority derived from severity)
   ▼
2. Diagnostic Agent ─────── probable causes, hypothesis, confidence, tagged evidence, inspections
   ▼
3. Safety Agent ─────────── hazards, LOTO isolation, PPE, permits, emergency action, operating constraint
   ▼
4. Maintenance Planner ──── actions, sequence, tools, spares, manpower, hours
   │  (Python: match spares to DB stock/cost/lead time, estimate downtime and cost)
   ▼
5. Senior Engineer Agent ── reconciles conflicts (safety first), decision, urgency, conditions
   ▼
6. Work Order Agent ─────── title, job steps, safety requirements, acceptance criteria
   │  (SQLite autoincrement → WO-YYYY-NNNN)
   ▼
7. Report Agent ─────────── executive & management summary, RCA template, follow-up
   ▼
Database & Dashboard update ── request status, machine status, audit log, PDF available
```

* **One LLM call per agent**: 7 calls per workflow.
* Every agent has its **own system prompt** and a **Pydantic output model**.
* Prompts stress evidence-based reasoning, honest uncertainty, no invented data, a clear line between facts and
  assumptions, and safety first.
* Simple calculations are done **in Python, not by the LLM**.
* Agents only see the data they need: the 6 most recent history records for that equipment, the compatible spare
  parts, and compact JSON.

### LLM robustness (`agents/llm.py`)

* A single shared `GroqClient`. The app creates it once with `st.cache_resource`.
* Request timeout. The SDK retries timeouts, connection errors, 429 and 5xx.
* JSON mode, then JSON extraction (code fences, surrounding prose, trailing commas), then Pydantic validation.
  If that fails, it retries **once** with the error fed back, then raises a **controlled error**.
* Lenient field coercion, so `"8 hours"` becomes `8.0` and a string becomes a list.
* An in-memory response cache. Identical demo requests make no new LLM calls.
* Friendly messages for a missing or invalid key, an unknown model, rate limits and network problems.
  The key is never shown, logged or stored.

---

## Technology stack

Python 3.11+ · Streamlit · Groq Python SDK · Pydantic v2 · Pandas · Plotly · SQLite (built in) · ReportLab ·
python-dotenv · openpyxl (Excel uploads) · pypdf (PDF upload text extraction)

---

## Installation (local)

```bash
git clone <your-repo-url>
cd ai-maintenance-copilot
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | yes (for AI workflow) | Your Groq API key |
| `GROQ_MODEL` | no | Groq chat model. Default: `openai/gpt-oss-120b` |
| `COPILOT_DB_PATH` | no | Custom SQLite file path. Default: `data/copilot.db` |

Settings are read from **Streamlit secrets** first, then from **environment / `.env`**.

### Groq setup

1. Create a free account at <https://console.groq.com>.
2. Go to **API Keys** and create a key.
3. For local runs, copy `.env.example` to `.env` and paste the key:
   ```
   GROQ_API_KEY="gsk_..."
   ```
   You can use `.streamlit/secrets.toml` instead. Copy it from `secrets.toml.example`.
4. Optional: set `GROQ_MODEL` to any chat model listed under **Models** in the Groq console. Groq retires models
   from time to time. If you see "model not found" or "decommissioned", change `GROQ_MODEL` and you don't need to
   touch any code. On the free tier, if you hit rate limits, try a model with higher tokens-per-minute limits
   (for example `openai/gpt-oss-20b` or `qwen/qwen3.8-27b`, if listed). For `openai/gpt-oss-*` reasoning
   models the client automatically uses low reasoning effort and extra output-token headroom.

### Run locally

```bash
streamlit run app.py
```

The SQLite database is created and filled with synthetic data on the first run. No setup is needed.
Every page except the AI workflow also works without an API key.

---

## Deploy to Streamlit Community Cloud

1. **Create a GitHub repository** (public or private).
2. **Upload the project.** Push the contents of `ai-maintenance-copilot/` to the repository root.
   `.gitignore` already keeps out `.env`, `.streamlit/secrets.toml` and the database file.
3. Go to <https://share.streamlit.io>, click **Create app**, choose **Deploy a public app from GitHub**,
   pick the repository and branch, and set **Main file path** to `app.py`.
   Under **Advanced settings**, choose Python 3.11 or newer.
4. **Add the secret.** In **Advanced settings → Secrets** (or after deployment under **App settings → Secrets**), paste:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   # GROQ_MODEL = "openai/gpt-oss-120b"
   ```
5. Click **Deploy** and wait for the app to start.

No separate backend, database server, Docker or Node.js is needed.
Note that the Community Cloud file system is temporary, so the demo database resets when the app restarts.

---

## Demo instructions (3–5 minute pitch)

1. Open the **Dashboard**. Point out the title, the five capability cards (Generative AI, Agentic AI, Multi-Agent,
   AI Workflow, Business Automation) and the KPIs.
2. Click **Run demo: Pump P-102 high vibration**, or go to **New Maintenance Request → Load Demo Case**.
3. Open **Platform pre-checks** to show that the Python calculations put the severity floor at **High**.
4. Click **RUN AI MAINTENANCE WORKFLOW**. Each agent appears as *Running…* and then
   *🟢 Completed · x.xs*, with a short result.
5. Go through the tabs:
   **Triage** (severity and priority), **Diagnostic** (probable causes and tagged evidence), **Safety** (LOTO, PPE,
   permits), **Maintenance Plan** (tools, spares matched against stock, cost and downtime), **Senior Engineer**
   (decision and resolved conflicts), **Work Order** (created automatically).
6. Open **View Agent Reasoning / Trace**.
7. Click **Download PDF Maintenance Report** and **Generate Notification**.
8. Click **View updated Dashboard**. The new work order appears and the open-request and active-WO KPIs have gone up.
9. Optional: run Case 2 (compressor) or Case 3 (cooling fan) to show different agent outputs.
   Then show **Work Orders** (click the new WO) and **Analytics** (agent execution telemetry).

### Example workflow output (P-102 demo, abbreviated)

| Step | Example result |
|---|---|
| Input Validation | REQ-2026-0034 created · 5 readings · 5 history records · severity floor High |
| Triage | Mechanical · High · P2 · Mechanical Maintenance |
| Diagnostic | Likely DE bearing degradation (requires verification), confidence Medium |
| Safety | Risk High, rotating parts and hot surfaces, Permit to Work, electrical LOTO |
| Planner | BRG-6208 ×2 (in stock), ISO VG 68 oil, 8 man-h, est. downtime 6 h |
| Senior Engineer | Controlled shutdown for planned inspection, within 24 hours |
| Work Order | WO-2026-0034, P2, "Inspect and replace P-102 DE bearing" |
| Report | Executive summary, RCA template, follow-up, PDF |

*(Real wording varies with the LLM.)*

---

## Hackathon skills demonstrated

* **Generative AI**: engineering summaries, work-order text, RCA template, report narrative.
* **Agentic AI**: agents reason over evidence and produce decisions and actions.
* **Multi-Agent System**: seven specialised agents with separate prompts and structured contracts.
* **AI Workflow**: an orchestrator chains the agents, Python calculations, guard-rails and DB look-ups.
* **Business Process Automation**: request, work order, spare parts, cost, report, database and dashboard,
  with no manual steps.

---

## Project limitations

* Synthetic equipment and data. There are no real-time sensor feeds and no SCADA or CMMS integration.
* Thresholds in `utils/calculations.py` are **demo/configurable values**, not OEM limits or standards.
* LLM output can be incomplete or wrong. Every recommendation needs review by qualified personnel.
* SQLite on Streamlit Community Cloud is temporary. There is no authentication or role-based access.
* Notifications are drafts only. No email is sent.

## Future improvements

* CMMS integration (SAP PM, IBM Maximo) and historian/OPC-UA data ingestion.
* Retrieval over OEM manuals and site procedures (RAG) with citations.
* Human-in-the-loop approval step before a work order is released.
* Optional email/Teams/Slack sending, user roles, and a persistent cloud database.
* Vibration spectrum and trend analytics, and failure prediction from history.

---

## Safety disclaimer

AI Maintenance Copilot is a decision-support and workflow automation prototype. AI-generated recommendations are not
a substitute for qualified engineering judgment, manufacturer instructions, plant operating procedures, permits,
isolation/LOTO procedures, or HSE requirements. The application never recommends bypassing safety systems. For
potentially dangerous conditions, follow site emergency procedures and involve qualified personnel.
