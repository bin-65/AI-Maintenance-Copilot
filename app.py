"""AI Maintenance Copilot - Multi-Agent Industrial Maintenance Automation Platform.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud (add GROQ_API_KEY under Secrets)
"""
from __future__ import annotations

import io
import sqlite3
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

st.set_page_config(page_title="AI Maintenance Copilot", page_icon="🛠️", layout="wide", initial_sidebar_state="expanded")

from agents.llm import GroqClient  # noqa: E402
from agents.safety_agent import SAFETY_DISCLAIMER  # noqa: E402
from database import database as db  # noqa: E402
from utils import calculations as calc  # noqa: E402
from utils.pdf_generator import DISCLAIMER, build_pdf  # noqa: E402
from utils.sample_data import DEMO_CASES, DEMO_NOTICE  # noqa: E402
from workflow.maintenance_workflow import STEPS, MaintenanceRequest, MaintenanceWorkflow, build_notification  # noqa: E402

PAGES = ["Dashboard", "New Maintenance Request", "Machines", "Maintenance History", "Work Orders", "Analytics", "About"]
OPERATING_CONDITIONS = ["Normal operation", "High load", "Low load / turndown", "Start-up", "Shutdown", "Standby",
                        "Abnormal / upset condition"]
WO_STATUSES = ["Open", "In Progress", "Awaiting Parts", "Completed", "Cancelled"]
SEV_COLORS = {"Low": "#22C55E", "Medium": "#3B82F6", "High": "#EAB308", "Critical": "#EF4444"}
PALETTE = ["#3B82F6", "#22C55E", "#14B8A6", "#60A5FA", "#A3E635", "#38BDF8", "#818CF8", "#94A3B8"]
READING_FIELDS = [  # (session key, label, request field, min, max, step)
    ("f_load", "Load %", "load_pct", 0.0, 150.0, 1.0),
    ("f_runtime", "Runtime hours", "runtime_hours", 0.0, 1_000_000.0, 100.0),
    ("f_temp", "Temperature (°C)", "temperature_c", -50.0, 500.0, 1.0),
    ("f_pressure", "Pressure (bar)", "pressure_bar", 0.0, 500.0, 0.1),
    ("f_vibration", "Vibration (mm/s RMS)", "vibration_mm_s", 0.0, 100.0, 0.1),
    ("f_oil_p", "Oil pressure (bar)", "oil_pressure_bar", 0.0, 100.0, 0.1),
    ("f_oil_t", "Oil temperature (°C)", "oil_temperature_c", -20.0, 300.0, 1.0),
]

CSS = """
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1400px;}
[data-testid="stSidebar"] {background-color: #08101F; border-right: 1px solid #1B2A45;}
.brand {font-weight: 800; letter-spacing: .14em; font-size: .95rem; color: #E6ECF5; margin-bottom: 0;}
.brand-sub {font-size: .7rem; color: #7F93B3; letter-spacing: .05em; margin-bottom: .6rem;}
.hero {border: 1px solid #1B2A45; background: #0E1A30; border-radius: 14px; padding: 26px 30px 22px; margin-bottom: 16px;}
.hero h1 {font-size: 2.1rem; margin: 0; padding: 0; color: #F1F5FB;}
.hero .sub {color: #7FB2FF; font-size: 1.08rem; margin: .35rem 0 .7rem;}
.tag {display: inline-block; font-size: .72rem; padding: 2px 10px; border-radius: 999px; border: 1px solid #2B4470;
      color: #9FB6D9; margin: 0 6px 4px 0;}
.cap-card {border: 1px solid #1B2A45; background: #0E1A30; border-radius: 12px; padding: 14px 16px; min-height: 118px;}
.cap-card h4 {font-size: .76rem; letter-spacing: .12em; color: #5EA1FF; margin: 0 0 6px; padding: 0;}
.cap-card p {font-size: .86rem; color: #C3CFE2; margin: 0;}
div[data-testid="stMetric"] {background: #0E1A30; border: 1px solid #1B2A45; border-radius: 12px; padding: 12px 16px;}
.pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .76rem; font-weight: 700;}
.sev-Low {background: #0F3324; color: #5FD39A;} .sev-Medium {background: #10264A; color: #7FB2FF;}
.sev-High {background: #3A3110; color: #F2C94C;} .sev-Critical {background: #44151A; color: #FF8080;}
.notice {border-left: 3px solid #3B82F6; background: #0E1A30; padding: 8px 12px; border-radius: 6px; color: #B9C7DC;
         font-size: .84rem; margin: 4px 0 12px;}
.warn {border-left-color: #EAB308;}
.pipeline {font-size: .82rem; color: #9FB6D9; margin: 6px 0 2px;}
.pipeline b {color: #E6ECF5; font-weight: 600;}
.step-wait {padding: 9px 14px; border: 1px solid #1B2A45; border-radius: 8px; color: #6F82A3; margin-bottom: 8px; font-size: .92rem;}
.kv {color: #8FA3C0; font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 0;}
.kv-val {color: #E6ECF5; font-size: 1rem; margin-bottom: .6rem;}
</style>
"""


# =============================================================================
# Cached resources & small helpers
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_client() -> GroqClient:
    return GroqClient()


@st.cache_data(ttl=300, show_spinner=False)
def connection_status(model: str, configured: bool) -> tuple[bool, str]:
    return get_client().check_connection()


@st.cache_data(show_spinner=False, max_entries=20)
def summarize_upload(name: str, data: bytes) -> tuple[str, pd.DataFrame | None, str]:
    """Turn an uploaded file into a short text summary for the agents. Returns (summary, preview, error)."""
    ext = name.rsplit(".", 1)[-1].lower()
    try:
        if ext == "txt":
            text = " ".join(data.decode("utf-8", errors="replace").split())
            return (f"Text file '{name}': {text[:1500]}", None, "") if text else ("", None, "The text file is empty.")
        if ext == "pdf":
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            text = " ".join(" ".join((page.extract_text() or "") for page in reader.pages[:5]).split())
            if not text:
                return "", None, "No extractable text found in the PDF (it may be a scanned image)."
            return f"PDF '{name}' ({len(reader.pages)} pages): {text[:1500]}", None, ""
        if ext == "csv":
            df = pd.read_csv(io.BytesIO(data))
        elif ext == "xlsx":
            df = pd.read_excel(io.BytesIO(data))
        else:
            return "", None, "Unsupported file type."
    except Exception as exc:
        return "", None, f"Could not read '{name}' ({type(exc).__name__}). Please check the file format and content."
    if df.empty:
        return "", None, f"'{name}' contains no data rows."
    parts = [f"Table '{name}': {len(df)} rows; columns: {', '.join(map(str, df.columns[:15]))}."]
    numeric = df.select_dtypes("number")
    for col in list(numeric.columns)[:8]:
        series = numeric[col].dropna()
        if not series.empty:
            parts.append(f"{col}: min {series.min():.4g}, max {series.max():.4g}, mean {series.mean():.4g}, "
                         f"last {series.iloc[-1]:.4g}.")
    return " ".join(parts)[:1500], df.head(50), ""


def pill(text: str, level: str | None = None) -> str:
    level = level if level in SEV_COLORS else calc.normalize_severity(level or text, "Medium")
    return f"<span class='pill sev-{level}'>{text}</span>"


def bullets(items: list, empty: str = "_None provided._") -> None:
    st.markdown("\n".join(f"- {i}" for i in items) if items else empty)


def money(value: float) -> str:
    return f"{value:,.0f} {calc.CURRENCY}"


def style_fig(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#C3CFE2"),
                      legend=dict(orientation="h", y=-0.15), title_font=dict(size=14))
    fig.update_xaxes(gridcolor="#1B2A45")
    fig.update_yaxes(gridcolor="#1B2A45")
    return fig


def chart(fig: go.Figure, height: int = 320) -> None:
    st.plotly_chart(style_fig(fig, height), width="stretch", config={"displayModeBar": False})


def kv(label: str, value) -> None:
    st.markdown(f"<p class='kv'>{label}</p><p class='kv-val'>{value if value not in (None, '') else '-'}</p>",
                unsafe_allow_html=True)


def notice(text: str, warn: bool = False) -> None:
    st.markdown(f"<div class='notice{' warn' if warn else ''}'>{text}</div>", unsafe_allow_html=True)


# =============================================================================
# Session state & navigation callbacks
# =============================================================================
def init_state(machines: pd.DataFrame) -> None:
    first = machines.iloc[0].to_dict() if not machines.empty else {}
    now = datetime.now()
    defaults = {
        "nav": "Dashboard", "demo_mode": True, "last_result": None, "show_notification": False,
        "f_equipment_id": first.get("equipment_id", ""), "f_equipment_name": first.get("name", ""),
        "f_equipment_type": first.get("type", ""), "f_plant_area": first.get("plant_area", ""),
        "f_operator": "", "f_date": now.date(), "f_time": now.time().replace(second=0, microsecond=0),
        "f_complaint": "", "f_condition": OPERATING_CONDITIONS[0], "f_additional": "",
        "f_demo_case": DEMO_CASES[0]["label"],
    }
    defaults.update({key: None for key, *_ in READING_FIELDS})
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def go_to(page: str) -> None:
    st.session_state.nav = page


def fill_machine_fields() -> None:
    machine = db.get_machine(st.session_state.f_equipment_id) or {}
    st.session_state.f_equipment_name = machine.get("name", "")
    st.session_state.f_equipment_type = machine.get("type", "")
    st.session_state.f_plant_area = machine.get("plant_area", "")
    st.session_state.f_runtime = float(machine["runtime_hours"]) if machine.get("runtime_hours") is not None else None


def load_demo_case(label: str | None = None, navigate: bool = False) -> None:
    label = label or st.session_state.f_demo_case
    case = next((c for c in DEMO_CASES if c["label"] == label), DEMO_CASES[0])
    st.session_state.f_demo_case = case["label"]
    st.session_state.f_equipment_id = case["equipment_id"]
    fill_machine_fields()
    now = datetime.now()
    st.session_state.update({
        "f_operator": case["operator"], "f_date": now.date(), "f_time": now.time().replace(second=0, microsecond=0),
        "f_complaint": case["complaint"], "f_condition": case["operating_condition"],
        "f_additional": case["additional_readings"], "f_load": case["load_pct"], "f_temp": case["temperature_c"],
        "f_pressure": case["pressure_bar"], "f_vibration": case["vibration_mm_s"],
        "f_oil_p": case["oil_pressure_bar"], "f_oil_t": case["oil_temperature_c"],
        "last_result": None, "show_notification": False,
    })
    if navigate:
        st.session_state.nav = "New Maintenance Request"


def start_analysis(equipment_id: str) -> None:
    """'Start AI Analysis' from the Machines page: prefill the form and navigate."""
    st.session_state.f_equipment_id = equipment_id
    fill_machine_fields()
    st.session_state.update({"f_complaint": "", "f_additional": "", "f_operator": "", "last_result": None,
                             **{key: None for key, *_ in READING_FIELDS if key != "f_runtime"}})
    st.session_state.nav = "New Maintenance Request"


def clear_form() -> None:
    st.session_state.update({"f_complaint": "", "f_additional": "", "f_operator": "", "last_result": None,
                             **{key: None for key, *_ in READING_FIELDS}})


def set_wo_status(wo_number: str, key: str) -> None:
    db.update_work_order_status(wo_number, st.session_state[key])


def reset_demo_database() -> None:
    db.reset_database()
    st.session_state.last_result = None


# =============================================================================
# Sidebar
# =============================================================================
def render_sidebar(client: GroqClient) -> None:
    with st.sidebar:
        st.markdown("<p class='brand'>AI MAINTENANCE COPILOT</p>"
                    "<p class='brand-sub'>Multi-Agent Maintenance Automation</p>", unsafe_allow_html=True)
        st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
        st.divider()
        ok, message = connection_status(client.model, client.configured)
        if ok:
            st.markdown("**Groq API:** 🟢 Connected")
        elif not client.configured:
            st.markdown("**Groq API:** 🔴 Not configured")
            st.caption("Add GROQ_API_KEY to Streamlit Secrets or a local .env file.")
        else:
            st.markdown("**Groq API:** 🔴 Unavailable")
            st.caption(message)
        st.markdown("**Selected model:**")
        st.code(client.model, language=None)
        st.caption(f"LLM calls this session: {client.calls} · cache hits: {client.cache_hits}")
        st.divider()
        st.toggle("Demo Mode", key="demo_mode", help="Show synthetic demo cases and demo-data tools.")
        if st.session_state.demo_mode:
            st.caption(DEMO_NOTICE)
            with st.expander("Demo data tools"):
                st.caption("Recreate the SQLite database with fresh synthetic data.")
                confirm = st.checkbox("I understand all generated records will be removed")
                st.button("Reset demo database", disabled=not confirm, on_click=reset_demo_database)


# =============================================================================
# Dashboard
# =============================================================================
def render_dashboard(client: GroqClient) -> None:
    st.markdown(
        "<div class='hero'><h1>AI Maintenance Copilot</h1>"
        "<p class='sub'>From Maintenance Complaint to AI-Assisted Work Order in Minutes</p>"
        "<span class='tag'>Multi-Agent Industrial Maintenance Automation Platform</span>"
        "<span class='tag'>7 specialised AI agents</span><span class='tag'>Groq LLM</span>"
        "<span class='tag'>Automatic work orders &amp; PDF reports</span></div>", unsafe_allow_html=True)
    notice(f"{DEMO_NOTICE} No SCADA / plant-control integration - this is a decision-support and workflow automation prototype.")

    cards = [
        ("GENERATIVE AI", "Generate maintenance reports and engineering summaries."),
        ("AGENTIC AI", "AI agents analyse and act on maintenance requests."),
        ("MULTI-AGENT", "Multiple specialised agents collaborate through structured outputs."),
        ("AI WORKFLOW", "Automates the complete maintenance lifecycle."),
        ("BUSINESS AUTOMATION", "Automatically creates work orders and documentation."),
    ]
    for col, (title, text) in zip(st.columns(5), cards):
        col.markdown(f"<div class='cap-card'><h4>{title}</h4><p>{text}</p></div>", unsafe_allow_html=True)
    st.markdown("<p class='pipeline'>Workflow: <b>Complaint</b> → Triage → Diagnostic → Safety → Maintenance Planner → "
                "Senior Engineer → <b>Work Order</b> → <b>Report</b> → Database &amp; KPI update</p>", unsafe_allow_html=True)
    c1, c2, _ = st.columns([1.3, 1.3, 3])
    c1.button("Run demo: Pump P-102 high vibration", type="primary", on_click=load_demo_case,
              args=(DEMO_CASES[0]["label"], True), width="stretch")
    c2.button("New Maintenance Request", on_click=go_to, args=("New Maintenance Request",), width="stretch")

    last = st.session_state.get("last_result")
    if last and last.get("success"):
        wo = last["work_order"]
        st.success(f"Latest automation: **{wo['wo_number']}** ({wo['priority']}) created for **{wo['equipment_id']}** - "
                   f"{wo['title']}. KPIs below include this work order.")

    stats = db.dashboard_stats()
    st.subheader("Operations KPIs")
    row1, row2 = st.columns(4), st.columns(4)
    row1[0].metric("Total Machines", stats["total_machines"])
    row1[1].metric("Open Maintenance Requests", stats["open_requests"])
    row1[2].metric("Critical Issues", stats["critical_open"])
    row1[3].metric("Active Work Orders", stats["active_work_orders"], help=f"{stats['ai_work_orders']} generated by the AI workflow")
    row2[0].metric("Completed Work Orders", stats["completed_work_orders"])
    row2[1].metric("Avg Resolution Time", f"{stats['avg_resolution_hours']:.1f} h")
    row2[2].metric("Maintenance Cost (history)", money(stats["maintenance_cost"]),
                   help=f"Open work orders estimated: {money(stats['open_estimated_cost'])}")
    row2[3].metric("Repeat Failures", stats["repeat_failures"], help="Equipment/failure-mode combinations seen more than once")

    requests, history = db.get_requests(), db.get_history()
    st.subheader("Maintenance insights")
    a, b = st.columns(2)
    with a:
        counts = requests["category"].replace("", "Unclassified").value_counts().reset_index()
        counts.columns = ["category", "requests"]
        chart(px.bar(counts.sort_values("requests"), x="requests", y="category", orientation="h",
                     title="Maintenance requests by category", color_discrete_sequence=PALETTE))
    with b:
        sev = requests[requests["severity"].isin(calc.SEVERITY_LEVELS)]["severity"].value_counts().reindex(
            calc.SEVERITY_LEVELS, fill_value=0).reset_index()
        sev.columns = ["severity", "requests"]
        chart(px.pie(sev, names="severity", values="requests", hole=0.55, title="Requests by severity",
                     color="severity", color_discrete_map=SEV_COLORS))
    a, b = st.columns(2)
    with a:
        req = requests.assign(month=pd.to_datetime(requests["created_at"], errors="coerce").dt.to_period("M").astype(str))
        hist = history.assign(month=pd.to_datetime(history["date"], errors="coerce").dt.to_period("M").astype(str))
        monthly = req.groupby("month").size().rename("requests").to_frame().join(
            hist.groupby("month")["maintenance_cost"].sum().rename("cost"), how="outer").fillna(0).reset_index()
        fig = go.Figure()
        fig.add_bar(x=monthly["month"], y=monthly["requests"], name="Requests", marker_color="#3B82F6")
        fig.add_scatter(x=monthly["month"], y=monthly["cost"], name=f"Cost ({calc.CURRENCY})", yaxis="y2",
                        line=dict(color="#22C55E", width=2))
        fig.update_layout(title="Monthly maintenance trend", yaxis2=dict(overlaying="y", side="right", showgrid=False))
        chart(fig)
    with b:
        freq = history.groupby("equipment_id").size().rename("failures").reset_index().sort_values("failures", ascending=False)
        chart(px.bar(freq, x="equipment_id", y="failures", title="Equipment failure frequency",
                     color_discrete_sequence=["#14B8A6"]))
    a, b = st.columns(2)
    with a:
        down = history.groupby("equipment_id")["downtime_hours"].sum().reset_index().sort_values("downtime_hours", ascending=False)
        chart(px.bar(down, x="equipment_id", y="downtime_hours", title="Downtime by equipment (h)",
                     color_discrete_sequence=["#60A5FA"]))
    with b:
        modes = history["failure_type"].value_counts().head(8).reset_index()
        modes.columns = ["failure_mode", "events"]
        chart(px.bar(modes.sort_values("events"), x="events", y="failure_mode", orientation="h",
                     title="Top recurring failure modes", color_discrete_sequence=["#818CF8"]))

    st.subheader("Recent work orders")
    wos = db.get_work_orders().head(8)
    st.dataframe(wos[["wo_number", "equipment_id", "title", "priority", "department", "created_at", "status"]].rename(columns={
        "wo_number": "WO ID", "equipment_id": "Equipment", "title": "Issue", "priority": "Priority",
        "department": "Department", "created_at": "Created", "status": "Status"}), hide_index=True, width="stretch")


# =============================================================================
# New maintenance request + workflow execution
# =============================================================================
def render_new_request(client: GroqClient) -> None:
    st.title("New Maintenance Request")
    st.caption("Submit an equipment complaint. Seven AI agents will triage, diagnose, assess safety, plan, decide, "
               "create a work order and write the report - automatically.")
    if not client.configured:
        notice("Groq API key is not configured. Add <b>GROQ_API_KEY</b> to Streamlit Secrets (or .env locally) to run "
               "the AI workflow. All other pages work without it.", warn=True)

    machines = db.get_machines()
    with st.container(border=True):
        st.markdown("**Demo cases** - synthetic scenarios for presentation")
        c1, c2, c3 = st.columns([3, 1.2, 1])
        c1.selectbox("Demo case", [c["label"] for c in DEMO_CASES], key="f_demo_case", label_visibility="collapsed")
        c2.button("Load Demo Case", on_click=load_demo_case, width="stretch")
        c3.button("Clear form", on_click=clear_form, width="stretch")
        if st.session_state.demo_mode:
            cols = st.columns(len(DEMO_CASES))
            for col, case in zip(cols, DEMO_CASES):
                col.button(case["label"].split(" - ", 1)[1].capitalize(), key=f"quick_{case['key']}",
                           on_click=load_demo_case, args=(case["label"],), width="stretch")

    st.markdown("##### Equipment")
    c = st.columns(4)
    ids = machines["equipment_id"].tolist()
    if st.session_state.f_equipment_id not in ids and ids:
        st.session_state.f_equipment_id = ids[0]
    c[0].selectbox("Equipment ID", ids, key="f_equipment_id", on_change=fill_machine_fields)
    c[1].text_input("Equipment Name", key="f_equipment_name")
    c[2].text_input("Equipment Type", key="f_equipment_type")
    c[3].text_input("Plant / Area", key="f_plant_area")
    c = st.columns(4)
    c[0].text_input("Operator", key="f_operator", placeholder="Name or shift")
    c[1].date_input("Date", key="f_date")
    c[2].time_input("Time", key="f_time")
    c[3].selectbox("Operating Condition", OPERATING_CONDITIONS, key="f_condition")
    st.text_area("Complaint", key="f_complaint", height=100,
                 placeholder="e.g. Pump vibration increased significantly and bearing temperature reached 92 °C.")

    st.markdown("##### Sensor readings (optional)")
    thr = calc.get_thresholds(st.session_state.f_equipment_type)
    cols = st.columns(4)
    for i, (key, label, _field, lo, hi, step) in enumerate(READING_FIELDS):
        help_text = f"Main monitored temperature for this type: {thr['temp_label']}" if key == "f_temp" else None
        cols[i % 4].number_input(label, min_value=lo, max_value=hi, step=step, key=key, help=help_text)
    cols[3].caption("Leave blank if not measured.")
    st.text_area("Any additional readings", key="f_additional", height=70,
                 placeholder="e.g. Motor current 41 A, discharge pressure fluctuating ±0.3 bar")

    uploaded = st.file_uploader("Attach supporting data (optional): CSV, Excel, PDF or TXT", type=["csv", "xlsx", "pdf", "txt"])
    attachment_summary, attachment_name = "", ""
    if uploaded is not None:
        summary, preview, error = summarize_upload(uploaded.name, uploaded.getvalue())
        if error:
            st.warning(f"{error} The request can still be submitted without the attachment.")
        else:
            attachment_summary, attachment_name = summary, uploaded.name
            with st.expander(f"Attachment summary sent to agents - {uploaded.name}"):
                st.caption(summary)
                if preview is not None:
                    st.dataframe(preview, width="stretch", height=200)

    readings = {field: st.session_state[key] for key, _l, field, *_ in READING_FIELDS if st.session_state[key] is not None}
    machine = db.get_machine(st.session_state.f_equipment_id) or {}
    machine = {**machine, "type": st.session_state.f_equipment_type}
    indicators = calc.compute_indicators(readings, machine, st.session_state.f_date)
    if indicators["checks"]:
        with st.expander(f"Platform pre-checks (deterministic Python, no AI) - severity floor: "
                         f"{indicators['severity_floor']}", expanded=False):
            st.dataframe(pd.DataFrame(indicators["checks"])[["parameter", "value", "assessment", "level"]],
                         hide_index=True, width="stretch")
            mt = indicators["maintenance"]
            if mt.get("maintenance_overdue"):
                st.warning(f"Preventive maintenance overdue by {mt['days_overdue']} days (next due {mt['next_maintenance']}).")
            st.caption(calc.THRESHOLDS_NOTE)

    run = st.button("RUN AI MAINTENANCE WORKFLOW", type="primary", width="stretch")
    just_ran = False
    if run:
        request, error = build_request(attachment_name, attachment_summary)
        if error:
            st.error(error)
        elif not client.configured:
            st.error("Groq API key is not configured. Add GROQ_API_KEY to Streamlit Secrets.")
        else:
            result = execute_workflow(client, request)
            st.session_state.last_result = result
            st.session_state.show_notification = False
            just_ran = True

    result = st.session_state.get("last_result")
    if result:
        if not just_ran:
            render_step_summary(result)
        render_results(result)


def build_request(attachment_name: str, attachment_summary: str) -> tuple[MaintenanceRequest | None, str]:
    s = st.session_state
    try:
        request = MaintenanceRequest(
            equipment_id=s.f_equipment_id or "", equipment_name=s.f_equipment_name, equipment_type=s.f_equipment_type,
            plant_area=s.f_plant_area, operator=s.f_operator, reported_at=datetime.combine(s.f_date, s.f_time),
            complaint=s.f_complaint, operating_condition=s.f_condition, additional_readings=s.f_additional,
            attachment_name=attachment_name, attachment_summary=attachment_summary,
            **{field: s[key] for key, _l, field, *_ in READING_FIELDS},
        )
        return request, ""
    except ValidationError as exc:
        messages = [e["msg"].removeprefix("Value error, ") for e in exc.errors()]
        return None, "Please fix the form: " + " ".join(messages)


def execute_workflow(client: GroqClient, request: MaintenanceRequest) -> dict:
    """Run the multi-agent workflow with a live, per-agent status display."""
    st.markdown("### AI WORKFLOW")
    progress = st.progress(0.0, text="Starting multi-agent workflow...")
    slots = {key: st.empty() for key, _ in STEPS}
    for key, name in STEPS:
        slots[key].markdown(f"<div class='step-wait'>⚪ {name} — Waiting</div>", unsafe_allow_html=True)
    boxes: dict = {}
    order = [key for key, _ in STEPS]

    def on_event(step) -> None:
        if step.status == "running":
            boxes[step.key] = slots[step.key].status(f"{step.name} — Running...", state="running", expanded=False)
            progress.progress(order.index(step.key) / len(order), text=f"{step.name} is working...")
            return
        box = boxes.get(step.key) or slots[step.key].status(step.name)
        if step.status == "done":
            box.update(label=f"🟢 {step.name} — Completed · {step.seconds:.1f}s", state="complete")
            with box:
                st.markdown(f"**Result:** {step.summary}")
                if step.rationale:
                    st.caption(step.rationale)
        else:
            box.update(label=f"🔴 {step.name} — Failed · {step.seconds:.1f}s", state="error", expanded=True)
            with box:
                st.error(step.error)

    result = MaintenanceWorkflow(client).run(request, on_event).to_dict()
    if result["success"]:
        progress.progress(1.0, text="Workflow complete - work order created and dashboard updated.")
        st.success(f"Automation complete: **{result['work_order']['wo_number']}** created for "
                   f"**{result['request']['equipment_id']}** · request {result['request_code']} saved · PDF report ready.")
    else:
        progress.empty()
        st.error(f"The workflow stopped: {result['error']}")
        for key, _ in STEPS:
            if result_step(result, key)["status"] == "pending":
                slots[key].markdown(f"<div class='step-wait'>⚪ {result_step(result, key)['name']} — Skipped</div>",
                                    unsafe_allow_html=True)
    return result


def result_step(result: dict, key: str) -> dict:
    return next(s for s in result["steps"] if s["key"] == key)


def render_step_summary(result: dict) -> None:
    st.markdown("### AI WORKFLOW")
    for step in result["steps"]:
        if step["status"] == "done":
            with st.status(f"🟢 {step['name']} — Completed · {step['seconds']:.1f}s", state="complete", expanded=False):
                st.markdown(f"**Result:** {step['summary']}")
                if step["rationale"]:
                    st.caption(step["rationale"])
        elif step["status"] == "failed":
            with st.status(f"🔴 {step['name']} — Failed", state="error", expanded=True):
                st.error(step["error"])
        else:
            st.markdown(f"<div class='step-wait'>⚪ {step['name']} — Not run</div>", unsafe_allow_html=True)


def render_results(result: dict) -> None:
    out = result["outputs"]
    if not result["success"]:
        st.info("Partial results are shown below where available. Fix the issue above and run the workflow again.")
    triage, diag, safety = out.get("triage"), out.get("diagnostic"), out.get("safety")
    plan, plan_calc, senior = out.get("planner"), out.get("planner_calc"), out.get("senior")
    report, wo = out.get("report"), result.get("work_order")

    st.markdown("### Agent outputs")
    tabs = st.tabs(["Triage", "Diagnostic", "Safety", "Maintenance Plan", "Senior Engineer", "Work Order", "Report & RCA"])
    with tabs[0]:
        if triage:
            c = st.columns(4)
            c[0].markdown(f"<p class='kv'>Category</p><p class='kv-val'>{triage['category']}</p>", unsafe_allow_html=True)
            c[1].markdown(f"<p class='kv'>Severity</p>{pill(triage['severity'], triage['severity'])}", unsafe_allow_html=True)
            c[2].markdown(f"<p class='kv'>Priority</p>{pill(triage['priority'], triage['severity'])}", unsafe_allow_html=True)
            c[3].markdown(f"<p class='kv'>Department</p><p class='kv-val'>{triage['department']}</p>", unsafe_allow_html=True)
            st.markdown(f"**Summary:** {triage['summary']}")
            st.markdown(f"**Safety risk:** {triage['safety_risk']}  \n**Immediate shutdown required:** "
                        f"{'Yes - follow site emergency procedures' if triage['requires_immediate_shutdown'] else 'No'}")
            st.caption(f"Rationale: {triage['reasoning']}")
            if result.get("triage_adjustment"):
                notice(result["triage_adjustment"], warn=True)
            checks = result.get("indicators", {}).get("checks", [])
            if checks:
                st.markdown("**Deterministic pre-checks used as evidence** (demo thresholds)")
                st.dataframe(pd.DataFrame(checks)[["parameter", "value", "assessment", "level"]], hide_index=True,
                             width="stretch")
        else:
            st.info("Triage did not complete.")
    with tabs[1]:
        if diag:
            st.markdown(f"**Primary hypothesis (requires verification):** {diag['primary_hypothesis']}")
            st.markdown(f"**Confidence:** {diag['confidence']}  \n**Operating recommendation:** {diag['operating_recommendation']}")
            a, b = st.columns(2)
            with a:
                st.markdown("**Probable causes**")
                bullets(diag["probable_causes"])
                st.markdown("**Recommended inspections**")
                bullets(diag["recommended_inspections"])
            with b:
                st.markdown("**Evidence**")
                bullets(diag["evidence"])
                st.markdown("**Additional data required**")
                bullets(diag["additional_data_required"])
            if result.get("history"):
                with st.expander(f"Maintenance history provided to the agent ({len(result['history'])} records)"):
                    st.dataframe(pd.DataFrame(result["history"]), hide_index=True, width="stretch")
            st.caption("AI diagnosis is a hypothesis for verification - not a certified engineering diagnosis.")
        else:
            st.info("Diagnostic Agent did not complete.")
    with tabs[2]:
        if safety:
            st.warning(SAFETY_DISCLAIMER)
            st.markdown(f"Risk level: {pill(safety['risk_level'] or 'n/a', safety['risk_level'])}", unsafe_allow_html=True)
            if safety.get("operating_constraint"):
                st.markdown(f"**Operating constraint:** {safety['operating_constraint']}")
            a, b = st.columns(2)
            with a:
                st.markdown("**Hazards**")
                bullets(safety["hazards"])
                st.markdown("**Required isolation (LOTO)**")
                bullets(safety["required_isolation"])
                st.markdown("**Permits**")
                bullets(safety["permit_requirements"])
            with b:
                st.markdown("**PPE**")
                bullets(safety["ppe"])
                st.markdown("**Precautions**")
                bullets(safety["precautions"])
            st.markdown(f"**Emergency action:** {safety['emergency_action']}")
        else:
            st.info("Safety Agent did not complete.")
    with tabs[3]:
        if plan and plan_calc:
            m = st.columns(4)
            m[0].metric("Estimated man-hours", f"{plan['estimated_hours']:g} h")
            m[1].metric("Est. downtime", f"{plan_calc['downtime']['total_downtime_hours']:g} h",
                        help=plan_calc["downtime"]["basis"])
            m[2].metric("Est. cost", money(plan_calc["cost"]["total_cost"]), help=plan_calc["cost"]["basis"])
            m[3].metric("Spare lines", len(plan_calc["parts"]))
            a, b = st.columns(2)
            with a:
                st.markdown("**Inspection sequence**")
                bullets(plan["inspection_sequence"])
                st.markdown("**Maintenance actions**")
                bullets(plan["maintenance_actions"])
                st.markdown("**Post-maintenance checks**")
                bullets(plan["post_maintenance_checks"])
            with b:
                st.markdown("**Tools required**")
                bullets(plan["tools_required"])
                st.markdown("**Manpower**")
                bullets(plan["manpower"])
                st.markdown("**Skill requirements**")
                bullets(plan["skill_requirements"])
            st.markdown("**Spare parts** (matched against the spare-parts database by Python)")
            if plan_calc["parts"]:
                st.dataframe(parts_frame(plan_calc["parts"]), hide_index=True, width="stretch")
            else:
                st.caption("No spare parts identified.")
        else:
            st.info("Maintenance Planner did not complete.")
    with tabs[4]:
        if senior:
            c = st.columns(2)
            c[0].markdown(f"<p class='kv'>Decision</p><p class='kv-val'>{senior['decision']}</p>", unsafe_allow_html=True)
            c[1].markdown(f"<p class='kv'>Urgency</p><p class='kv-val'>{senior['urgency']}</p>", unsafe_allow_html=True)
            st.markdown(f"**Engineering assessment:** {senior['engineering_assessment']}")
            st.markdown(f"**Recommended action:** {senior['recommended_action']}")
            a, b = st.columns(2)
            with a:
                st.markdown("**Conditions**")
                bullets(senior["conditions"])
            with b:
                st.markdown("**Next steps**")
                bullets(senior["next_steps"])
            st.markdown("**Agent conflicts reconciled**")
            bullets(senior.get("conflicts_resolved", []), "_No conflicts between agents were identified._")
            st.caption(f"Rationale: {senior['reasoning']}")
        else:
            st.info("Senior Engineer Agent did not complete.")
    with tabs[5]:
        if wo:
            render_work_order_details(wo)
        else:
            st.info("No work order was generated.")
    with tabs[6]:
        if report:
            st.markdown(f"**Executive summary**  \n{report['executive_summary']}")
            st.markdown(f"**Management summary**  \n{report['management_summary']}")
            st.markdown("#### Root Cause Analysis (template - complete after inspection)")
            st.markdown(f"**Problem statement:** {report['rca_problem_statement']}")
            a, b = st.columns(2)
            with a:
                st.markdown("**Possible root causes (to be verified)**")
                bullets(report["rca_possible_root_causes"])
                st.markdown("**5-Whys (provisional)**")
                bullets(report["rca_five_whys"])
                st.markdown("**Contributing factors**")
                bullets(report["rca_contributing_factors"])
            with b:
                st.markdown("**Verification actions**")
                bullets(report["rca_verification_actions"])
                st.markdown("**Corrective actions**")
                bullets(report["corrective_actions"])
                st.markdown("**Preventive actions**")
                bullets(report["preventive_actions"])
            st.markdown("**Recommended follow-up**")
            bullets(report["follow_up"])
        else:
            st.info("Report Agent did not complete.")

    with st.expander("View Agent Reasoning / Trace"):
        st.caption("Concise structured explanations and evidence produced by each agent. Hidden chain-of-thought is "
                   "never requested or displayed.")
        trace = [{"Step": s["name"], "Status": s["status"], "Time (s)": s["seconds"], "Result": s["summary"] or s["error"],
                  "Explanation": s["rationale"]} for s in result["steps"]]
        st.dataframe(pd.DataFrame(trace), hide_index=True, width="stretch")
        if result.get("triage_adjustment"):
            st.markdown(f"**Platform guard-rail:** {result['triage_adjustment']}")
        if diag and diag.get("evidence"):
            st.markdown("**Diagnostic evidence**")
            bullets(diag["evidence"])
        st.caption(f"Model: {result['model']} · Request: {result.get('request_code') or '-'}")

    if result["success"]:
        st.markdown("### Automated outputs")
        c1, c2, c3 = st.columns(3)
        pdf, pdf_error = get_pdf(result)
        if pdf:
            c1.download_button("Download PDF Maintenance Report", data=pdf, type="primary", width="stretch",
                               file_name=f"{wo['wo_number']}_maintenance_report.pdf", mime="application/pdf")
        else:
            c1.error(pdf_error)
        if c2.button("Generate Notification", width="stretch"):
            st.session_state.show_notification = True
        c3.button("View updated Dashboard", on_click=go_to, args=("Dashboard",), width="stretch")
        if st.session_state.show_notification:
            note = build_notification(result)
            with st.container(border=True):
                st.markdown("**Notification draft** - not sent (no email provider configured). Use the copy icons.")
                st.caption("Subject")
                st.code(note["subject"], language=None)
                st.caption("Recipients (demo addresses)")
                st.code("; ".join(note["recipients"]), language=None)
                st.caption("Body")
                st.code(note["body"], language=None)


def parts_frame(parts: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(parts)
    cols = {"part_number": "Part no.", "part_name": "Description", "quantity": "Qty", "stock": "Stock",
            "unit_cost": f"Unit cost ({calc.CURRENCY})", "lead_time_days": "Lead time (d)", "availability": "Availability",
            "note": "Note"}
    return df[[c for c in cols if c in df.columns]].rename(columns=cols)


def get_pdf(result: dict) -> tuple[bytes | None, str]:
    key = f"pdf_{result['work_order'].get('wo_number')}"
    if key not in st.session_state:
        try:
            st.session_state[key] = build_pdf(result)
        except Exception as exc:
            return None, f"PDF generation failed ({type(exc).__name__}). The work order is saved; try again later."
    return st.session_state[key], ""


def render_work_order_details(wo: dict) -> None:
    st.markdown(f"#### {wo.get('wo_number', '')} — {wo.get('title', '')}")
    st.markdown(f"{pill(wo.get('priority', ''), wo.get('severity') or wo.get('priority'))} &nbsp; "
                f"**{wo.get('equipment_id', '')}** {wo.get('equipment_name', '')} · {wo.get('department', '')} · "
                f"Status: **{wo.get('status', '')}**", unsafe_allow_html=True)
    c = st.columns(4)
    c[0].metric("Est. man-hours", f"{float(wo.get('estimated_manhours') or 0):g} h")
    c[1].metric("Est. downtime", f"{float(wo.get('estimated_downtime_hours') or 0):g} h")
    c[2].metric("Est. cost", money(float(wo.get("estimated_cost") or 0)))
    c[3].metric("Created", str(wo.get("created_at", ""))[:16].replace("T", " "))
    st.markdown(f"**Problem description:** {wo.get('problem_description', '')}")
    st.markdown(f"**Assigned team:** {wo.get('assigned_team', '')}")
    a, b = st.columns(2)
    with a:
        st.markdown("**Job steps**")
        steps = wo.get("job_steps") or []
        st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)) if steps else "_Not recorded._")
        st.markdown("**Acceptance criteria**")
        bullets(wo.get("acceptance_criteria") or [], "_Not recorded._")
    with b:
        st.markdown("**Safety requirements**")
        bullets(wo.get("safety_requirements") or [], "_Not recorded._")
        st.markdown("**Required tools**")
        bullets(wo.get("tools_required") or [], "_Not recorded._")
    if wo.get("spare_parts"):
        st.markdown("**Required spare parts**")
        st.dataframe(parts_frame(wo["spare_parts"]), hide_index=True, width="stretch")


# =============================================================================
# Machines
# =============================================================================
def render_machines(client: GroqClient) -> None:
    st.title("Machines")
    notice("Demonstration / simulated equipment - not connected to real industrial assets.")
    machines, history = db.get_machines(), db.get_history()
    overview = machines.copy()
    overview["failures"] = overview["equipment_id"].map(history.groupby("equipment_id").size()).fillna(0).astype(int)
    st.dataframe(overview[["equipment_id", "name", "type", "plant_area", "status", "criticality", "runtime_hours",
                           "last_maintenance", "next_maintenance", "failures"]], hide_index=True, width="stretch")

    eid = st.selectbox("Select equipment", machines["equipment_id"],
                       format_func=lambda e: f"{e} - {machines.set_index('equipment_id').loc[e, 'name']}")
    machine = db.get_machine(eid)
    events = history[history["equipment_id"] == eid]
    timing = calc.maintenance_timing(machine)
    st.markdown(f"### {machine['equipment_id']} — {machine['name']}")
    m = st.columns(5)
    m[0].metric("Status", machine["status"])
    m[1].metric("Criticality", machine["criticality"])
    m[2].metric("Runtime", f"{machine['runtime_hours']:,.0f} h")
    m[3].metric("Failure count", len(events))
    m[4].metric("Avg downtime", f"{events['downtime_hours'].mean():.1f} h" if len(events) else "-")
    a, b, c = st.columns(3)
    with a:
        kv("Type", machine["type"])
        kv("Plant / Area", machine["plant_area"])
        kv("Manufacturer / Model", f"{machine['manufacturer']} / {machine['model']}")
    with b:
        kv("Commissioned", machine["commission_date"])
        kv("Last maintenance", machine["last_maintenance"])
        kv("Next maintenance", machine["next_maintenance"])
    with c:
        kv("Days since maintenance", timing.get("days_since_maintenance"))
        kv("Est. running hours since maintenance", timing.get("est_running_hours_since_maintenance"))
        if timing.get("maintenance_overdue"):
            st.error(f"Preventive maintenance overdue by {timing['days_overdue']} days")
        else:
            st.success(f"Next maintenance due in {timing.get('days_until_due', '-')} days")
    st.button("Start AI Analysis", type="primary", on_click=start_analysis, args=(eid,))

    st.markdown("#### Recent maintenance events")
    if events.empty:
        st.caption("No maintenance history recorded for this equipment.")
    else:
        st.dataframe(events[["date", "failure_type", "symptoms", "action_taken", "downtime_hours", "maintenance_cost",
                             "root_cause", "status"]].head(10), hide_index=True, width="stretch")
    with st.expander("Compatible spare parts"):
        st.dataframe(pd.DataFrame(db.get_compatible_parts(eid, limit=50)), hide_index=True, width="stretch")


# =============================================================================
# Maintenance history
# =============================================================================
def render_history(client: GroqClient) -> None:
    st.title("Maintenance History")
    notice("Synthetic maintenance records for demonstration. The Diagnostic Agent receives the most recent records for "
           "the selected equipment as context.")
    history = db.get_history()
    c = st.columns([2, 1.2, 1.2])
    search = c[0].text_input("Search", placeholder="Equipment ID, failure type, symptom, root cause...")
    equipment = c[1].multiselect("Equipment", sorted(history["equipment_id"].unique()))
    category = c[2].multiselect("Category", sorted(history["category"].dropna().unique()))
    df = history
    if search:
        text = df[["equipment_id", "failure_type", "symptoms", "action_taken", "root_cause", "recommendation"]].astype(str)
        df = df[text.apply(lambda col: col.str.contains(search, case=False, regex=False)).any(axis=1)]
    if equipment:
        df = df[df["equipment_id"].isin(equipment)]
    if category:
        df = df[df["category"].isin(category)]

    m = st.columns(4)
    m[0].metric("Events", len(df))
    m[1].metric("Total downtime", f"{df['downtime_hours'].sum():,.1f} h")
    m[2].metric("Maintenance cost", money(df["maintenance_cost"].sum()))
    repeats = df.groupby(["equipment_id", "failure_type"]).size()
    m[3].metric("Repeat failure modes", int((repeats > 1).sum()))
    st.dataframe(df[["date", "equipment_id", "category", "failure_type", "symptoms", "action_taken", "downtime_hours",
                     "maintenance_cost", "root_cause", "status"]], hide_index=True, width="stretch")
    if not df.empty:
        a, b = st.columns(2)
        with a:
            down = df.groupby("equipment_id")["downtime_hours"].sum().reset_index()
            chart(px.bar(down, x="equipment_id", y="downtime_hours", title="Downtime (h) - filtered",
                         color_discrete_sequence=["#3B82F6"]), 300)
        with b:
            fails = df["failure_type"].value_counts().head(8).reset_index()
            fails.columns = ["failure_type", "events"]
            chart(px.bar(fails.sort_values("events"), x="events", y="failure_type", orientation="h",
                         title="Previous failures - filtered", color_discrete_sequence=["#14B8A6"]), 300)
        with st.expander("Previous recommendations"):
            for row in df.head(15).itertuples():
                st.markdown(f"- **{row.date} · {row.equipment_id} · {row.failure_type}:** {row.recommendation}")


# =============================================================================
# Work orders
# =============================================================================
def render_work_orders(client: GroqClient) -> None:
    st.title("Work Orders")
    wos = db.get_work_orders()
    c = st.columns(4)
    priority = c[0].multiselect("Priority", ["P1", "P2", "P3", "P4"])
    equipment = c[1].multiselect("Equipment", sorted(wos["equipment_id"].unique()))
    status = c[2].multiselect("Status", WO_STATUSES)
    category = c[3].multiselect("Category", sorted(wos["category"].replace("", pd.NA).dropna().unique()))
    df = wos
    for column, values in (("priority", priority), ("equipment_id", equipment), ("status", status), ("category", category)):
        if values:
            df = df[df[column].isin(values)]
    st.caption(f"{len(df)} work orders · {int(df['ai_generated'].sum())} generated by the AI workflow · "
               "select a row to view full details")
    table = df[["wo_number", "equipment_id", "title", "priority", "department", "created_at", "status", "ai_generated"]].rename(
        columns={"wo_number": "WO ID", "equipment_id": "Equipment", "title": "Issue", "priority": "Priority",
                 "department": "Department", "created_at": "Created", "status": "Status", "ai_generated": "AI"})
    table["AI"] = table["AI"].map({1: "AI", 0: "-"})
    event = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                         selection_mode="single-row", key="wo_table")
    rows = event.selection.rows if event and hasattr(event, "selection") else []
    if not rows:
        st.info("Select a work order in the table to display its details.")
        return
    wo = db.get_work_order(table.iloc[rows[0]]["WO ID"])
    if not wo:
        return
    with st.container(border=True):
        render_work_order_details(wo)
        c1, c2, _ = st.columns([1.2, 1.4, 2])
        key = f"status_{wo['wo_number']}"
        st.session_state.setdefault(key, wo["status"] if wo["status"] in WO_STATUSES else "Open")
        c1.selectbox("Update status", WO_STATUSES, key=key, on_change=set_wo_status, args=(wo["wo_number"], key))
        if wo.get("result"):
            pdf, error = get_pdf(wo["result"])
            if pdf:
                c2.download_button("Download PDF report", data=pdf, file_name=f"{wo['wo_number']}_maintenance_report.pdf",
                                   mime="application/pdf", width="stretch")
            else:
                c2.error(error)
        else:
            c2.caption("Synthetic seed record - no AI report attached.")


# =============================================================================
# Analytics
# =============================================================================
def render_analytics(client: GroqClient) -> None:
    st.title("Analytics")
    requests, history, wos, stats = db.get_requests(), db.get_history(), db.get_work_orders(), db.dashboard_stats()
    m = st.columns(6)
    m[0].metric("Total requests", len(requests))
    m[1].metric("Critical requests", int((requests["severity"] == "Critical").sum()))
    m[2].metric("Avg downtime", f"{history['downtime_hours'].mean():.1f} h")
    m[3].metric("Maintenance cost", money(history["maintenance_cost"].sum()))
    m[4].metric("Avg resolution", f"{stats['avg_resolution_hours']:.1f} h")
    m[5].metric("AI work orders", stats["ai_work_orders"])

    a, b = st.columns(2)
    with a:
        cats = history["category"].value_counts().reset_index()
        cats.columns = ["category", "events"]
        chart(px.pie(cats, names="category", values="events", hole=0.5, title="Most frequent failure categories",
                     color_discrete_sequence=PALETTE))
    with b:
        top = history.groupby("equipment_id").agg(failures=("id", "count"), downtime=("downtime_hours", "sum")).reset_index()
        chart(px.bar(top.sort_values("failures", ascending=False), x="equipment_id", y="failures", color="downtime",
                     title="Top equipment by failures (colour = downtime h)", color_continuous_scale="Blues"))
    a, b = st.columns(2)
    with a:
        state = wos["status"].map(lambda s: "Closed" if s in ("Completed", "Cancelled") else "Open").value_counts().reset_index()
        state.columns = ["state", "work_orders"]
        chart(px.pie(state, names="state", values="work_orders", hole=0.55, title="Open vs closed work orders",
                     color="state", color_discrete_map={"Open": "#3B82F6", "Closed": "#22C55E"}))
    with b:
        cost = history.groupby("equipment_id")["maintenance_cost"].sum().reset_index().sort_values("maintenance_cost")
        chart(px.bar(cost, x="maintenance_cost", y="equipment_id", orientation="h",
                     title=f"Maintenance cost by equipment ({calc.CURRENCY})", color_discrete_sequence=["#14B8A6"]))
    a, b = st.columns(2)
    with a:
        done = wos[wos["status"] == "Completed"].copy()
        done["resolution_h"] = (pd.to_datetime(done["completed_at"]) - pd.to_datetime(done["created_at"])).dt.total_seconds() / 3600
        res = done.groupby("category")["resolution_h"].mean().round(1).reset_index()
        chart(px.bar(res, x="category", y="resolution_h", title="Average resolution time by category (h)",
                     color_discrete_sequence=["#60A5FA"]))
    with b:
        runs = db.get_agent_runs()
        if runs.empty:
            st.info("Agent execution telemetry appears here after the first AI workflow run.")
        else:
            perf = runs[runs["status"] == "done"].groupby("agent_name")["duration_s"].mean().round(2).reset_index()
            chart(px.bar(perf, x="agent_name", y="duration_s", title="Average agent execution time (s)",
                         color_discrete_sequence=["#818CF8"]))
    with st.expander("Agent run audit log"):
        st.dataframe(db.get_agent_runs(), hide_index=True, width="stretch")


# =============================================================================
# About
# =============================================================================
def render_about(client: GroqClient) -> None:
    st.title("About AI Maintenance Copilot")
    st.markdown("AI Maintenance Copilot is a **multi-agent AI decision-support and workflow automation prototype** for "
                "industrial maintenance. It turns a free-text equipment complaint into a triaged, diagnosed, "
                "safety-assessed and planned **work order** with a downloadable **PDF report** - and records everything "
                "in a database that drives the dashboard.")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#0E1A30", color="#3B82F6", fontcolor="#E6ECF5", fontname="Helvetica", fontsize=11];
      edge [color="#5E7BA8"];
      C [label="Complaint +\\nreadings", color="#22C55E"]; V [label="Validation +\\nPython calcs"];
      T [label="Triage\\nAgent"]; D [label="Diagnostic\\nAgent"]; S [label="Safety\\nAgent"]; P [label="Maintenance\\nPlanner"];
      E [label="Senior Engineer\\nAgent"]; W [label="Work Order\\nAgent"]; R [label="Report\\nAgent"];
      DB [label="SQLite\\nwork orders, history,\\nspare parts", shape=cylinder, color="#14B8A6"];
      O [label="Dashboard KPIs\\nPDF report\\nNotification", color="#22C55E"];
      C -> V -> T -> D -> S -> P -> E -> W -> R -> O;
      DB -> D [style=dashed, label=" history", fontcolor="#8FA3C0", fontsize=9];
      DB -> P [style=dashed, label=" spare parts", fontcolor="#8FA3C0", fontsize=9];
      W -> DB [style=dashed, label=" WO number", fontcolor="#8FA3C0", fontsize=9];
    }""")
    st.markdown("#### Agents")
    st.dataframe(pd.DataFrame([
        ["Triage Agent", "Classifies category, severity, priority, department, shutdown need", "Complaint, readings, Python pre-checks"],
        ["Diagnostic Agent", "Probable causes, evidence, inspections, data gaps (with uncertainty)", "Triage, readings, equipment history"],
        ["Safety Agent", "Hazards, LOTO isolation, PPE, permits, emergency action", "Triage, diagnosis"],
        ["Maintenance Planner", "Actions, inspection sequence, tools, spares, manpower, time", "Diagnosis, safety, spare-parts DB"],
        ["Senior Engineer", "Reconciles conflicts (safety first) and makes the decision", "All previous agents, timing, history"],
        ["Work Order Agent", "Converts the decision into an executable work order", "Senior decision, plan, safety"],
        ["Report Agent", "Executive summary, management summary, RCA template, follow-up", "All outputs + work order"],
    ], columns=["Agent", "Responsibility", "Inputs"]), hide_index=True, width="stretch")
    st.markdown("#### Hackathon skills demonstrated")
    st.markdown("""
- **Generative AI** - engineering summaries, work-order text, RCA template and PDF report content.
- **Agentic AI** - agents receive context, reason over evidence and produce actions (decisions, work orders).
- **Multi-Agent System** - seven specialised agents with separate prompts and Pydantic output contracts.
- **AI Workflow** - an orchestrator chains agents, Python calculations, database look-ups and guard-rails.
- **Business Process Automation** - request → work order → spare parts → report → database → dashboard, automatically.
""")
    st.markdown("#### Design principles")
    st.markdown("""
- One LLM call per agent (7 per workflow) with compact JSON prompts, output token limits and response caching.
- Deterministic maths (vibration zones, temperature/pressure deviation, overdue maintenance, downtime, cost) is done in Python - never by the LLM.
- A deterministic **severity floor** prevents the LLM from under-rating conditions that exceed demo thresholds.
- Structured outputs are validated with Pydantic; malformed JSON is extracted, retried once, then reported as a controlled error.
- The API key is read from Streamlit Secrets / environment and is never displayed, logged or stored.
""")
    st.markdown("#### Limitations")
    st.markdown("""
- Synthetic equipment and data; no SCADA, historian, CMMS or plant-control integration.
- Thresholds are demo values, not OEM limits or standards. AI outputs can be incomplete or wrong.
- SQLite on Streamlit Community Cloud is ephemeral - data resets when the app restarts.
""")
    st.warning(DISCLAIMER)


# =============================================================================
# Main
# =============================================================================
RENDERERS = {
    "Dashboard": render_dashboard, "New Maintenance Request": render_new_request, "Machines": render_machines,
    "Maintenance History": render_history, "Work Orders": render_work_orders, "Analytics": render_analytics,
    "About": render_about,
}


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        db.init_db()
        machines = db.get_machines()
    except Exception as exc:
        st.error(f"Database error: could not initialise the SQLite database ({type(exc).__name__}). "
                 "Check that the app folder is writable.")
        st.stop()
    init_state(machines)
    client = get_client()
    render_sidebar(client)
    try:
        RENDERERS.get(st.session_state.nav, render_dashboard)(client)
    except sqlite3.Error as exc:
        st.error(f"Database error ({type(exc).__name__}). Try 'Reset demo database' in the sidebar (Demo Mode).")
    except Exception as exc:
        st.error("Something went wrong while rendering this page. The rest of the app is still available.")
        with st.expander("Technical details"):
            st.exception(exc)
    st.markdown("---")
    st.caption("AI Maintenance Copilot · decision-support and workflow automation prototype · " + DEMO_NOTICE +
               " AI-generated recommendations must be verified by qualified personnel.")


main()
