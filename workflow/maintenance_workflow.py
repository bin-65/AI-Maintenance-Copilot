"""Multi-agent maintenance workflow (the orchestrator).

Complaint -> Validation -> Triage -> Diagnostic -> Safety -> Planner -> Senior Engineer
          -> Work Order -> Report -> Database / dashboard update

Each agent makes exactly one LLM call (7 per run). Deterministic calculations,
database look-ups and record creation are done in Python between agent steps.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from pydantic import BaseModel, field_validator

from agents import (diagnostic_agent, maintenance_agent, report_agent, safety_agent, senior_engineer_agent,
                    triage_agent, work_order_agent)
from agents.llm import GroqClient, LLMError
from database import database as db
from utils import calculations as calc

STEPS = [
    ("validation", "Input Validation"),
    ("triage", triage_agent.AGENT_NAME),
    ("diagnostic", diagnostic_agent.AGENT_NAME),
    ("safety", safety_agent.AGENT_NAME),
    ("planner", maintenance_agent.AGENT_NAME),
    ("senior", senior_engineer_agent.AGENT_NAME),
    ("work_order", work_order_agent.AGENT_NAME),
    ("report", report_agent.AGENT_NAME),
    ("automation", "Database & Dashboard Update"),
]
AGENT_STEP_KEYS = {"triage", "diagnostic", "safety", "planner", "senior", "work_order", "report"}


class MaintenanceRequest(BaseModel):
    """Validated input from the New Maintenance Request form."""

    equipment_id: str
    equipment_name: str = ""
    equipment_type: str = ""
    plant_area: str = ""
    operator: str = ""
    reported_at: datetime
    complaint: str
    operating_condition: str = ""
    load_pct: Optional[float] = None
    runtime_hours: Optional[float] = None
    temperature_c: Optional[float] = None
    pressure_bar: Optional[float] = None
    vibration_mm_s: Optional[float] = None
    oil_pressure_bar: Optional[float] = None
    oil_temperature_c: Optional[float] = None
    additional_readings: str = ""
    attachment_name: str = ""
    attachment_summary: str = ""

    @field_validator("equipment_id")
    @classmethod
    def _equipment_required(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("Equipment ID is required.")
        return value

    @field_validator("complaint")
    @classmethod
    def _complaint_required(cls, value: str) -> str:
        value = (value or "").strip()
        if len(value) < 10:
            raise ValueError("Please describe the complaint (at least 10 characters).")
        return value[:3000]

    def readings(self) -> dict[str, float]:
        keys = ("load_pct", "runtime_hours", "temperature_c", "pressure_bar", "vibration_mm_s",
                "oil_pressure_bar", "oil_temperature_c")
        return {k: getattr(self, k) for k in keys if getattr(self, k) is not None}


@dataclass
class StepRecord:
    key: str
    name: str
    status: str = "pending"  # pending | running | done | failed
    summary: str = ""
    rationale: str = ""
    seconds: float = 0.0
    error: str = ""


@dataclass
class WorkflowResult:
    request: dict
    model: str
    steps: list[StepRecord]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    request_id: Optional[int] = None
    request_code: str = ""
    machine: dict = field(default_factory=dict)
    indicators: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    outputs: dict = field(default_factory=dict)
    work_order: dict = field(default_factory=dict)
    triage_adjustment: str = ""
    success: bool = False
    error: str = ""

    def step(self, key: str) -> StepRecord:
        return next(s for s in self.steps if s.key == key)

    def to_dict(self) -> dict:
        return asdict(self)


class StepFailed(Exception):
    pass


class MaintenanceWorkflow:
    """Simple sequential orchestrator - easy to read, easy to extend."""

    def __init__(self, client: GroqClient):
        self.client = client

    # ------------------------------------------------------------------ #
    def run(self, request: MaintenanceRequest, on_event: Callable[[StepRecord], None] | None = None) -> WorkflowResult:
        notify = on_event or (lambda step: None)
        result = WorkflowResult(
            request={**request.model_dump(mode="json"), "readings": request.readings()},
            model=self.client.model,
            steps=[StepRecord(key, name) for key, name in STEPS],
        )
        try:
            ctx = self._step(result, "validation", notify, lambda: self._prepare(request, result),
                             lambda c: (f"{result.request_code} created - {len(c['readings'])} readings, "
                                        f"{len(c['history'])} history records, {len(c['spare_parts'])} compatible parts",
                                        f"Deterministic severity floor: {c['indicators']['severity_floor']} "
                                        f"({calc.THRESHOLDS_NOTE.lower()})"))

            triage = self._step(result, "triage", notify, lambda: self._triage(ctx, result),
                                lambda t: (f"{t.category} | {t.severity} | {t.priority} | {t.department}", t.reasoning))
            db.update_request(result.request_id, category=triage.category, severity=triage.severity,
                              priority=triage.priority, status="Analysed")

            diagnosis = self._step(result, "diagnostic", notify, lambda: diagnostic_agent.run(self.client, ctx, triage),
                                   lambda d: (f"{d.primary_hypothesis} (confidence: {d.confidence})", d.reasoning))

            safety = self._step(result, "safety", notify, lambda: safety_agent.run(self.client, ctx, triage, diagnosis),
                                lambda s: (f"Risk {s.risk_level} - {len(s.hazards)} hazards, {len(s.permit_requirements)} permits",
                                           s.reasoning))

            plan, plan_calc = self._step(result, "planner", notify, lambda: self._plan(ctx, triage, diagnosis, safety, result),
                                         lambda pc: (f"{len(pc[0].maintenance_actions)} actions, {len(pc[1]['parts'])} spare lines, "
                                                     f"{pc[0].estimated_hours:g} man-h, est. downtime "
                                                     f"{pc[1]['downtime']['total_downtime_hours']:g} h, est. cost "
                                                     f"{pc[1]['cost']['total_cost']:,.0f} {calc.CURRENCY}", pc[0].reasoning))

            senior = self._step(result, "senior", notify,
                                lambda: senior_engineer_agent.run(self.client, ctx, triage, diagnosis, safety, plan, plan_calc),
                                lambda s: (f"{s.decision} - urgency: {s.urgency}", s.reasoning))

            work_order = self._step(result, "work_order", notify,
                                    lambda: self._work_order(ctx, triage, senior, plan, safety, plan_calc, result),
                                    lambda wo: (f"{wo['wo_number']} - {wo['priority']} - {wo['title']}",
                                                f"Assigned to {wo['assigned_team']}; {len(wo['job_steps'])} job steps."))

            self._step(result, "report", notify,
                       lambda: report_agent.run(self.client, ctx, triage, diagnosis, safety, senior, work_order),
                       lambda r: ("Executive summary, RCA template and follow-up generated", r.management_summary))

            self._step(result, "automation", notify, lambda: self._finalise(result, triage),
                       lambda _: (f"Saved {result.request_code} and {work_order['wo_number']}; 7 agent runs logged; dashboard KPIs updated",
                                  "Request status set to 'Work Order Created'; PDF report available for download."))
            result.success = True
        except StepFailed as exc:
            result.error = str(exc)
            if result.request_id:
                try:
                    db.update_request(result.request_id, status="Analysis Failed")
                except Exception:
                    pass
        return result

    # ------------------------------------------------------------------ #
    def _step(self, result: WorkflowResult, key: str, notify, fn: Callable[[], Any],
              describe: Callable[[Any], tuple[str, str]]) -> Any:
        """Run one step with timing, UI notification, audit logging and controlled errors."""
        step = result.step(key)
        step.status = "running"
        notify(step)
        start = time.perf_counter()
        try:
            output = fn()
            step.summary, step.rationale = describe(output)
        except Exception as exc:
            step.seconds = round(time.perf_counter() - start, 2)
            step.status = "failed"
            step.error = str(exc) if isinstance(exc, LLMError) else f"{type(exc).__name__}: {exc}"
            notify(step)
            self._log(result, step, None)
            raise StepFailed(f"{step.name} failed: {step.error}") from None
        step.seconds = round(time.perf_counter() - start, 2)
        step.status = "done"
        if isinstance(output, BaseModel):
            result.outputs.setdefault(key, output.model_dump())
        notify(step)
        if key in AGENT_STEP_KEYS:
            self._log(result, step, result.outputs.get(key))
        return output

    def _log(self, result: WorkflowResult, step: StepRecord, output: Any) -> None:
        try:
            db.log_agent_run(result.request_id, step.name, step.status, step.seconds,
                             step.summary or step.error, output, self.client.model)
        except Exception:
            pass  # audit logging must never break the workflow

    # ------------------------------------------------------------------ #
    def _prepare(self, request: MaintenanceRequest, result: WorkflowResult) -> dict:
        if not self.client.configured:
            raise LLMError(self.client.init_error or "Groq API key is not configured. Add GROQ_API_KEY to Streamlit Secrets.")
        machine = db.get_machine(request.equipment_id) or {
            "equipment_id": request.equipment_id, "name": request.equipment_name, "type": request.equipment_type,
            "plant_area": request.plant_area, "criticality": "Unknown", "status": "Unknown",
        }
        if request.equipment_name:
            machine = {**machine, "name": request.equipment_name}
        if request.equipment_type:
            machine = {**machine, "type": request.equipment_type}
        history = db.get_recent_history(request.equipment_id, limit=6)
        parts = db.get_compatible_parts(request.equipment_id)
        readings = request.readings()
        indicators = calc.compute_indicators(readings, machine, request.reported_at.date(), history)

        request_id, code = db.create_request({
            "created_at": request.reported_at.isoformat(timespec="seconds"), "equipment_id": request.equipment_id,
            "equipment_name": machine.get("name", ""), "equipment_type": machine.get("type", ""),
            "plant_area": request.plant_area or machine.get("plant_area", ""), "operator": request.operator,
            "complaint": request.complaint, "operating_condition": request.operating_condition,
            "readings": readings, "status": "In Analysis", "source": "AI workflow",
        })
        result.request_id, result.request_code = request_id, code
        result.machine, result.indicators, result.history = machine, indicators, history

        brief_keys = ("equipment_id", "name", "type", "plant_area", "criticality", "status", "runtime_hours",
                      "last_maintenance", "next_maintenance")
        return {
            "machine_brief": {k: machine.get(k) for k in brief_keys if machine.get(k) not in (None, "")},
            "request": {
                "complaint": request.complaint, "operating_condition": request.operating_condition,
                "additional_readings": request.additional_readings, "attachment_summary": request.attachment_summary,
            },
            "readings": readings,
            "indicators": indicators,
            "indicators_brief": {
                "note": indicators["thresholds_note"],
                "checks": [{k: c[k] for k in ("parameter", "value", "assessment", "level")} for c in indicators["checks"]],
                "severity_floor": indicators["severity_floor"],
                "maintenance": indicators["maintenance"],
                "history": indicators["history"],
            },
            "history": history,
            "spare_parts": parts,
            "currency": calc.CURRENCY,
        }

    def _triage(self, ctx: dict, result: WorkflowResult) -> triage_agent.TriageResult:
        triage = triage_agent.run(self.client, ctx)
        # Deterministic guard-rails: normalise labels and never let the LLM rate below the calculated floor.
        triage.category = calc.normalize_category(triage.category)
        severity = calc.normalize_severity(triage.severity)
        floor = ctx["indicators"]["severity_floor"]
        if calc.severity_rank(severity) < calc.severity_rank(floor):
            result.triage_adjustment = (f"Severity raised from {severity} to {floor} by the deterministic safety floor: "
                                        + "; ".join(ctx["indicators"]["floor_reasons"]))
            severity = floor
        triage.severity = severity
        triage.priority = calc.PRIORITY_BY_SEVERITY[severity]
        if not triage.department:
            triage.department = calc.DEPARTMENT_BY_CATEGORY[triage.category]
        result.outputs["triage"] = triage.model_dump()
        return triage

    def _plan(self, ctx, triage, diagnosis, safety, result: WorkflowResult):
        plan =maintenance_agent.run(self.client, ctx, triage, diagnosis, safety)
        parts = calc.match_spare_parts([p.model_dump() for p in plan.spare_parts], ctx["spare_parts"])
        plan_calc = {
            "parts": parts,
            "downtime": calc.estimate_downtime(plan.estimated_downtime_hours, plan.estimated_hours, parts),
            "cost": calc.estimate_cost(plan.estimated_hours, parts),
        }
        result.outputs["planner"] = plan.model_dump()
        result.outputs["planner_calc"] = plan_calc
        return plan, plan_calc

    def _work_order(self, ctx, triage, senior, plan, safety, plan_calc, result: WorkflowResult) -> dict:
        draft =work_order_agent.run(self.client, ctx, triage, senior, plan, safety)
        work_order = work_order_agent.build_work_order(draft, result.request, result.machine, triage, plan, plan_calc)
        _, number = db.create_work_order(work_order, result.request_id)
        work_order["wo_number"] = number
        result.work_order = work_order
        result.outputs["work_order"] = draft.model_dump()
        return work_order

    def _finalise(self, result: WorkflowResult, triage) -> None:
        db.update_request(result.request_id, status="Work Order Created")
        if triage.requires_immediate_shutdown or triage.severity == "Critical":
            db.update_machine_status(result.request["equipment_id"], "Stopped - Pending Maintenance")
        elif db.get_machine(result.request["equipment_id"]):
            db.update_machine_status(result.request["equipment_id"], f"Running - {triage.priority} WO Open")
        stored = result.to_dict()
        stored["success"] = True
        for step in stored["steps"]:
            if step["key"] == "automation":
                step.update(status="done", summary="Records saved; dashboard updated")
        db.attach_workflow_result(result.work_order["wo_number"], stored)


# ---------------------------------------------------------------------------- #
# Notification (generated, not sent - no email API required)
# ---------------------------------------------------------------------------- #
RECIPIENTS_BY_DEPARTMENT = {
    "Mechanical Maintenance": "mechanical.supervisor@plant.example",
    "Electrical Maintenance": "electrical.supervisor@plant.example",
    "Instrumentation & Control": "ic.supervisor@plant.example",
    "Operations / Process Engineering": "process.engineer@plant.example",
    "Civil & Structural": "civil.supervisor@plant.example",
}


def build_notification(result: dict) -> dict:
    triage = result["outputs"].get("triage", {})
    senior = result["outputs"].get("senior", {})
    safety = result["outputs"].get("safety", {})
    wo = result.get("work_order", {})
    priority = triage.get("priority", "P3")
    equipment = result["request"]["equipment_id"]
    recipients = ["maintenance.planner@plant.example",
                  RECIPIENTS_BY_DEPARTMENT.get(triage.get("department", ""), "maintenance.supervisor@plant.example"),
                  "shift.supervisor@plant.example"]
    if priority in ("P1", "P2"):
        recipients += ["maintenance.manager@plant.example", "hse.officer@plant.example"]
    level = {"P1": "Critical", "P2": "High-priority", "P3": "Medium-priority", "P4": "Low-priority"}.get(priority, "")
    body = f"""Hello team,

A {level.lower()} maintenance request has been generated for {equipment} ({result['machine'].get('name', '')}).

Work order:        {wo.get('wo_number', '-')}
Request:           {result.get('request_code', '-')}
Priority/Severity: {priority} / {triage.get('severity', '-')}
Department:        {triage.get('department', '-')}
Issue:             {triage.get('summary', '-')}

Engineering decision: {senior.get('decision', '-')} (urgency: {senior.get('urgency', '-')})
Recommended action:   {senior.get('recommended_action', '-')}
Safety risk level:    {safety.get('risk_level', '-')} - permits: {', '.join(safety.get('permit_requirements', [])) or 'per site procedure'}

Estimated man-hours: {wo.get('estimated_manhours', 0):g} h | Estimated downtime: {wo.get('estimated_downtime_hours', 0):g} h

Please review the work order in AI Maintenance Copilot and confirm planning.
AI-generated guidance - verify with qualified personnel and site HSE procedures before execution.

-- AI Maintenance Copilot (demo environment)"""
    return {"subject": f"{priority} Maintenance Alert - {equipment} - {wo.get('wo_number', '')}",
            "recipients": recipients, "body": body}
