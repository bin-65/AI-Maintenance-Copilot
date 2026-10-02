"""Agent 6 - Work Order Agent: converts the engineering decision into a work order.

The LLM drafts the human-readable parts (title, job steps, acceptance criteria).
Deterministic fields (priority, parts, cost, downtime, WO number) come from
upstream agents, Python calculations and the database - never from the LLM.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import Field

from agents.llm import AgentOutput, GroqClient, Text, TextList, compact
from agents.maintenance_agent import MaintenancePlan
from agents.safety_agent import SafetyResult
from agents.senior_engineer_agent import SeniorEngineerResult
from agents.triage_agent import TriageResult

AGENT_NAME = "Work Order Generator"

SYSTEM_PROMPT = """You are the WORK ORDER AGENT of an industrial maintenance decision-support platform.
Your job: convert the senior engineer's decision and the maintenance plan into a clear, executable work order.
- Title: short (max 10 words), action-oriented.
- job_steps: numbered-style ordered steps a technician can follow, starting with permit/isolation verification and ending
  with return-to-service checks. Keep each step to one sentence.
- safety_requirements: condensed from the safety agent (isolation, PPE, permits). Do not drop any isolation requirement.
- acceptance_criteria: measurable where the data allows (e.g. "vibration below site acceptance limit after 2 h run-in"),
  without inventing numeric limits that were not supplied.
- assigned_team: a role-based team description (e.g. "Mechanical maintenance crew + vibration analyst")."""


class WorkOrderDraft(AgentOutput):
    title: Text = Field("", description="short action-oriented title")
    problem_description: Text = Field("", description="2 sentence problem description")
    assigned_team: Text = Field("", description="role-based team")
    job_steps: TextList = Field(default_factory=list, description="ordered job step")
    safety_requirements: TextList = Field(default_factory=list, description="safety requirement")
    acceptance_criteria: TextList = Field(default_factory=list, description="acceptance criterion")
    notes: Text = Field("", description="optional short note for the planner")


def run(client: GroqClient, ctx: dict, triage: TriageResult, senior: SeniorEngineerResult, plan: MaintenancePlan,
        safety: SafetyResult) -> WorkOrderDraft:
    payload = {
        "equipment": ctx["machine_brief"],
        "complaint": ctx["request"]["complaint"],
        "triage": triage.model_dump(include={"category", "priority", "department", "summary"}),
        "senior_engineer": senior.model_dump(include={"decision", "recommended_action", "urgency", "conditions"}),
        "maintenance_plan": plan.model_dump(include={"maintenance_actions", "inspection_sequence", "post_maintenance_checks", "manpower"}),
        "safety": safety.model_dump(include={"required_isolation", "ppe", "permit_requirements", "precautions"}),
    }
    return client.generate_json(SYSTEM_PROMPT, "Draft the work order.\nDATA: " + compact(payload),
                                WorkOrderDraft, max_tokens=900)


def build_work_order(draft: WorkOrderDraft, request: dict, machine: dict, triage: TriageResult,
                     plan: MaintenancePlan, plan_calc: dict) -> dict:
    """Assemble the final work-order record (number is assigned when saved to SQLite)."""
    return {
        "wo_number": "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "equipment_id": request["equipment_id"],
        "equipment_name": machine.get("name") or request.get("equipment_name", ""),
        "title": draft.title or triage.summary[:80],
        "problem_description": draft.problem_description or request["complaint"],
        "category": triage.category,
        "severity": triage.severity,
        "priority": triage.priority,
        "department": triage.department,
        "assigned_team": draft.assigned_team or triage.department,
        "tools_required": plan.tools_required,
        "spare_parts": plan_calc["parts"],
        "safety_requirements": draft.safety_requirements,
        "job_steps": draft.job_steps or plan.maintenance_actions,
        "estimated_manhours": round(plan.estimated_hours, 1),
        "estimated_downtime_hours": plan_calc["downtime"]["total_downtime_hours"],
        "estimated_cost": plan_calc["cost"]["total_cost"],
        "acceptance_criteria": draft.acceptance_criteria or plan.post_maintenance_checks,
        "notes": draft.notes,
        "status": "Open",
    }
