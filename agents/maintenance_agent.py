"""Agent 4 - Maintenance Planning Agent: actions, tools, spares, manpower, time."""
from __future__ import annotations

from typing import Annotated, Any

from pydantic import BeforeValidator, Field

from agents.diagnostic_agent import DiagnosticResult
from agents.llm import AgentOutput, GroqClient, Number, Text, TextList, compact
from agents.safety_agent import SafetyResult
from agents.triage_agent import TriageResult

AGENT_NAME = "Maintenance Planner"

SYSTEM_PROMPT = """You are the MAINTENANCE PLANNING AGENT of an industrial maintenance decision-support platform.
Your job: turn the diagnosis and safety assessment into a practical, executable maintenance plan.
- Start with verification/inspection steps, then corrective actions conditional on findings ("If bearing damage is confirmed, ...").
- Spare parts: prefer parts listed in 'spare_parts_database' and copy their part_number exactly. If a needed part is not
  listed, give part_number "N/A" and say "not in database" in the note. Do not invent stock levels or prices.
- manpower items like "1 x Mechanical technician"; estimated_hours = total man-hours; estimated_downtime_hours = equipment
  unavailable time for the repair itself (numbers only).
- Include post-maintenance checks that verify the fix (e.g. vibration and temperature re-measurement after run-in).
- Respect the safety agent's isolation and permit requirements."""


def _to_part_lines(value: Any) -> Any:
    """Allow the model to return plain strings for spare parts."""
    if isinstance(value, str):
        value = [value]
    if isinstance(value, list):
        return [{"part_name": item} if isinstance(item, str) else item for item in value]
    return [] if value is None else value


class SparePartLine(AgentOutput):
    part_number: Text = Field("", description="part_number from database or N/A")
    part_name: Text = Field("", description="part name")
    quantity: Number = Field(1, description="quantity")
    note: Text = Field("", description="why it is needed / conditional use")


class MaintenancePlan(AgentOutput):
    maintenance_actions: TextList = Field(default_factory=list, description="maintenance action")
    inspection_sequence: TextList = Field(default_factory=list, description="ordered inspection step")
    tools_required: TextList = Field(default_factory=list, description="tool or instrument")
    spare_parts: Annotated[list[SparePartLine], BeforeValidator(_to_part_lines)] = Field(default_factory=list)
    manpower: TextList = Field(default_factory=list, description="e.g. 1 x Mechanical technician")
    estimated_hours: Number = 0
    estimated_downtime_hours: Number = 0
    skill_requirements: TextList = Field(default_factory=list, description="skill")
    post_maintenance_checks: TextList = Field(default_factory=list, description="check after maintenance")
    reasoning: Text = Field("", description="1-2 sentence justification")


def run(client: GroqClient, ctx: dict, triage: TriageResult, diagnosis: DiagnosticResult,
        safety: SafetyResult) -> MaintenancePlan:
    payload = {
        "equipment": ctx["machine_brief"],
        "complaint": ctx["request"]["complaint"],
        "triage": triage.model_dump(include={"category", "severity", "priority"}),
        "diagnosis": diagnosis.model_dump(include={"primary_hypothesis", "probable_causes", "recommended_inspections"}),
        "safety": safety.model_dump(include={"required_isolation", "permit_requirements", "operating_constraint"}),
        "spare_parts_database": ctx["spare_parts"],
    }
    return client.generate_json(SYSTEM_PROMPT, "Create the maintenance plan.\nDATA: " + compact(payload),
                                MaintenancePlan, max_tokens=1100)
