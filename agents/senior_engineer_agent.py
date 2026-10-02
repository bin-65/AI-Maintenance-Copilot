"""Agent 5 - Senior Engineer Agent: reconciles all agent outputs into one decision."""
from __future__ import annotations

from pydantic import Field

from agents.diagnostic_agent import DiagnosticResult
from agents.llm import AgentOutput, GroqClient, Text, TextList, compact
from agents.maintenance_agent import MaintenancePlan
from agents.safety_agent import SafetyResult
from agents.triage_agent import TriageResult

AGENT_NAME = "Senior Engineer"

SYSTEM_PROMPT = """You are the SENIOR ENGINEER AGENT - the decision/orchestration agent of an industrial maintenance
decision-support platform. You review the outputs of the Triage, Diagnostic, Safety and Maintenance Planning agents.
- Reconcile conflicts between agents. Safety constraints ALWAYS take precedence over production or convenience: if the
  Diagnostic agent suggests continued operation but the Safety agent requires inspection first, the decision must respect
  the safety constraint. List each conflict you found and how you resolved it in 'conflicts_resolved' (empty if none).
- decision options: "Stop and inspect immediately", "Controlled shutdown for planned inspection",
  "Continue with enhanced monitoring until planned repair", "Continue normal operation".
- urgency: Immediate | Within 24 hours | Within 7 days | Next planned shutdown.
- Do not add facts that are not in the agent outputs. Conditions = what must be true to proceed (permits, isolation, parts).
- Your output supports, and does not replace, the judgment of a qualified engineer."""


class SeniorEngineerResult(AgentOutput):
    engineering_assessment: Text = Field("", description="2-3 sentence consolidated assessment")
    recommended_action: Text = Field("", description="single clear recommended action")
    urgency: Text = Field("", description="Immediate|Within 24 hours|Within 7 days|Next planned shutdown")
    decision: Text = Field("", description="one of the decision options")
    reasoning: Text = Field("", description="1-2 sentence justification")
    conditions: TextList = Field(default_factory=list, description="condition that must be met")
    next_steps: TextList = Field(default_factory=list, description="next step")
    conflicts_resolved: TextList = Field(default_factory=list, description="conflict between agents and resolution")


def run(client: GroqClient, ctx: dict, triage: TriageResult, diagnosis: DiagnosticResult, safety: SafetyResult,
        plan: MaintenancePlan, plan_calc: dict) -> SeniorEngineerResult:
    payload = {
        "equipment": ctx["machine_brief"],
        "triage": triage.model_dump(exclude={"reasoning"}),
        "diagnosis": diagnosis.model_dump(include={"primary_hypothesis", "confidence", "probable_causes", "operating_recommendation"}),
        "safety": safety.model_dump(include={"risk_level", "hazards", "operating_constraint", "emergency_action"}),
        "maintenance_plan": plan.model_dump(include={"maintenance_actions", "estimated_hours", "estimated_downtime_hours"}),
        "parts_availability": [f"{p['part_number']}: {p['availability']}" for p in plan_calc["parts"]],
        "maintenance_timing": ctx["indicators_brief"]["maintenance"],
        "history_summary": ctx["indicators_brief"]["history"],
    }
    return client.generate_json(SYSTEM_PROMPT, "Review all agent outputs and decide.\nDATA: " + compact(payload),
                                SeniorEngineerResult, max_tokens=900)
