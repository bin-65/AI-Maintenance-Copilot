"""Agent 7 - Report Agent: executive summary, RCA template and follow-up.

The PDF itself is assembled deterministically (utils/pdf_generator.py) from
all agent outputs; this agent writes the narrative sections.
"""
from __future__ import annotations

from pydantic import Field

from agents.diagnostic_agent import DiagnosticResult
from agents.llm import AgentOutput, GroqClient, Text, TextList, compact
from agents.safety_agent import SafetyResult
from agents.senior_engineer_agent import SeniorEngineerResult
from agents.triage_agent import TriageResult

AGENT_NAME = "Report Generator"

SYSTEM_PROMPT = """You are the REPORT AGENT of an industrial maintenance decision-support platform.
Your job: write the narrative sections of a professional maintenance assessment report for engineers and management.
- executive_summary: 3-4 sentences - what happened, how serious, what is being done, the work order number.
- management_summary: 2-3 plain-language sentences on business impact (downtime, cost estimate, risk) for a plant manager.
- initial_condition: 1-2 sentences describing the reported condition using only supplied readings.
- Root Cause Analysis is a TEMPLATE to be completed after inspection: root causes are hypotheses "to be verified";
  the 5-Whys chain must be marked as provisional. Do not state a confirmed root cause.
- Corrective actions fix this event; preventive actions stop recurrence (use maintenance history patterns if supplied).
- follow_up: concrete follow-up items (e.g. re-measure vibration after 24 h, review lubrication interval)."""


class ReportResult(AgentOutput):
    executive_summary: Text = Field("", description="3-4 sentences")
    management_summary: Text = Field("", description="2-3 sentences, business impact")
    initial_condition: Text = Field("", description="1-2 sentences")
    rca_problem_statement: Text = Field("", description="one-sentence problem statement")
    rca_possible_root_causes: TextList = Field(default_factory=list, description="hypothesis to be verified")
    rca_five_whys: TextList = Field(default_factory=list, description="Why 1: ... (provisional)")
    rca_contributing_factors: TextList = Field(default_factory=list, description="contributing factor")
    rca_verification_actions: TextList = Field(default_factory=list, description="how to confirm the root cause")
    corrective_actions: TextList = Field(default_factory=list, description="corrective action")
    preventive_actions: TextList = Field(default_factory=list, description="preventive action")
    follow_up: TextList = Field(default_factory=list, description="follow-up item")


def run(client: GroqClient, ctx: dict, triage: TriageResult, diagnosis: DiagnosticResult, safety: SafetyResult,
        senior: SeniorEngineerResult, work_order: dict) -> ReportResult:
    payload = {
        "equipment": ctx["machine_brief"],
        "complaint": ctx["request"]["complaint"],
        "readings": ctx["readings"],
        "triage": triage.model_dump(include={"category", "severity", "priority", "summary"}),
        "diagnosis": diagnosis.model_dump(include={"primary_hypothesis", "probable_causes", "confidence"}),
        "safety": safety.model_dump(include={"risk_level", "hazards"}),
        "senior_engineer": senior.model_dump(include={"decision", "recommended_action", "urgency"}),
        "work_order": {k: work_order.get(k) for k in ("wo_number", "title", "priority", "estimated_manhours",
                                                       "estimated_downtime_hours", "estimated_cost")},
        "cost_currency": ctx["currency"],
        "history_summary": ctx["indicators_brief"]["history"],
    }
    return client.generate_json(SYSTEM_PROMPT, "Write the report sections.\nDATA: " + compact(payload),
                                ReportResult, max_tokens=1300)
