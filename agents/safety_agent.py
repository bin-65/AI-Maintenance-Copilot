"""Agent 3 - Safety Agent: hazards, isolation, PPE and permits."""
from __future__ import annotations

from pydantic import Field

from agents.diagnostic_agent import DiagnosticResult
from agents.llm import AgentOutput, GroqClient, Text, TextList, compact
from agents.triage_agent import TriageResult

AGENT_NAME = "Safety Agent"

SAFETY_DISCLAIMER = "AI-generated guidance - verify with site HSE procedures."

SYSTEM_PROMPT = """You are the SAFETY AGENT of an industrial maintenance decision-support platform.
Your job: assess the safety implications of the reported problem and of the maintenance work it will require.
- Identify realistic hazards for this equipment type (rotating parts, stored energy, hot surfaces, pressure, electrical, etc.).
- List isolations needed before work (electrical, mechanical, process, pressure release) referring to LOTO generically.
- List PPE, precautions and permits typically required (e.g. Permit to Work, hot work, confined space, work at height) - only if relevant.
- operating_constraint: what must happen before continued operation, if anything (e.g. "Inspect before continued operation").
- emergency_action: what operators should do if the condition escalates - always defer to site emergency procedures.
- Never suggest bypassing interlocks, trips, guards or safety systems. Your output is guidance to be verified by site HSE."""


class SafetyResult(AgentOutput):
    risk_level: Text = Field("", description="Low|Medium|High|Critical")
    hazards: TextList = Field(default_factory=list, description="hazard")
    required_isolation: TextList = Field(default_factory=list, description="isolation step (LOTO)")
    ppe: TextList = Field(default_factory=list, description="PPE item")
    precautions: TextList = Field(default_factory=list, description="precaution")
    permit_requirements: TextList = Field(default_factory=list, description="permit")
    emergency_action: Text = Field("", description="action if condition escalates")
    operating_constraint: Text = Field("", description="safety constraint on continued operation")
    reasoning: Text = Field("", description="1-2 sentence justification")


def run(client: GroqClient, ctx: dict, triage: TriageResult, diagnosis: DiagnosticResult) -> SafetyResult:
    payload = {
        "equipment": ctx["machine_brief"],
        "complaint": ctx["request"]["complaint"],
        "readings": ctx["readings"],
        "calculated_indicators": ctx["indicators_brief"],
        "triage": triage.model_dump(include={"category", "severity", "safety_risk", "requires_immediate_shutdown"}),
        "diagnosis": diagnosis.model_dump(include={"primary_hypothesis", "probable_causes", "operating_recommendation"}),
    }
    return client.generate_json(SYSTEM_PROMPT, "Assess the safety implications.\nDATA: " + compact(payload),
                                SafetyResult, max_tokens=800)
