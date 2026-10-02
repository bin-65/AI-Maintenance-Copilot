"""Agent 1 - Triage Agent: classifies the complaint and sets severity/priority."""
from __future__ import annotations

from pydantic import Field

from agents.llm import AgentOutput, Flag, GroqClient, Text, compact

AGENT_NAME = "Triage Agent"

SYSTEM_PROMPT = """You are the TRIAGE AGENT of an industrial maintenance decision-support platform.
Your job: classify a maintenance complaint so it is routed to the right team with the right urgency.
- Category options: Mechanical, Electrical, Instrumentation, Process, Lubrication, Cooling, Structural, Other.
- Severity options: Low, Medium, High, Critical. Priority mapping: Critical=P1, High=P2, Medium=P3, Low=P4.
- 'calculated_indicators' were computed deterministically by the platform from the supplied readings; use them as evidence.
- Set requires_immediate_shutdown=true only if the supplied evidence indicates an imminent risk to people or equipment.
- Department examples: Mechanical Maintenance, Electrical Maintenance, Instrumentation & Control, Operations / Process Engineering.
- When the evidence is ambiguous, choose the more conservative (safer) classification and say why."""


class TriageResult(AgentOutput):
    category: Text = Field("Other", description="Mechanical|Electrical|Instrumentation|Process|Lubrication|Cooling|Structural|Other")
    severity: Text = Field("Medium", description="Low|Medium|High|Critical")
    priority: Text = Field("P3", description="P1|P2|P3|P4")
    department: Text = Field("", description="responsible maintenance department")
    safety_risk: Text = Field("", description="Low|Medium|High|Critical - short phrase naming the main risk")
    requires_immediate_shutdown: Flag = False
    summary: Text = Field("", description="one-sentence problem summary")
    reasoning: Text = Field("", description="1-2 sentence justification citing the supplied evidence")


def run(client: GroqClient, ctx: dict) -> TriageResult:
    payload = {
        "equipment": ctx["machine_brief"],
        "complaint": ctx["request"]["complaint"],
        "operating_condition": ctx["request"]["operating_condition"],
        "readings": ctx["readings"],
        "additional_readings": ctx["request"]["additional_readings"],
        "attachment_summary": ctx["request"]["attachment_summary"][:800],
        "calculated_indicators": ctx["indicators_brief"],
    }
    return client.generate_json(SYSTEM_PROMPT, "Classify this maintenance request.\nDATA: " + compact(payload),
                                TriageResult, max_tokens=500)
