"""Agent 2 - Diagnostic Agent: probable causes, evidence and inspections."""
from __future__ import annotations

from pydantic import Field

from agents.llm import AgentOutput, GroqClient, Text, TextList, compact
from agents.triage_agent import TriageResult

AGENT_NAME = "Diagnostic Agent"

SYSTEM_PROMPT = """You are the DIAGNOSTIC AGENT of an industrial maintenance decision-support platform.
Your job: analyse the symptoms, readings and this equipment's own maintenance history, and propose probable causes.
- Rank probable causes from most to least likely. Prefix each with "Possible cause:" or "Likely:".
- Every evidence item must start with a source tag: [Reading], [Complaint], [History] or [Assumption].
- Use [History] only for records present in 'maintenance_history'. If history is empty, say no history was provided.
- Confidence: Low, Medium or High, plus a short qualifier. Never claim certainty; diagnosis always "requires verification".
- Recommend practical inspections that would confirm or rule out each main cause.
- operating_recommendation: your view on continued operation (e.g. "Continue with enhanced monitoring", "Inspect before continued operation")."""


class DiagnosticResult(AgentOutput):
    probable_causes: TextList = Field(default_factory=list, description="'Likely: ...' / 'Possible cause: ...' ranked")
    primary_hypothesis: Text = Field("", description="most likely cause, phrased as a hypothesis requiring verification")
    confidence: Text = Field("", description="Low|Medium|High - short qualifier")
    evidence: TextList = Field(default_factory=list, description="[Reading]/[Complaint]/[History]/[Assumption] evidence item")
    recommended_inspections: TextList = Field(default_factory=list, description="inspection or test to verify a cause")
    additional_data_required: TextList = Field(default_factory=list, description="missing data that would improve the diagnosis")
    operating_recommendation: Text = Field("", description="recommendation on continued operation")
    reasoning: Text = Field("", description="1-2 sentence justification")


def run(client: GroqClient, ctx: dict, triage: TriageResult) -> DiagnosticResult:
    payload = {
        "equipment": ctx["machine_brief"],
        "complaint": ctx["request"]["complaint"],
        "operating_condition": ctx["request"]["operating_condition"],
        "readings": ctx["readings"],
        "additional_readings": ctx["request"]["additional_readings"],
        "attachment_summary": ctx["request"]["attachment_summary"][:800],
        "calculated_indicators": ctx["indicators_brief"],
        "triage": triage.model_dump(include={"category", "severity", "summary"}),
        "maintenance_history": ctx["history"],
    }
    return client.generate_json(SYSTEM_PROMPT, "Diagnose this equipment problem.\nDATA: " + compact(payload),
                                DiagnosticResult, max_tokens=900)
