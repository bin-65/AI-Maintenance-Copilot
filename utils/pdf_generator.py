"""Professional PDF maintenance report built with ReportLab (from workflow results)."""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from utils.calculations import CURRENCY

FOOTER_TEXT = ("AI-generated maintenance assistance. Verify all recommendations with qualified personnel and "
               "applicable site procedures before execution.")
DISCLAIMER = ("AI Maintenance Copilot is a decision-support and workflow automation prototype. AI-generated "
              "recommendations are not a substitute for qualified engineering judgment, manufacturer instructions, "
              "plant operating procedures, permits, isolation/LOTO procedures, or HSE requirements. This report was "
              "produced in a demo environment using synthetic equipment and maintenance data.")

NAVY = colors.HexColor("#0B1F3A")
BLUE = colors.HexColor("#2563EB")
LIGHT = colors.HexColor("#EEF3FA")
GRID = colors.HexColor("#C9D4E5")

_REPLACEMENTS = {"≥": ">=", "≤": "<=", "→": "->", "←": "<-", "±": "+/-", "≈": "~",
                 "✓": "OK", "•": "-", "−": "-", " ": " "}


def _clean(value: Any) -> str:
    """Make text safe for ReportLab paragraphs and the built-in (Latin-1/cp1252) fonts."""
    text = "" if value is None else str(value)
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = text.encode("cp1252", "replace").decode("cp1252")
    return escape(text)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold", fontSize=18, textColor=NAVY,
                                alignment=TA_LEFT, spaceAfter=2),
        "subtitle": ParagraphStyle("st", parent=base["Normal"], fontSize=11, textColor=BLUE, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, textColor=NAVY,
                             spaceBefore=10, spaceAfter=4),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=10, textColor=BLUE,
                             spaceBefore=4, spaceAfter=2),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9.2, leading=12.5),
        "small": ParagraphStyle("s", parent=base["Normal"], fontSize=8, leading=10, textColor=colors.HexColor("#4B5563")),
        "cell": ParagraphStyle("c", parent=base["Normal"], fontSize=8.4, leading=10.5),
        "cellb": ParagraphStyle("cb", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=8.4, leading=10.5),
    }


def _table(rows: list[list[Any]], widths: list[float], st: dict, header: bool = False) -> Table:
    data = []
    for i, row in enumerate(rows):
        style = st["cellb"] if header and i == 0 else st["cell"]
        data.append([Paragraph(_clean(c), st["cellb"] if (not header and j == 0) else style) for j, c in enumerate(row)])
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    cmds = [("GRID", (0, 0), (-1, -1), 0.4, GRID), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), LIGHT)]
    else:
        cmds += [("BACKGROUND", (0, 0), (0, -1), LIGHT)]
    table.setStyle(TableStyle(cmds))
    return table


def _bullets(items: list[str], st: dict, empty: str = "None provided.") -> list:
    if not items:
        return [Paragraph(_clean(empty), st["body"])]
    return [Paragraph("&bull; " + _clean(item), st["body"]) for item in items]


def _page_decor(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 14 * mm, width, 14 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(15 * mm, height - 9 * mm, "AI Maintenance Copilot")
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(width - 15 * mm, height - 9 * mm, "Maintenance Assessment Report")
    canvas.setFillColor(colors.HexColor("#4B5563"))
    canvas.setFont("Helvetica", 7)
    canvas.drawString(15 * mm, 9 * mm, FOOTER_TEXT)
    canvas.drawRightString(width - 15 * mm, 5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(result: dict) -> bytes:
    """Return the PDF report as bytes for st.download_button."""
    st = _styles()
    out = result.get("outputs", {})
    triage, diag, safety = out.get("triage", {}), out.get("diagnostic", {}), out.get("safety", {})
    plan, plan_calc, senior = out.get("planner", {}), out.get("planner_calc", {}), out.get("senior", {})
    report, wo = out.get("report", {}), result.get("work_order", {})
    req, machine, ind = result.get("request", {}), result.get("machine", {}), result.get("indicators", {})
    full_width = 180 * mm

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=22 * mm,
                            bottomMargin=16 * mm, title="Maintenance Assessment Report", author="AI Maintenance Copilot")
    story: list = [
        Paragraph("AI Maintenance Copilot", st["title"]),
        Paragraph("Maintenance Assessment Report", st["subtitle"]),
        _table([
            ["Report ID", f"RPT-{wo.get('wo_number', 'DRAFT')}", "Date", datetime.now().strftime("%Y-%m-%d %H:%M")],
            ["Equipment", f"{req.get('equipment_id', '')} - {machine.get('name', '')}", "Priority",
             f"{triage.get('priority', '-')} ({triage.get('severity', '-')})"],
            ["Work Order", wo.get("wo_number", "-"), "Request", result.get("request_code", "-")],
            ["AI model", result.get("model", "-"), "Status", wo.get("status", "-")],
        ], [28 * mm, 62 * mm, 28 * mm, 62 * mm], st),
        Spacer(1, 4),
    ]

    def section(title: str, *flowables) -> None:
        story.append(Paragraph(_clean(title), st["h2"]))
        story.extend(flowables)

    section("1. Executive Summary", Paragraph(_clean(report.get("executive_summary", "")), st["body"]),
            Paragraph("<b>Management summary:</b> " + _clean(report.get("management_summary", "")), st["body"]))

    section("2. Equipment Information", _table([
        ["Equipment ID", machine.get("equipment_id", req.get("equipment_id", "")), "Type", machine.get("type", "")],
        ["Name", machine.get("name", ""), "Plant / Area", machine.get("plant_area", req.get("plant_area", ""))],
        ["Criticality", machine.get("criticality", "-"), "Runtime hours", machine.get("runtime_hours", "-")],
        ["Last maintenance", machine.get("last_maintenance", "-"), "Next maintenance", machine.get("next_maintenance", "-")],
    ], [30 * mm, 60 * mm, 30 * mm, 60 * mm], st))

    section("3. Reported Problem", Paragraph(_clean(req.get("complaint", "")), st["body"]),
            Paragraph(_clean(f"Reported by {req.get('operator') or '-'} at {str(req.get('reported_at', ''))[:16].replace('T', ' ')}; "
                             f"operating condition: {req.get('operating_condition') or '-'}."), st["small"]))

    checks = ind.get("checks", [])
    condition_rows = [["Parameter", "Value", "Assessment (demo thresholds)", "Level"]] + [
        [c["parameter"], c["value"], c["assessment"], c["level"]] for c in checks]
    section("4. Initial Condition", Paragraph(_clean(report.get("initial_condition", "")), st["body"]), Spacer(1, 3),
            _table(condition_rows, [38 * mm, 25 * mm, 95 * mm, 22 * mm], st, header=True) if checks
            else Paragraph("No numeric readings supplied.", st["body"]),
            Paragraph(_clean(ind.get("thresholds_note", "")), st["small"]))

    section("5. AI Triage", _table([
        ["Category", triage.get("category", "")], ["Severity / Priority", f"{triage.get('severity', '')} / {triage.get('priority', '')}"],
        ["Department", triage.get("department", "")], ["Safety risk", triage.get("safety_risk", "")],
        ["Immediate shutdown", "Yes" if triage.get("requires_immediate_shutdown") else "No"],
        ["Summary", triage.get("summary", "")], ["Rationale", triage.get("reasoning", "")],
    ], [40 * mm, 140 * mm], st))
    if result.get("triage_adjustment"):
        story.append(Paragraph(_clean(result["triage_adjustment"]), st["small"]))

    section("6. Diagnostic Assessment",
            Paragraph("<b>Primary hypothesis (requires verification):</b> " + _clean(diag.get("primary_hypothesis", "")), st["body"]),
            Paragraph("<b>Confidence:</b> " + _clean(diag.get("confidence", "")), st["body"]),
            Paragraph("Probable causes", st["h3"]), *_bullets(diag.get("probable_causes", []), st),
            Paragraph("Evidence", st["h3"]), *_bullets(diag.get("evidence", []), st),
            Paragraph("Recommended inspections", st["h3"]), *_bullets(diag.get("recommended_inspections", []), st),
            Paragraph("Additional data required", st["h3"]), *_bullets(diag.get("additional_data_required", []), st))

    section("7. Safety Assessment",
            Paragraph("<b>Risk level:</b> " + _clean(safety.get("risk_level", "")) +
                      " &nbsp; <i>AI-generated guidance - verify with site HSE procedures.</i>", st["body"]),
            Paragraph("Hazards", st["h3"]), *_bullets(safety.get("hazards", []), st),
            Paragraph("Required isolation (LOTO)", st["h3"]), *_bullets(safety.get("required_isolation", []), st),
            Paragraph("PPE", st["h3"]), *_bullets(safety.get("ppe", []), st),
            Paragraph("Permits", st["h3"]), *_bullets(safety.get("permit_requirements", []), st),
            Paragraph("Precautions", st["h3"]), *_bullets(safety.get("precautions", []), st),
            Paragraph("<b>Emergency action:</b> " + _clean(safety.get("emergency_action", "")), st["body"]))

    parts = plan_calc.get("parts", [])
    part_rows = [["Part no.", "Description", "Qty", "Stock", f"Unit cost ({CURRENCY})", "Availability"]] + [
        [p["part_number"], p["part_name"], p["quantity"], "-" if p["stock"] is None else p["stock"],
         f"{p['unit_cost']:,.0f}", p["availability"]] for p in parts]
    downtime, cost = plan_calc.get("downtime", {}), plan_calc.get("cost", {})
    section("8. Maintenance Plan",
            Paragraph("<b>Engineering decision:</b> " + _clean(f"{senior.get('decision', '')} - urgency: {senior.get('urgency', '')}"), st["body"]),
            Paragraph("<b>Assessment:</b> " + _clean(senior.get("engineering_assessment", "")), st["body"]),
            Paragraph("Inspection sequence", st["h3"]), *_bullets(plan.get("inspection_sequence", []), st),
            Paragraph("Maintenance actions", st["h3"]), *_bullets(plan.get("maintenance_actions", []), st),
            Paragraph("Tools required", st["h3"]), Paragraph(_clean(", ".join(plan.get("tools_required", [])) or "-"), st["body"]),
            Paragraph("Spare parts", st["h3"]),
            _table(part_rows, [26 * mm, 52 * mm, 12 * mm, 14 * mm, 26 * mm, 50 * mm], st, header=True) if parts
            else Paragraph("No spare parts identified.", st["body"]),
            Paragraph("Manpower", st["h3"]), *_bullets(plan.get("manpower", []), st),
            Paragraph(_clean(f"Estimated effort {plan.get('estimated_hours', 0):g} man-hours; estimated downtime "
                             f"{downtime.get('total_downtime_hours', 0):g} h ({downtime.get('basis', '')}); estimated cost "
                             f"{cost.get('total_cost', 0):,.0f} {CURRENCY} ({cost.get('basis', '')})."), st["body"]),
            Paragraph("Post-maintenance checks", st["h3"]), *_bullets(plan.get("post_maintenance_checks", []), st))

    section("9. Work Order", _table([
        ["WO number", wo.get("wo_number", "")], ["Title", wo.get("title", "")],
        ["Problem", wo.get("problem_description", "")],
        ["Priority / Department", f"{wo.get('priority', '')} / {wo.get('department', '')}"],
        ["Assigned team", wo.get("assigned_team", "")],
        ["Man-hours / Downtime", f"{wo.get('estimated_manhours', 0):g} h / {wo.get('estimated_downtime_hours', 0):g} h"],
        ["Estimated cost", f"{wo.get('estimated_cost', 0):,.0f} {CURRENCY}"], ["Status", wo.get("status", "")],
    ], [40 * mm, 140 * mm], st),
        Paragraph("Safety requirements", st["h3"]), *_bullets(wo.get("safety_requirements", []), st),
        Paragraph("Job steps", st["h3"]), *[Paragraph(_clean(f"{i}. {s}"), st["body"]) for i, s in enumerate(wo.get("job_steps", []), 1)],
        Paragraph("Acceptance criteria", st["h3"]), *_bullets(wo.get("acceptance_criteria", []), st))

    section("10. Root Cause Analysis (template - to be completed after inspection)",
            Paragraph("<b>Problem statement:</b> " + _clean(report.get("rca_problem_statement", "")), st["body"]),
            Paragraph("Possible root causes (to be verified)", st["h3"]), *_bullets(report.get("rca_possible_root_causes", []), st),
            Paragraph("5-Whys (provisional)", st["h3"]), *_bullets(report.get("rca_five_whys", []), st),
            Paragraph("Contributing factors", st["h3"]), *_bullets(report.get("rca_contributing_factors", []), st),
            Paragraph("Verification actions", st["h3"]), *_bullets(report.get("rca_verification_actions", []), st),
            Paragraph("Corrective actions", st["h3"]), *_bullets(report.get("corrective_actions", []), st),
            Paragraph("Preventive actions", st["h3"]), *_bullets(report.get("preventive_actions", []), st),
            Paragraph("Confirmed root cause: ______________________ &nbsp; Verified by: ____________ &nbsp; Date: ________", st["body"]))

    section("11. Recommended Follow-up", *_bullets(report.get("follow_up", []) + senior.get("next_steps", []), st))
    story.append(KeepTogether([Paragraph("12. Disclaimer", st["h2"]), Paragraph(_clean(DISCLAIMER), st["body"])]))

    doc.build(story, onFirstPage=_page_decor, onLaterPages=_page_decor)
    return buffer.getvalue()
