"""Deterministic engineering calculations (no LLM involved).

IMPORTANT: every threshold in this file is a DEMO / CONFIGURABLE value chosen
for this prototype. They are NOT universal engineering standards. Replace them
with OEM limits and site standards before any real-world use.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Iterable

SEVERITY_LEVELS = ["Low", "Medium", "High", "Critical"]
PRIORITY_BY_SEVERITY = {"Low": "P4", "Medium": "P3", "High": "P2", "Critical": "P1"}
CATEGORIES = ["Mechanical", "Electrical", "Instrumentation", "Process", "Lubrication", "Cooling", "Structural", "Other"]
DEPARTMENT_BY_CATEGORY = {
    "Mechanical": "Mechanical Maintenance",
    "Lubrication": "Mechanical Maintenance",
    "Cooling": "Mechanical Maintenance",
    "Electrical": "Electrical Maintenance",
    "Instrumentation": "Instrumentation & Control",
    "Process": "Operations / Process Engineering",
    "Structural": "Civil & Structural",
    "Other": "Maintenance Planning",
}

# ---- Demo cost / planning parameters (configurable) -------------------------
CURRENCY = "PKR"
LABOR_RATE_PER_HOUR = 2500.0  # blended technician rate, demo value
RUNNING_UTILISATION = 0.85  # fraction of calendar hours a machine runs, demo value

THRESHOLDS_NOTE = "Demo/configurable screening thresholds - not universal engineering standards."

# Vibration velocity zones (mm/s RMS). Demo values loosely modelled on common
# zone-based practice for medium-size machines. Configure per machine class.
VIBRATION_ZONES = [
    (2.8, "Zone A (good)", "Low"),
    (4.5, "Zone B (acceptable)", "Low"),
    (7.1, "Zone C (unsatisfactory)", "Medium"),
    (11.2, "Zone D (unacceptable)", "High"),
]
VIBRATION_CRITICAL = 11.2

# Per equipment-type demo limits. "temperature" is the main monitored temperature.
DEMO_THRESHOLDS: dict[str, dict[str, Any]] = {
    "default": {"temp_label": "Temperature", "temp_normal": 70.0, "temp_alarm": 85.0, "temp_trip": 100.0,
                "pressure_nominal": None, "oil_pressure_min": 1.5, "oil_temp_max": 80.0},
    "Centrifugal Pump": {"temp_label": "Bearing temperature", "temp_normal": 65.0, "temp_alarm": 80.0, "temp_trip": 95.0,
                         "pressure_nominal": 6.0, "oil_pressure_min": None, "oil_temp_max": 75.0},
    "Screw Compressor": {"temp_label": "Discharge temperature", "temp_normal": 85.0, "temp_alarm": 100.0, "temp_trip": 110.0,
                         "pressure_nominal": 7.5, "oil_pressure_min": 2.0, "oil_temp_max": 90.0},
    "Axial Cooling Fan": {"temp_label": "Bearing / gearbox temperature", "temp_normal": 60.0, "temp_alarm": 75.0, "temp_trip": 90.0,
                          "pressure_nominal": None, "oil_pressure_min": None, "oil_temp_max": 80.0},
    "Gas Engine": {"temp_label": "Jacket water temperature", "temp_normal": 85.0, "temp_alarm": 95.0, "temp_trip": 102.0,
                   "pressure_nominal": None, "oil_pressure_min": 3.0, "oil_temp_max": 95.0},
    "Wind Turbine": {"temp_label": "Gearbox bearing temperature", "temp_normal": 70.0, "temp_alarm": 85.0, "temp_trip": 95.0,
                     "pressure_nominal": None, "oil_pressure_min": 1.0, "oil_temp_max": 80.0},
    "Belt Conveyor": {"temp_label": "Drive bearing temperature", "temp_normal": 60.0, "temp_alarm": 75.0, "temp_trip": 90.0,
                      "pressure_nominal": None, "oil_pressure_min": None, "oil_temp_max": 80.0},
}


# ---------------------------------------------------------------------------- #
# Severity helpers
# ---------------------------------------------------------------------------- #
def severity_rank(level: str) -> int:
    return SEVERITY_LEVELS.index(level) if level in SEVERITY_LEVELS else 1


def max_severity(levels: Iterable[str]) -> str:
    levels = [lvl for lvl in levels if lvl in SEVERITY_LEVELS]
    return max(levels, key=severity_rank) if levels else "Low"


def normalize_severity(value: Any, default: str = "Medium") -> str:
    """Map free text to Low/Medium/High/Critical (most severe word wins - conservative)."""
    text = str(value or "").lower()
    for level in reversed(SEVERITY_LEVELS):
        if re.search(rf"\b{level.lower()}\b", text):
            return level
    return default


def normalize_category(value: Any) -> str:
    text = str(value or "").lower()
    for category in CATEGORIES:
        if category.lower() in text:
            return category
    return "Other"


def get_thresholds(equipment_type: str) -> dict[str, Any]:
    return {**DEMO_THRESHOLDS["default"], **DEMO_THRESHOLDS.get(equipment_type or "", {})}


def parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)[:19]).date()
    except ValueError:
        return None


# ---------------------------------------------------------------------------- #
# Individual checks
# ---------------------------------------------------------------------------- #
def classify_vibration(value: float | None) -> dict | None:
    if value is None:
        return None
    if value >= VIBRATION_CRITICAL:
        zone, level = "Beyond Zone D (severe)", "Critical"
    else:
        zone, level = next(((z, lvl) for limit, z, lvl in VIBRATION_ZONES if value < limit), ("Zone D (unacceptable)", "High"))
    return {"parameter": "Vibration", "value": f"{value:g} mm/s", "assessment": zone, "level": level}


def temperature_deviation(value: float | None, thr: dict) -> dict | None:
    if value is None:
        return None
    normal, alarm, trip = thr["temp_normal"], thr["temp_alarm"], thr["temp_trip"]
    deviation = value - normal
    if value >= trip:
        level, status = "Critical", f"at/above demo trip limit ({trip:g} C)"
    elif value >= alarm:
        level, status = "High", f"above demo alarm limit ({alarm:g} C)"
    elif value > normal:
        level, status = "Medium", f"above demo normal ({normal:g} C)"
    else:
        level, status = "Low", "within demo normal range"
    return {
        "parameter": thr["temp_label"],
        "value": f"{value:g} C",
        "assessment": f"{deviation:+.1f} C vs normal ({deviation / normal * 100:+.0f}%), {status}",
        "level": level,
        "deviation_c": round(deviation, 1),
    }


def pressure_deviation(value: float | None, nominal: float | None) -> dict | None:
    if value is None:
        return None
    if not nominal:
        return {"parameter": "Pressure", "value": f"{value:g} bar", "assessment": "No demo nominal defined for this type", "level": "Low"}
    pct = (value - nominal) / nominal * 100
    level = "High" if abs(pct) > 35 else "Medium" if abs(pct) > 20 else "Low"
    return {"parameter": "Pressure", "value": f"{value:g} bar", "assessment": f"{pct:+.0f}% vs demo nominal {nominal:g} bar",
            "level": level, "deviation_pct": round(pct, 1)}


def oil_pressure_check(value: float | None, minimum: float | None) -> dict | None:
    if value is None:
        return None
    if not minimum:
        return {"parameter": "Oil pressure", "value": f"{value:g} bar", "assessment": "No demo minimum defined", "level": "Low"}
    level = "High" if value < minimum else "Medium" if value < minimum * 1.15 else "Low"
    return {"parameter": "Oil pressure", "value": f"{value:g} bar",
            "assessment": f"demo minimum {minimum:g} bar" + (" - BELOW minimum" if value < minimum else ""), "level": level}


def oil_temperature_check(value: float | None, maximum: float | None) -> dict | None:
    if value is None:
        return None
    if not maximum:
        return {"parameter": "Oil temperature", "value": f"{value:g} C", "assessment": "No demo maximum defined", "level": "Low"}
    over = value - maximum
    level = "High" if over >= 10 else "Medium" if over > 0 else "Low"
    return {"parameter": "Oil temperature", "value": f"{value:g} C", "assessment": f"{over:+.1f} C vs demo max {maximum:g} C", "level": level}


def load_check(value: float | None) -> dict | None:
    if value is None:
        return None
    level = "Medium" if value > 100 else "Low"
    note = "above rated load" if value > 100 else "within rated load"
    return {"parameter": "Load", "value": f"{value:g} %", "assessment": note, "level": level}


def maintenance_timing(machine: dict, ref: date | None = None) -> dict:
    """Days since last maintenance, estimated running hours since, and overdue status."""
    ref = ref or date.today()
    last = parse_date(machine.get("last_maintenance"))
    nxt = parse_date(machine.get("next_maintenance"))
    out: dict[str, Any] = {"last_maintenance": str(last or "unknown"), "next_maintenance": str(nxt or "unknown")}
    if last:
        days = (ref - last).days
        out["days_since_maintenance"] = days
        out["est_running_hours_since_maintenance"] = round(days * 24 * RUNNING_UTILISATION)
    if nxt:
        overdue = (ref - nxt).days
        out["maintenance_overdue"] = overdue > 0
        out["days_overdue"] = max(overdue, 0)
        out["days_until_due"] = max(-overdue, 0)
    return out


def history_stats(history: list[dict]) -> dict:
    failure_types = [h.get("failure_type", "") for h in history]
    repeats = sorted({f for f in failure_types if f and failure_types.count(f) > 1})
    return {
        "events": len(history),
        "total_downtime_hours": round(sum(float(h.get("downtime_hours") or 0) for h in history), 1),
        "repeat_failure_types": repeats,
    }


def compute_indicators(readings: dict, machine: dict, ref: date | None = None, history: list[dict] | None = None) -> dict:
    """Run all deterministic checks and derive a severity floor for the triage step."""
    thr = get_thresholds(machine.get("type", ""))
    checks = [
        classify_vibration(readings.get("vibration_mm_s")),
        temperature_deviation(readings.get("temperature_c"), thr),
        pressure_deviation(readings.get("pressure_bar"), thr["pressure_nominal"]),
        oil_pressure_check(readings.get("oil_pressure_bar"), thr["oil_pressure_min"]),
        oil_temperature_check(readings.get("oil_temperature_c"), thr["oil_temp_max"]),
        load_check(readings.get("load_pct")),
    ]
    checks = [c for c in checks if c]
    floor = max_severity(c["level"] for c in checks) if checks else "Low"
    reasons = [f"{c['parameter']} {c['value']}: {c['assessment']}" for c in checks if c["level"] == floor and floor != "Low"]
    return {
        "thresholds_note": THRESHOLDS_NOTE,
        "checks": checks,
        "severity_floor": floor,
        "floor_reasons": reasons,
        "maintenance": maintenance_timing(machine, ref),
        "history": history_stats(history or []),
    }


# ---------------------------------------------------------------------------- #
# Planning calculations
# ---------------------------------------------------------------------------- #
def match_spare_parts(lines: list[dict], parts: list[dict]) -> list[dict]:
    """Match planner spare-part lines to the spare-parts database (stock, cost, lead time)."""
    by_number = {str(p["part_number"]).upper(): p for p in parts}
    matched: list[dict] = []
    seen: set[str] = set()
    for line in lines:
        number = str(line.get("part_number") or "").strip().upper()
        name = str(line.get("part_name") or "").strip()
        try:
            qty = max(1, int(round(float(line.get("quantity") or 1))))
        except (TypeError, ValueError):
            qty = 1
        part = by_number.get(number)
        if part is None:
            text = f"{number} {name}".upper()
            part = next((p for pn, p in by_number.items() if pn in text), None)
        if part is None and name:
            lname = name.lower()
            part = next((p for p in parts if lname in p["part_name"].lower() or p["part_name"].lower() in lname), None)
        if part is not None:
            if part["part_number"] in seen:
                continue
            seen.add(part["part_number"])
            stock, lead = int(part["stock"]), int(part["lead_time_days"])
            unit = float(part["unit_cost"])
            matched.append({
                "part_number": part["part_number"], "part_name": part["part_name"], "quantity": qty,
                "unit_cost": unit, "line_cost": unit * qty, "stock": stock, "lead_time_days": lead,
                "availability": "In stock" if stock >= qty else f"Insufficient stock - lead time {lead} d",
                "in_database": True, "note": str(line.get("note") or ""),
            })
        elif name or number:
            matched.append({
                "part_number": number or "N/A", "part_name": name or number, "quantity": qty,
                "unit_cost": 0.0, "line_cost": 0.0, "stock": None, "lead_time_days": None,
                "availability": "Not in spare-parts database - verify", "in_database": False, "note": str(line.get("note") or ""),
            })
    return matched


def estimate_downtime(planner_downtime: float, planner_hours: float, matched_parts: list[dict]) -> dict:
    repair = planner_downtime if planner_downtime and planner_downtime > 0 else planner_hours
    repair = round(max(float(repair or 0), 0.5), 1)
    waits = [p["lead_time_days"] * 24 for p in matched_parts
             if p["in_database"] and p["stock"] is not None and p["stock"] < p["quantity"]]
    parts_wait = float(max(waits)) if waits else 0.0
    return {
        "repair_downtime_hours": repair,
        "parts_wait_hours": parts_wait,
        "total_downtime_hours": round(repair + parts_wait, 1),
        "basis": "Planner repair estimate" + (" + longest lead time of out-of-stock parts" if parts_wait else "; all matched parts in stock"),
    }


def estimate_cost(manhours: float, matched_parts: list[dict]) -> dict:
    labour = round(max(float(manhours or 0), 0) * LABOR_RATE_PER_HOUR)
    parts = round(sum(p["line_cost"] for p in matched_parts))
    return {"currency": CURRENCY, "labour_cost": labour, "parts_cost": parts, "total_cost": labour + parts,
            "basis": f"Demo labour rate {LABOR_RATE_PER_HOUR:,.0f} {CURRENCY}/h + database unit costs"}


def severity_from_downtime(hours: float) -> str:
    """Used only to classify synthetic seed records."""
    return "High" if hours >= 12 else "Medium" if hours >= 5 else "Low"
