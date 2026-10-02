"""Synthetic demo data: predefined demo cases and seed work orders.

All equipment, readings and records are SIMULATED for demonstration only.
They are not connected to any real industrial equipment.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from utils import calculations as calc

DEMO_NOTICE = "Demo environment using synthetic equipment and maintenance data."

DEMO_CASES = [
    {
        "key": "pump",
        "label": "Case 1 - Pump P-102 high vibration",
        "equipment_id": "P-102",
        "operator": "Operator - Shift B",
        "complaint": "Pump vibration increased significantly over the last two shifts and drive-end bearing temperature "
                     "reached 92 C. Operator reports a rumbling noise from the drive-end bearing housing.",
        "operating_condition": "High load",
        "load_pct": 85.0,
        "temperature_c": 92.0,
        "pressure_bar": 5.4,
        "vibration_mm_s": 8.2,
        "oil_pressure_bar": None,
        "oil_temperature_c": 78.0,
        "additional_readings": "Discharge pressure fluctuating +/-0.3 bar. Motor current steady at 41 A. No visible leakage.",
    },
    {
        "key": "compressor",
        "label": "Case 2 - Compressor M-201 high discharge temperature",
        "equipment_id": "M-201",
        "operator": "Operator - Shift A",
        "complaint": "Compressor discharge temperature has risen steadily over three days and is now 108 C with the "
                     "high-temperature pre-alarm active. Compressor is unloading more frequently than normal.",
        "operating_condition": "High load",
        "load_pct": 92.0,
        "temperature_c": 108.0,
        "pressure_bar": 7.2,
        "vibration_mm_s": 3.1,
        "oil_pressure_bar": 2.4,
        "oil_temperature_c": 88.0,
        "additional_readings": "Ambient temperature 41 C. Cooler fan running. Oil level at lower mark of sight glass.",
    },
    {
        "key": "fan",
        "label": "Case 3 - Cooling fan F-301 abnormal noise",
        "equipment_id": "F-301",
        "operator": "Operator - Shift C",
        "complaint": "Abnormal intermittent knocking noise from the cooling tower fan, noticeable during start-up and at "
                     "full speed. Routine vibration route shows a slight increase.",
        "operating_condition": "Normal operation",
        "load_pct": 100.0,
        "temperature_c": 68.0,
        "pressure_bar": None,
        "vibration_mm_s": 5.1,
        "oil_pressure_bar": None,
        "oil_temperature_c": 72.0,
        "additional_readings": "Gearbox oil level OK at last check. Light deposits possibly seen on one blade tip during "
                               "walk-down (not confirmed).",
    },
]

# Open / in-progress synthetic work orders so the dashboard is not empty on first run.
SEED_OPEN_ITEMS = [
    {
        "equipment_id": "WTG-01", "days_ago": 4, "category": "Mechanical", "severity": "Critical",
        "complaint": "Gearbox high-speed bearing temperature trip; turbine stopped automatically.",
        "title": "Inspect gearbox HSS bearing after temperature trip", "status": "Awaiting Parts",
        "team": "Mechanical crew B + OEM technician", "hours": 16, "downtime": 36,
    },
    {
        "equipment_id": "GE-01", "days_ago": 2, "category": "Lubrication", "severity": "High",
        "complaint": "Lube oil pressure trending down to 3.4 bar at full load.",
        "title": "Investigate low lube oil pressure trend", "status": "In Progress",
        "team": "Mechanical crew A", "hours": 6, "downtime": 4,
    },
    {
        "equipment_id": "CV-401", "days_ago": 1, "category": "Mechanical", "severity": "Medium",
        "complaint": "Belt drifting towards the north side near the tail pulley.",
        "title": "Belt tracking adjustment and idler check", "status": "Open",
        "team": "Mechanical crew C", "hours": 4, "downtime": 3,
    },
    {
        "equipment_id": "P-101", "days_ago": 6, "category": "Mechanical", "severity": "Low",
        "complaint": "Minor seal weep observed during operator round (about 2 drops/min).",
        "title": "Monitor and plan seal inspection", "status": "Open",
        "team": "Mechanical crew A", "hours": 2, "downtime": 0,
    },
]


def build_seed_records(history: list[dict], machines: dict[str, dict], now: datetime | None = None) -> list[tuple[dict, dict]]:
    """Create (request, work_order) pairs from synthetic history + open items."""
    now = now or datetime.now()
    records: list[tuple[dict, dict]] = []
    for h in history:
        created = datetime.fromisoformat(f"{h['date']}T08:00:00")
        downtime = float(h["downtime_hours"])
        severity = calc.severity_from_downtime(downtime)
        category = h.get("category") or "Mechanical"
        machine = machines.get(h["equipment_id"], {})
        request = {
            "created_at": created.isoformat(timespec="seconds"), "equipment_id": h["equipment_id"],
            "equipment_name": machine.get("name", ""), "equipment_type": machine.get("type", ""),
            "plant_area": machine.get("plant_area", ""), "operator": "Synthetic record",
            "complaint": h["symptoms"], "operating_condition": "", "readings": {},
            "category": category, "severity": severity, "priority": calc.PRIORITY_BY_SEVERITY[severity],
            "status": "Closed", "source": "Synthetic seed",
        }
        work_order = {
            "created_at": request["created_at"], "equipment_id": h["equipment_id"],
            "equipment_name": machine.get("name", ""), "title": h["failure_type"],
            "problem_description": h["symptoms"], "category": category, "severity": severity,
            "priority": request["priority"], "department": calc.DEPARTMENT_BY_CATEGORY.get(category, "Maintenance Planning"),
            "assigned_team": "Maintenance crew", "tools_required": [], "spare_parts": [], "safety_requirements": [],
            "job_steps": [h["action_taken"]], "acceptance_criteria": [], "estimated_manhours": downtime * 1.5,
            "estimated_downtime_hours": downtime, "estimated_cost": float(h["maintenance_cost"]),
            "status": "Completed",
            "completed_at": (created + timedelta(hours=downtime + 4)).isoformat(timespec="seconds"),
        }
        records.append((request, work_order))

    for item in SEED_OPEN_ITEMS:
        created = (now - timedelta(days=item["days_ago"])).replace(minute=0, second=0, microsecond=0)
        machine = machines.get(item["equipment_id"], {})
        severity = item["severity"]
        request = {
            "created_at": created.isoformat(timespec="seconds"), "equipment_id": item["equipment_id"],
            "equipment_name": machine.get("name", ""), "equipment_type": machine.get("type", ""),
            "plant_area": machine.get("plant_area", ""), "operator": "Synthetic record",
            "complaint": item["complaint"], "operating_condition": "", "readings": {},
            "category": item["category"], "severity": severity, "priority": calc.PRIORITY_BY_SEVERITY[severity],
            "status": "Work Order Created", "source": "Synthetic seed",
        }
        work_order = {
            "created_at": request["created_at"], "equipment_id": item["equipment_id"],
            "equipment_name": machine.get("name", ""), "title": item["title"],
            "problem_description": item["complaint"], "category": item["category"], "severity": severity,
            "priority": request["priority"], "department": calc.DEPARTMENT_BY_CATEGORY[item["category"]],
            "assigned_team": item["team"], "tools_required": [], "spare_parts": [], "safety_requirements": [],
            "job_steps": [], "acceptance_criteria": [], "estimated_manhours": float(item["hours"]),
            "estimated_downtime_hours": float(item["downtime"]),
            "estimated_cost": float(item["hours"]) * calc.LABOR_RATE_PER_HOUR,
            "status": item["status"], "completed_at": None,
        }
        records.append((request, work_order))
    return records
