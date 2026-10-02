"""SQLite persistence layer.

The database file is created automatically on first run and seeded with the
synthetic demo data from /data. No external database server is required.
Note: on Streamlit Community Cloud the file system is ephemeral, so records
reset when the app restarts - acceptable for a demo/prototype.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from utils.sample_data import build_seed_records

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = Path(os.getenv("COPILOT_DB_PATH", str(DATA_DIR / "copilot.db")))

OPEN_WO_STATUSES = ("Open", "In Progress", "Awaiting Parts")
CLOSED_REQUEST_STATUSES = ("Closed", "Completed", "Cancelled")

SCHEMA = """
CREATE TABLE IF NOT EXISTS machines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    equipment_id TEXT UNIQUE NOT NULL,
    name TEXT, type TEXT, plant_area TEXT, manufacturer TEXT, model TEXT,
    commission_date TEXT, runtime_hours REAL, status TEXT, criticality TEXT,
    last_maintenance TEXT, next_maintenance TEXT
);
CREATE TABLE IF NOT EXISTS maintenance_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_code TEXT UNIQUE,
    created_at TEXT, equipment_id TEXT, equipment_name TEXT, equipment_type TEXT,
    plant_area TEXT, operator TEXT, complaint TEXT, operating_condition TEXT,
    readings_json TEXT, category TEXT, severity TEXT, priority TEXT,
    status TEXT, source TEXT
);
CREATE TABLE IF NOT EXISTS work_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wo_number TEXT UNIQUE,
    request_id INTEGER, created_at TEXT, completed_at TEXT,
    equipment_id TEXT, equipment_name TEXT, title TEXT, problem_description TEXT,
    category TEXT, severity TEXT, priority TEXT, department TEXT, assigned_team TEXT,
    tools_json TEXT, spares_json TEXT, safety_json TEXT, steps_json TEXT, acceptance_json TEXT,
    estimated_manhours REAL, estimated_downtime_hours REAL, estimated_cost REAL,
    status TEXT, result_json TEXT
);
CREATE TABLE IF NOT EXISTS maintenance_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT, equipment_id TEXT, category TEXT, failure_type TEXT, symptoms TEXT,
    action_taken TEXT, downtime_hours REAL, maintenance_cost REAL, root_cause TEXT,
    recommendation TEXT, status TEXT
);
CREATE TABLE IF NOT EXISTS spare_parts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    part_number TEXT UNIQUE NOT NULL,
    part_name TEXT, category TEXT, compatible_equipment TEXT, stock INTEGER,
    unit_cost REAL, lead_time_days INTEGER, criticality TEXT
);
CREATE TABLE IF NOT EXISTS agent_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER, agent_name TEXT, status TEXT, started_at TEXT,
    duration_s REAL, summary TEXT, output_json TEXT, model TEXT
);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _loads(value: Any, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


# ---------------------------------------------------------------------------- #
# Setup & seeding
# ---------------------------------------------------------------------------- #
def init_db() -> None:
    """Create tables if missing and seed synthetic demo data on first run."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)
        if conn.execute("SELECT COUNT(*) FROM machines").fetchone()[0] == 0:
            _seed(conn)


def _seed(conn: sqlite3.Connection) -> None:
    machines = pd.read_csv(DATA_DIR / "machines.csv")
    history = pd.read_csv(DATA_DIR / "maintenance_history.csv")
    parts = pd.read_csv(DATA_DIR / "spare_parts.csv")
    machines.to_sql("machines", conn, if_exists="append", index=False)
    history.to_sql("maintenance_history", conn, if_exists="append", index=False)
    parts.to_sql("spare_parts", conn, if_exists="append", index=False)
    machine_map = {m["equipment_id"]: m for m in machines.to_dict("records")}
    for request, work_order in build_seed_records(history.to_dict("records"), machine_map):
        request_id, _ = _insert_request(conn, request)
        _insert_work_order(conn, work_order, request_id)


def reset_database() -> None:
    """Delete the database file and recreate it with fresh demo data."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    init_db()


# ---------------------------------------------------------------------------- #
# Machines, history, spare parts
# ---------------------------------------------------------------------------- #
def get_machines() -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query("SELECT * FROM machines ORDER BY equipment_id", conn)


def get_machine(equipment_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM machines WHERE equipment_id = ?", (equipment_id,)).fetchone()
    return dict(row) if row else None


def update_machine_status(equipment_id: str, status: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE machines SET status = ? WHERE equipment_id = ?", (status, equipment_id))


def get_history(equipment_id: str | None = None) -> pd.DataFrame:
    with connect() as conn:
        if equipment_id:
            return pd.read_sql_query(
                "SELECT * FROM maintenance_history WHERE equipment_id = ? ORDER BY date DESC", conn, params=(equipment_id,))
        return pd.read_sql_query("SELECT * FROM maintenance_history ORDER BY date DESC", conn)


def get_recent_history(equipment_id: str, limit: int = 6) -> list[dict]:
    """Only the most relevant (recent, same-equipment) records are sent to the LLM."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT date, failure_type, symptoms, action_taken, downtime_hours, root_cause, recommendation "
            "FROM maintenance_history WHERE equipment_id = ? ORDER BY date DESC LIMIT ?", (equipment_id, limit)).fetchall()
    return [dict(r) for r in rows]


def get_spare_parts() -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query("SELECT * FROM spare_parts ORDER BY part_number", conn)


def get_compatible_parts(equipment_id: str, limit: int = 15) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT part_number, part_name, category, stock, unit_cost, lead_time_days, criticality FROM spare_parts "
            "WHERE ',' || REPLACE(compatible_equipment, ' ', '') || ',' LIKE ? OR compatible_equipment = 'ALL' "
            "ORDER BY criticality, part_number LIMIT ?", (f"%,{equipment_id},%", limit)).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------- #
# Maintenance requests
# ---------------------------------------------------------------------------- #
def _insert_request(conn: sqlite3.Connection, data: dict) -> tuple[int, str]:
    cur = conn.execute(
        "INSERT INTO maintenance_requests (created_at, equipment_id, equipment_name, equipment_type, plant_area, operator, "
        "complaint, operating_condition, readings_json, category, severity, priority, status, source) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (data.get("created_at") or datetime.now().isoformat(timespec="seconds"), data["equipment_id"],
         data.get("equipment_name", ""), data.get("equipment_type", ""), data.get("plant_area", ""),
         data.get("operator", ""), data.get("complaint", ""), data.get("operating_condition", ""),
         _dumps(data.get("readings", {})), data.get("category", ""), data.get("severity", ""),
         data.get("priority", ""), data.get("status", "New"), data.get("source", "AI workflow")))
    request_id = cur.lastrowid
    year = str(data.get("created_at") or datetime.now().isoformat())[:4]
    code = f"REQ-{year}-{request_id:04d}"
    conn.execute("UPDATE maintenance_requests SET request_code = ? WHERE id = ?", (code, request_id))
    return request_id, code


def create_request(data: dict) -> tuple[int, str]:
    with connect() as conn:
        return _insert_request(conn, data)


def update_request(request_id: int, **fields: Any) -> None:
    allowed = {"category", "severity", "priority", "status"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return
    assignments = ", ".join(f"{k} = ?" for k in fields)
    with connect() as conn:
        conn.execute(f"UPDATE maintenance_requests SET {assignments} WHERE id = ?", (*fields.values(), request_id))


def get_requests() -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query("SELECT * FROM maintenance_requests ORDER BY id DESC", conn)


# ---------------------------------------------------------------------------- #
# Work orders
# ---------------------------------------------------------------------------- #
def _insert_work_order(conn: sqlite3.Connection, wo: dict, request_id: int | None) -> tuple[int, str]:
    created = wo.get("created_at") or datetime.now().isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT INTO work_orders (request_id, created_at, completed_at, equipment_id, equipment_name, title, "
        "problem_description, category, severity, priority, department, assigned_team, tools_json, spares_json, "
        "safety_json, steps_json, acceptance_json, estimated_manhours, estimated_downtime_hours, estimated_cost, status) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (request_id, created, wo.get("completed_at"), wo["equipment_id"], wo.get("equipment_name", ""),
         wo.get("title", ""), wo.get("problem_description", ""), wo.get("category", ""), wo.get("severity", ""),
         wo.get("priority", ""), wo.get("department", ""), wo.get("assigned_team", ""),
         _dumps(wo.get("tools_required", [])), _dumps(wo.get("spare_parts", [])),
         _dumps(wo.get("safety_requirements", [])), _dumps(wo.get("job_steps", [])),
         _dumps(wo.get("acceptance_criteria", [])), float(wo.get("estimated_manhours") or 0),
         float(wo.get("estimated_downtime_hours") or 0), float(wo.get("estimated_cost") or 0), wo.get("status", "Open")))
    wo_id = cur.lastrowid
    # Sequential, collision-free number derived from the SQLite autoincrement id.
    number = f"WO-{str(created)[:4]}-{wo_id:04d}"
    conn.execute("UPDATE work_orders SET wo_number = ? WHERE id = ?", (number, wo_id))
    return wo_id, number


def create_work_order(wo: dict, request_id: int | None) -> tuple[int, str]:
    with connect() as conn:
        return _insert_work_order(conn, wo, request_id)


def attach_workflow_result(wo_number: str, result: dict) -> None:
    with connect() as conn:
        conn.execute("UPDATE work_orders SET result_json = ? WHERE wo_number = ?", (_dumps(result), wo_number))


def get_work_orders() -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT wo_number, equipment_id, equipment_name, title, category, severity, priority, department, "
            "assigned_team, created_at, completed_at, status, estimated_manhours, estimated_downtime_hours, "
            "estimated_cost, (result_json IS NOT NULL) AS ai_generated FROM work_orders ORDER BY id DESC", conn)


def get_work_order(wo_number: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM work_orders WHERE wo_number = ?", (wo_number,)).fetchone()
    if not row:
        return None
    wo = dict(row)
    wo["tools_required"] = _loads(wo.pop("tools_json"), [])
    wo["spare_parts"] = _loads(wo.pop("spares_json"), [])
    wo["safety_requirements"] = _loads(wo.pop("safety_json"), [])
    wo["job_steps"] = _loads(wo.pop("steps_json"), [])
    wo["acceptance_criteria"] = _loads(wo.pop("acceptance_json"), [])
    wo["result"] = _loads(wo.pop("result_json"), None)
    return wo


def update_work_order_status(wo_number: str, status: str) -> None:
    completed = datetime.now().isoformat(timespec="seconds") if status == "Completed" else None
    request_status = {"Completed": "Closed", "In Progress": "In Progress", "Cancelled": "Cancelled"}.get(status, "Work Order Created")
    with connect() as conn:
        conn.execute("UPDATE work_orders SET status = ?, completed_at = ? WHERE wo_number = ?", (status, completed, wo_number))
        conn.execute(
            "UPDATE maintenance_requests SET status = ? WHERE id = (SELECT request_id FROM work_orders WHERE wo_number = ?)",
            (request_status, wo_number))


# ---------------------------------------------------------------------------- #
# Agent runs (audit trail)
# ---------------------------------------------------------------------------- #
def log_agent_run(request_id: int | None, agent_name: str, status: str, duration_s: float,
                  summary: str, output: Any, model: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO agent_runs (request_id, agent_name, status, started_at, duration_s, summary, output_json, model) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (request_id, agent_name, status, datetime.now().isoformat(timespec="seconds"), round(duration_s, 2),
             summary, _dumps(output), model))


def get_agent_runs(limit: int = 200) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT request_id, agent_name, status, started_at, duration_s, summary, model FROM agent_runs "
            "ORDER BY id DESC LIMIT ?", conn, params=(limit,))


# ---------------------------------------------------------------------------- #
# KPIs
# ---------------------------------------------------------------------------- #
def dashboard_stats() -> dict:
    with connect() as conn:
        q = lambda sql, *p: conn.execute(sql, p).fetchone()[0]  # noqa: E731
        placeholders_req = ",".join("?" * len(CLOSED_REQUEST_STATUSES))
        placeholders_wo = ",".join("?" * len(OPEN_WO_STATUSES))
        stats = {
            "total_machines": q("SELECT COUNT(*) FROM machines"),
            "open_requests": q(f"SELECT COUNT(*) FROM maintenance_requests WHERE status NOT IN ({placeholders_req})",
                               *CLOSED_REQUEST_STATUSES),
            "critical_open": q(f"SELECT COUNT(*) FROM maintenance_requests WHERE severity = 'Critical' "
                               f"AND status NOT IN ({placeholders_req})", *CLOSED_REQUEST_STATUSES),
            "active_work_orders": q(f"SELECT COUNT(*) FROM work_orders WHERE status IN ({placeholders_wo})", *OPEN_WO_STATUSES),
            "completed_work_orders": q("SELECT COUNT(*) FROM work_orders WHERE status = 'Completed'"),
            "ai_work_orders": q("SELECT COUNT(*) FROM work_orders WHERE result_json IS NOT NULL"),
            "avg_resolution_hours": q("SELECT AVG((julianday(completed_at) - julianday(created_at)) * 24) "
                                      "FROM work_orders WHERE status = 'Completed' AND completed_at IS NOT NULL") or 0.0,
            "maintenance_cost": q("SELECT COALESCE(SUM(maintenance_cost), 0) FROM maintenance_history"),
            "open_estimated_cost": q(f"SELECT COALESCE(SUM(estimated_cost), 0) FROM work_orders WHERE status IN ({placeholders_wo})",
                                     *OPEN_WO_STATUSES),
            "repeat_failures": q("SELECT COUNT(*) FROM (SELECT equipment_id, failure_type FROM maintenance_history "
                                 "GROUP BY equipment_id, failure_type HAVING COUNT(*) > 1)"),
        }
    return stats
