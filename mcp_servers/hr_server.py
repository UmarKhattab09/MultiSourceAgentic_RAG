"""Zava HR MCP server: leave balances and leave requests for the signed-in employee.

The employee always comes from the ZAVA_EMPLOYEE_ID environment variable, which the agent sets per chat session.
Tools never take an employee id from the model. HR_INSECURE_EMPLOYEE_ID=1 adds an employee_id argument instead:
that is the deliberately vulnerable baseline for the Phase 9 red-team eval, never use it for real.

Run:     python mcp_servers/hr_server.py
Inspect: npx @modelcontextprotocol/inspector -e ZAVA_EMPLOYEE_ID=E001 python mcp_servers/hr_server.py
"""
import hashlib
import hmac
import os
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DB_PATH = Path(os.getenv("HR_DB", Path(__file__).with_name("hr.db")))
LEAVE_TYPES = ("vacation", "sick", "personal")
HR_DEPARTMENT = "Human Resources"  # its members review leave requests
DEMO_PASSWORD = "zava-demo"  # shortcut: every seeded account starts with this, change it with `python src/auth.py E001`
EMPLOYEES = [  # fake Zava employees: id, name, department, vacation, sick, personal days
    ("E001", "Ava Chen", "Engineering", 15, 8, 3), ("E002", "Marcus Reed", "Engineering", 12, 10, 2),
    ("E003", "Priya Nair", "Product Management", 18, 6, 3), ("E004", "Liam O'Connor", "Sales", 10, 9, 1),
    ("E005", "Sofia Alvarez", "Marketing", 20, 10, 3), ("E006", "Noah Kim", "Operations", 8, 4, 2),
    ("E007", "Grace Okafor", "Human Resources", 22, 10, 3), ("E008", "Ethan Brooks", "Customer Service", 14, 7, 2),
    ("E009", "Mia Rossi", "Research and Development", 16, 9, 3), ("E010", "Lucas Martin", "Engineering", 11, 5, 1),
    ("E011", "Hana Sato", "Sales", 13, 8, 2), ("E012", "Daniel Weber", "Executive", 25, 10, 3),
    ("E013", "Chloe Dubois", "Marketing", 9, 10, 2), ("E014", "Omar Haddad", "Operations", 17, 6, 3),
    ("E015", "Isabella Costa", "Human Resources", 19, 9, 2),
]

def connect() -> sqlite3.Connection:
    """Open the database, creating and seeding it on first use."""
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS employees (id TEXT PRIMARY KEY, name TEXT, department TEXT);
        CREATE TABLE IF NOT EXISTS leave_balances (employee_id TEXT REFERENCES employees(id), leave_type TEXT,
            days_remaining INTEGER, PRIMARY KEY (employee_id, leave_type));
        CREATE TABLE IF NOT EXISTS leave_requests (id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id TEXT REFERENCES employees(id),
            leave_type TEXT, start_date TEXT, end_date TEXT, days INTEGER, reason TEXT, status TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS accounts (employee_id TEXT PRIMARY KEY REFERENCES employees(id), salt BLOB, password_hash BLOB);
    """)
    if not db.execute("SELECT 1 FROM employees").fetchone():
        for eid, name, dept, vac, sick, personal in EMPLOYEES:
            db.execute("INSERT INTO employees VALUES (?, ?, ?)", (eid, name, dept))
            for leave_type, days in zip(LEAVE_TYPES, (vac, sick, personal)):
                db.execute("INSERT INTO leave_balances VALUES (?, ?, ?)", (eid, leave_type, days))
        db.executemany("INSERT INTO leave_requests (employee_id, leave_type, start_date, end_date, days, reason, status, created_at) "
                       "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       [("E001", "vacation", "2026-08-03", "2026-08-07", 5, "Family trip", "approved", "2026-07-01T09:00:00"),
                        ("E002", "sick", "2026-09-14", "2026-09-15", 2, "Flu", "approved", "2026-09-14T08:30:00"),
                        ("E005", "vacation", "2026-12-21", "2026-12-24", 4, "Holidays", "pending", "2026-10-10T14:00:00")])
        db.commit()
    if not db.execute("SELECT 1 FROM accounts").fetchone():
        for eid, *_ in EMPLOYEES:
            set_password(db, eid, DEMO_PASSWORD)
        db.commit()
    return db

def hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)

def set_password(db: sqlite3.Connection, employee_id: str, password: str) -> None:
    salt = os.urandom(16)
    db.execute("INSERT OR REPLACE INTO accounts VALUES (?, ?, ?)", (employee_id, salt, hash_password(password, salt)))

def login(employee_id: str, password: str) -> dict | None:
    """The employee (id, name, department, is_hr) if the password is right, else None."""
    with connect() as db:
        row = db.execute("SELECT e.*, a.salt, a.password_hash FROM employees e JOIN accounts a ON a.employee_id = e.id "
                         "WHERE e.id = ?", (employee_id.strip().upper(),)).fetchone()
        if not row or not hmac.compare_digest(hash_password(password, row["salt"]), row["password_hash"]):
            return None
        return {"id": row["id"], "name": row["name"], "department": row["department"], "is_hr": row["department"] == HR_DEPARTMENT}

def employee(db: sqlite3.Connection, employee_id: str | None) -> sqlite3.Row:
    if not employee_id:
        raise ValueError("No employee is signed in to this session.")
    row = db.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
    if not row:
        raise ValueError(f"Unknown employee {employee_id}.")
    return row

def weekdays(start: date, end: date) -> int:
    return sum((start + timedelta(d)).weekday() < 5 for d in range((end - start).days + 1))

def leave_balance(employee_id: str | None) -> dict:
    with connect() as db:
        emp = employee(db, employee_id)
        rows = db.execute("SELECT leave_type, days_remaining FROM leave_balances WHERE employee_id = ?", (emp["id"],))
        return {"employee": emp["name"], "balances": {r["leave_type"]: r["days_remaining"] for r in rows}}

def leave_request(employee_id: str | None, leave_type: str, start_date: str, end_date: str, reason: str = "") -> dict:
    with connect() as db:
        emp = employee(db, employee_id)
        if leave_type not in LEAVE_TYPES:
            raise ValueError(f"leave_type must be one of {LEAVE_TYPES}.")
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
        days = weekdays(start, end)
        if start < date.today():
            raise ValueError(f"start_date {start_date} is in the past; today is {date.today()}.")
        if end < start or days == 0:
            raise ValueError("The request must cover at least one weekday, with end_date on or after start_date.")
        balance = db.execute("SELECT days_remaining FROM leave_balances WHERE employee_id = ? AND leave_type = ?",
                             (emp["id"], leave_type)).fetchone()["days_remaining"]
        if days > balance:
            raise ValueError(f"Not enough {leave_type} days: requested {days}, remaining {balance}.")
        db.execute("UPDATE leave_balances SET days_remaining = days_remaining - ? WHERE employee_id = ? AND leave_type = ?",
                   (days, emp["id"], leave_type))
        cur = db.execute("INSERT INTO leave_requests (employee_id, leave_type, start_date, end_date, days, reason, status, created_at) "
                         "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
                         (emp["id"], leave_type, start_date, end_date, days, reason, datetime.now().isoformat(timespec="seconds")))
        return {"request_id": cur.lastrowid, "status": "pending", "days": days, "remaining": balance - days}

def request_status(employee_id: str | None, request_id: int) -> dict:
    with connect() as db:
        emp = employee(db, employee_id)
        row = db.execute("SELECT * FROM leave_requests WHERE id = ? AND employee_id = ?", (request_id, emp["id"])).fetchone()
        if not row:  # also hides other employees' requests
            raise ValueError(f"No leave request {request_id} found for {emp['name']}.")
        return dict(row)

def my_requests(employee_id: str | None) -> list[dict]:
    with connect() as db:
        emp = employee(db, employee_id)
        return [dict(r) for r in db.execute("SELECT * FROM leave_requests WHERE employee_id = ? ORDER BY id DESC", (emp["id"],))]

def pending_requests(reviewer_id: str | None) -> list[dict]:
    """Every pending request, for HR only."""
    with connect() as db:
        hr_reviewer(db, reviewer_id)
        return [dict(r) for r in db.execute("SELECT r.*, e.name, e.department FROM leave_requests r "
                                            "JOIN employees e ON e.id = r.employee_id WHERE r.status = 'pending' ORDER BY r.start_date")]

def decide_request(reviewer_id: str | None, request_id: int, approve: bool) -> dict:
    """HR approves or rejects a pending request; a rejection gives the days back."""
    with connect() as db:
        reviewer = hr_reviewer(db, reviewer_id)
        row = db.execute("SELECT * FROM leave_requests WHERE id = ? AND status = 'pending'", (request_id,)).fetchone()
        if not row:
            raise ValueError(f"No pending leave request {request_id}.")
        if row["employee_id"] == reviewer["id"]:
            raise ValueError("You cannot review your own leave request.")
        status = "approved" if approve else "rejected"
        db.execute("UPDATE leave_requests SET status = ? WHERE id = ?", (status, request_id))
        if not approve:
            db.execute("UPDATE leave_balances SET days_remaining = days_remaining + ? WHERE employee_id = ? AND leave_type = ?",
                       (row["days"], row["employee_id"], row["leave_type"]))
        return {"request_id": request_id, "status": status}

def hr_reviewer(db: sqlite3.Connection, reviewer_id: str | None) -> sqlite3.Row:
    reviewer = employee(db, reviewer_id)
    if reviewer["department"] != HR_DEPARTMENT:
        raise PermissionError("Only HR can review leave requests.")
    return reviewer

mcp = FastMCP("zava-hr", log_level="WARNING")
INSECURE = os.getenv("HR_INSECURE_EMPLOYEE_ID") == "1"

if INSECURE:
    @mcp.tool()
    def get_leave_balance(employee_id: str) -> dict:
        """Get the remaining vacation, sick and personal leave days of an employee."""
        return leave_balance(employee_id)

    @mcp.tool()
    def submit_leave_request(employee_id: str, leave_type: str, start_date: str, end_date: str, reason: str = "") -> dict:
        """Submit a leave request for an employee. leave_type: vacation, sick or personal. Dates as YYYY-MM-DD."""
        return leave_request(employee_id, leave_type, start_date, end_date, reason)

    @mcp.tool()
    def get_request_status(employee_id: str, request_id: int) -> dict:
        """Get the status of an employee's leave request by its id."""
        return request_status(employee_id, request_id)
else:
    @mcp.tool()
    def get_leave_balance() -> dict:
        """Get the signed-in employee's remaining vacation, sick and personal leave days."""
        return leave_balance(os.getenv("ZAVA_EMPLOYEE_ID"))

    @mcp.tool()
    def submit_leave_request(leave_type: str, start_date: str, end_date: str, reason: str = "") -> dict:
        """Submit a leave request for the signed-in employee. leave_type: vacation, sick or personal. Dates as YYYY-MM-DD."""
        return leave_request(os.getenv("ZAVA_EMPLOYEE_ID"), leave_type, start_date, end_date, reason)

    @mcp.tool()
    def get_request_status(request_id: int) -> dict:
        """Get the status of one of the signed-in employee's leave requests by its id."""
        return request_status(os.getenv("ZAVA_EMPLOYEE_ID"), request_id)

if __name__ == "__main__":
    mcp.run()
