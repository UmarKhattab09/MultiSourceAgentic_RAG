"""Set an employee's sign-in password. Run: python src/auth.py E001"""
import sys
from getpass import getpass
from embeddings import ROOT

sys.path.insert(0, str(ROOT / "mcp_servers"))
import hr_server

if __name__ == "__main__":
    employee_id = sys.argv[1].upper()
    password = getpass(f"New password for {employee_id}: ")
    if len(password) < 8 or password != getpass("Repeat: "):
        sys.exit("Passwords must match and be at least 8 characters.")
    with hr_server.connect() as db:
        hr_server.employee(db, employee_id)
        hr_server.set_password(db, employee_id, password)
    print("Password updated.")
