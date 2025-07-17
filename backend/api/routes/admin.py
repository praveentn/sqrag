# backend/api/routes/admin.py
from flask import Blueprint

admin = Blueprint("admin", __name__)

@admin.route("/admin")
def admin_index():
    return "Hello, Admin!"
