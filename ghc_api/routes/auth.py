"""
Auth-related routes:

  GET  /signup                         self-signup form (public)
  POST /signup                         create pending user, return token

  GET  /api/users                      list all users (admin: nginx-gated)
  POST /api/users/<user_id>/approve
  POST /api/users/<user_id>/revoke
  DELETE /api/users/<user_id>

The /api/users/* endpoints are NOT protected at the Flask layer. The dashboard
itself is expected to be behind nginx basic-auth (or equivalent) in production,
and these endpoints inherit the same gate. In a dev / single-machine deployment,
the operator is already implicitly the admin.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from ..auth import (
    STATUS_APPROVED,
    STATUS_REVOKED,
    get_user_registry,
)


auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/signup", methods=["GET"])
def signup_page():
    return render_template("signup.html")


@auth_bp.route("/signup", methods=["POST"])
def signup_submit():
    # Accept both form-urlencoded (HTML form post) and JSON (programmatic).
    if request.is_json:
        payload = request.get_json(silent=True) or {}
    else:
        payload = request.form

    user_id = (payload.get("user_id") or "").strip()
    display_name = (payload.get("display_name") or "").strip()

    record, err = get_user_registry().create_pending(user_id, display_name)
    if err is not None:
        return jsonify({"error": "signup_failed", "message": err}), 400

    return jsonify({
        "user_id": record.user_id,
        "display_name": record.display_name,
        "token": record.token,
        "status": record.status,
        "message": "Account registered. Your token will be usable once an administrator approves it.",
    }), 201


@auth_bp.route("/api/users", methods=["GET"])
def list_users():
    records = get_user_registry().list_all()
    return jsonify({
        "users": [r.to_public_dict() for r in records],
    })


@auth_bp.route("/api/users/<user_id>/approve", methods=["POST"])
def approve_user(user_id: str):
    record, err = get_user_registry().set_status(user_id, STATUS_APPROVED)
    if err is not None:
        return jsonify({"error": "approve_failed", "message": err}), 404 if "not found" in err else 400
    return jsonify({"user": record.to_public_dict()})


@auth_bp.route("/api/users/<user_id>/revoke", methods=["POST"])
def revoke_user(user_id: str):
    record, err = get_user_registry().set_status(user_id, STATUS_REVOKED)
    if err is not None:
        return jsonify({"error": "revoke_failed", "message": err}), 404 if "not found" in err else 400
    return jsonify({"user": record.to_public_dict()})


@auth_bp.route("/api/users/<user_id>", methods=["DELETE"])
def delete_user(user_id: str):
    ok, err = get_user_registry().delete(user_id)
    if not ok:
        return jsonify({"error": "delete_failed", "message": err}), 404
    return jsonify({"deleted": user_id})
