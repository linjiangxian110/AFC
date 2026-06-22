"""Backward-compatible MID SDD audit entrypoint.

Use :mod:`trustmoe_traj.scripts.audit_sdd_prediction_bundle` for new commands.
"""

from __future__ import annotations

from trustmoe_traj.scripts.audit_sdd_prediction_bundle import main


if __name__ == "__main__":
    main()
