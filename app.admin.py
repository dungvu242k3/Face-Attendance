"""Backward-compatible admin entrypoint.

Prefer ``python app_admin.py``. This wrapper is kept so existing launch
commands continue to work.
"""

import os

from app_admin import app


if __name__ == "__main__":
    app.run(
        host=os.getenv("ADMIN_HOST", "127.0.0.1"),
        port=int(os.getenv("ADMIN_PORT", "5000")),
        debug=app.config.get("DEBUG", False),
    )
