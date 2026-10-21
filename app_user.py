"""Employee-facing Flask entrypoint.

Use ``python app_user.py`` for local development or import ``app`` from a
WSGI server.
"""

from __future__ import annotations

import os

from web.app_factory import create_user_app


create_app = create_user_app
app = create_user_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("USER_HOST", "127.0.0.1"),
        port=int(os.getenv("USER_PORT", "8000")),
        debug=app.config.get("DEBUG", False),
    )
