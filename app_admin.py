"""Admin Flask entrypoint."""

from __future__ import annotations

import os

from web.app_factory import create_admin_app


create_app = create_admin_app
app = create_admin_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("ADMIN_HOST", "127.0.0.1"),
        port=int(os.getenv("ADMIN_PORT", "5000")),
        debug=app.config.get("DEBUG", False),
    )
