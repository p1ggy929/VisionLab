"""API 蓝图注册。"""

from __future__ import annotations


def register_blueprints(app) -> None:
    from . import (analysis, auth, camera, datasets, inference, media, models, projects,
                   reports, system)

    for module in (auth, projects, camera, datasets, models, inference, analysis,
                   reports, system, media):
        app.register_blueprint(module.bp)
