import os
import sys

# Ensure current working directory and module resolution point to krish_project
base_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(base_dir, "krish_project")

if os.path.exists(project_dir):
    os.chdir(project_dir)
    sys.path.insert(0, project_dir)
    app_target = os.path.join(project_dir, "app.py")
    with open(app_target, "r", encoding="utf-8") as f:
        code = compile(f.read(), app_target, "exec")
    exec(code, globals())
