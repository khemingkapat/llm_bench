"""Importing this package imports every module inside it, which is what
triggers each module's @register("name") decorator. This is the only
reason a new technique file doesn't need to be wired into main.py by hand
-- just drop it in this folder.
"""

from importlib import import_module
from pathlib import Path

_pkg_dir = Path(__file__).parent
for _file in sorted(_pkg_dir.glob("*.py")):
    if _file.stem != "__init__":
        import_module(f"{__name__}.{_file.stem}")
