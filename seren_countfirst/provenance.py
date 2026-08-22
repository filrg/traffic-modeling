from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone


def environment() -> dict:
    """Return the interpreter and library versions a result was produced under."""
    import numpy
    import pandas
    import scipy

    from . import __version__

    return {
        "package": __version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "platform": f"{platform.system()} {platform.release()}",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "argv": " ".join(sys.argv[1:]),
    }
