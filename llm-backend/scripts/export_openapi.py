from __future__ import annotations

import json
from pathlib import Path

from rag_app.main import app

target = Path(__file__).resolve().parents[1] / "openapi.json"
target.write_text(json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Wrote {target}")
