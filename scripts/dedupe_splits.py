"""De-duplicate dataset splits by sha256 so no identical image leaks across train/val/test."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def main():
    seen: dict[str, Path] = {}
    removed = 0
    kept: dict[str, int] = {}
    # train wins, then val, then test
    for split in ["train", "val", "test"]:
        for cls in ["fire", "smoke", "no_fire"]:
            d = ROOT / "dataset" / split / cls
            for f in sorted(d.glob("*")):
                try:
                    h = hashlib.sha256(f.read_bytes()).hexdigest()
                except Exception:
                    continue
                if h in seen:
                    f.unlink()
                    removed += 1
                else:
                    seen[h] = f
                    kept[f"{split}/{cls}"] = kept.get(f"{split}/{cls}", 0) + 1
    total = sum(kept.values())
    rep_path = ROOT / "outputs" / "dataset_report.json"
    rep = json.loads(rep_path.read_text()) if rep_path.is_file() else {}
    rep["leakage_fix"] = {"duplicates_removed": removed, "counts_after": kept, "total_after": total}
    rep_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep["leakage_fix"], indent=2))
if __name__ == "__main__":
    main()
