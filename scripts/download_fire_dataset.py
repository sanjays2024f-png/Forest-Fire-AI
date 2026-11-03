"""Download real Fire/Smoke/No-fire dataset (DeepQuestAI v1, ~3000 imgs)."""
from __future__ import annotations
import hashlib, json, shutil, sys, urllib.request, zipfile, random
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/DeepQuestAI/Fire-Smoke-Dataset/releases/download/v1/FIRE-SMOKE-DATASET.zip"
ZIP = ROOT / "dataset" / "_upstream" / "FIRE-SMOKE-DATASET.zip"
EXT = ROOT / "dataset" / "_upstream" / "extracted"
REP = ROOT / "outputs" / "dataset_report.json"
MAP = {"fire": "fire", "fires": "fire", "smoke": "smoke", "neutral": "no_fire",
 "no_fire": "no_fire", "nofire": "no_fire", "non_fire": "no_fire", "normal": "no_fire"}
def dl():
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    if ZIP.is_file() and ZIP.stat().st_size > 10_000_000:
        print(f"skip dl exists {ZIP.stat().st_size}"); return
    print(f"downloading 321MB -> {ZIP}")
    req = urllib.request.Request(URL, headers={"User-Agent": "forest-fire-ai"})
    with urllib.request.urlopen(req, timeout=120) as r, open(ZIP, "wb") as f:
        while True:
            c = r.read(1024*256)
            if not c: break
            f.write(c)
    print(f"done {ZIP.stat().st_size}")
def main():
    dl()
    if EXT.exists(): shutil.rmtree(EXT)
    EXT.mkdir(parents=True, exist_ok=True)
    print("extracting...")
    with zipfile.ZipFile(ZIP) as z: z.extractall(EXT)
    sufs = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    imgs = [p for p in EXT.rglob("*") if p.is_file() and p.suffix.lower() in sufs]
    print(f"found {len(imgs)} images")
    grp = defaultdict(list); unm = []
    for p in imgs:
        cls = None
        for part in [p.parent.name, p.parents[1].name if len(p.parents) > 1 else ""]:
            k = part.strip().lower().replace(" ", "_")
            if k in MAP: cls = MAP[k]; break
        if cls is None:
            low = str(p).lower()
            if "smoke" in low and "fire" not in p.parent.name.lower(): cls = "smoke"
            elif "fire" in low and "neutral" not in low: cls = "fire"
            elif "neutral" in low: cls = "no_fire"
        if cls is None: unm.append(str(p))
        else: grp[cls].append(p)
    print({k: len(v) for k, v in grp.items()}, "unmapped", len(unm))
    for s in ["train", "val", "test"]:
        for c in ["fire", "smoke", "no_fire"]:
            d = ROOT / "dataset" / s / c
            if d.exists(): shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
    for cls in ["fire", "smoke", "no_fire"]:
        fs = sorted(grp[cls]); random.Random(42).shuffle(fs)
        n = len(fs); a = int(n*0.70); b = int(n*0.15)
        for split, sub in [("train", fs[:a]), ("val", fs[a:a+b]), ("test", fs[a+b:])]:
            for src in sub:
                dst = ROOT / "dataset" / split / cls / src.name
                i = 1
                while dst.exists():
                    dst = ROOT / "dataset" / split / cls / f"{src.stem}_{i}{src.suffix}"; i += 1
                shutil.copy2(src, dst)
    from PIL import Image
    counts = {}; bad = []; seen = {}; dups = 0
    for s in ["train", "val", "test"]:
        for c in ["fire", "smoke", "no_fire"]:
            d = ROOT / "dataset" / s / c
            fl = list(d.glob("*")); counts[f"{s}/{c}"] = len(fl)
            for f in fl:
                try:
                    with Image.open(f) as im: im.verify()
                except Exception: bad.append(str(f.relative_to(ROOT)))
                try:
                    h = hashlib.sha256(f.read_bytes()).hexdigest()
                    if h in seen: dups += 1
                    else: seen[h] = 1
                except Exception: pass
    rep = {"source": "DeepQuestAI v1 https://github.com/DeepQuestAI/Fire-Smoke-Dataset/releases/tag/v1",
     "upstream_counts": {k: len(v) for k, v in grp.items()}, "split": "70/15/15 seed=42",
     "counts": counts, "total": sum(counts.values()), "corrupted_count": len(bad),
     "corrupted_sample": bad[:10], "duplicate_count": dups, "classes": ["fire", "smoke", "no_fire"]}
    REP.parent.mkdir(parents=True, exist_ok=True)
    REP.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    if rep["total"] < 30: sys.exit(1)
if __name__ == "__main__": main()
