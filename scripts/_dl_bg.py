"""Background resumable download + organize. Logs to outputs/_dl.log. Safe to relaunch."""
from __future__ import annotations
import hashlib, json, shutil, sys, urllib.request, zipfile, random, time
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/DeepQuestAI/Fire-Smoke-Dataset/releases/download/v1/FIRE-SMOKE-DATASET.zip"
ZIP = ROOT / "dataset" / "_upstream" / "FIRE-SMOKE-DATASET.zip"
EXT = ROOT / "dataset" / "_upstream" / "extracted"
REP = ROOT / "outputs" / "dataset_report.json"
LOG = ROOT / "outputs" / "_dl.log"
MAP = {"fire": "fire", "fires": "fire", "smoke": "smoke", "neutral": "no_fire",
 "no_fire": "no_fire", "nofire": "no_fire", "non_fire": "no_fire", "normal": "no_fire"}
def log(m):
    line = f"{time.strftime('%H:%M:%S')} {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f: f.write(line + "\n")
def download():
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    have = ZIP.stat().st_size if ZIP.is_file() else 0
    for attempt in range(12):
        try:
            req = urllib.request.Request(URL, headers={"User-Agent": "forest-fire-ai"})
            if have > 0:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=60) as r:
                code = r.status
                total = int(r.headers.get("Content-Range", "").split("/")[-1] or r.headers.get("Content-Length") or 0)
                mode = "ab" if code == 206 and have > 0 else "wb"
                if mode == "wb": have = 0
                log(f"attempt {attempt} http={code} have={have} total={total}")
                with open(ZIP, mode) as f:
                    while True:
                        c = r.read(1024*256)
                        if not c: break
                        f.write(c); have += len(c)
                        if have % (1024*1024*20) < 262144:
                            log(f"progress {have/1024/1024:.0f}MB" + (f"/{total/1024/1024:.0f}MB" if total else ""))
                if total and have < total:
                    log("short read, retrying"); time.sleep(3); continue
                log(f"download complete {have} bytes")
                return True
        except Exception as e:
            log(f"attempt {attempt} ERR {type(e).__name__}: {str(e)[:200]}")
            time.sleep(5)
    return False
def main():
    LOG.write_text("", encoding="utf-8")
    if not REP.exists():
        if not download(): log("DOWNLOAD FAILED"); return
        if EXT.exists(): shutil.rmtree(EXT)
        EXT.mkdir(parents=True, exist_ok=True)
        log("extracting...")
        with zipfile.ZipFile(ZIP) as z: z.extractall(EXT)
        sufs = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        imgs = [p for p in EXT.rglob("*") if p.is_file() and p.suffix.lower() in sufs]
        log(f"found {len(imgs)} images")
        grp = defaultdict(list); unm = 0
        for p in imgs:
            cls = None
            parts = [p.parent.name] + ([p.parents[1].name] if len(p.parents) > 1 else [])
            for part in parts:
                k = part.strip().lower().replace(" ", "_")
                if k in MAP: cls = MAP[k]; break
            if cls is None:
                low = str(p).lower()
                if "smoke" in low and "fire" not in p.parent.name.lower(): cls = "smoke"
                elif "fire" in low and "neutral" not in low: cls = "fire"
                elif "neutral" in low: cls = "no_fire"
            if cls is None: unm += 1
            else: grp[cls].append(p)
        log(f"grouped {dict((k, len(v)) for k, v in grp.items())} unmapped={unm}")
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
        log("REPORT: " + json.dumps(rep))
        log("DONE")
    else:
        log("report exists, nothing to do")
if __name__ == "__main__": main()
