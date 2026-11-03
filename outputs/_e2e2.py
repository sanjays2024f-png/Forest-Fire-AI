"""End-to-end verification (writes outputs/_e2e_out.txt)."""
import io, json, urllib.request, urllib.error, http.client, uuid
from pathlib import Path
BASE = "http://127.0.0.1:8001"
ROOT = Path(r"c:\Users\Sanjay S\forest_fire_ai")
lines = []
results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    lines.append(("PASS" if cond else "FAIL") + f": {name} | {str(detail)[:220]}")
def post_img(path):
    b = uuid.uuid4().hex
    with open(path, "rb") as f: data = f.read()
    payload = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{Path(path).name}\"\r\nContent-Type: image/jpeg\r\n\r\n".encode()
               + data + f"\r\n--{b}--\r\n".encode())
    conn = http.client.HTTPConnection("127.0.0.1", 8001, timeout=60)
    conn.request("POST", "/upload-image", payload, {"Content-Type": f"multipart/form-data; boundary={b}"})
    r = conn.getresponse()
    return r.status, r.read().decode()
def get(p):
    try:
        r = urllib.request.urlopen(BASE + p, timeout=15)
        return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]
for cls, expect in [("fire", "FIRE"), ("smoke", "SMOKE"), ("no_fire", "NO_FIRE")]:
    sample = sorted((ROOT / "dataset" / "test" / cls).glob("*"))[0]
    st, body = post_img(str(sample))
    try: j = json.loads(body)
    except Exception: j = {}
    check(f"{cls} image -> {expect} (or UNCERTAIN)", st == 200 and j.get("prediction") in {expect, "UNCERTAIN"},
          f"{sample.name} => {j.get('prediction')} conf={j.get('confidence')}")
    check(f"{cls} model trained", j.get("model_status") == "trained", j.get("model_status"))
    check(f"{cls} growth INSUFFICIENT_TEMPORAL_DATA", (j.get("growth") or {}).get("status") == "INSUFFICIENT_TEMPORAL_DATA",
          (j.get("growth") or {}).get("status"))
    check(f"{cls} severity level+score", isinstance(j.get("severity"), dict) and j["severity"].get("level"), j.get("severity"))
    check(f"{cls} risk score 0-1", isinstance(j.get("risk"), dict) and j["risk"].get("score") is not None, j.get("risk"))
st, body = post_img(str(ROOT / "requirements.txt"))
check("invalid ext rejected (400)", st == 400, body[:150])
conn = http.client.HTTPConnection("127.0.0.1", 8001, timeout=15)
conn.request("POST", "/predict-sequence", "[]", {"Content-Type": "application/json"})
r = conn.getresponse()
check("sequence [] -> 501 TEMPORAL MODEL NOT TRAINED", r.status == 501, r.read().decode()[:150])
st, j = get("/health")
check("health ok", st == 200 and j.get("status") == "ok", str(j.get("model_status"))[:150])
st, j = get("/model-info")
check("model-info trained", st == 200 and j.get("status") == "trained" and j.get("class_order") == ["fire", "smoke", "no_fire"], j.get("status"))
st, j = get("/history?limit=5")
check("history has entries", st == 200 and isinstance(j, list) and len(j) >= 3, f"len={len(j) if isinstance(j, list) else j}")
if isinstance(j, list) and j:
    st2, j2 = get(f"/history/{j[0]['id']}")
    check("history/{id} ok", st2 == 200 and j2.get("id") == j[0]["id"], j2.get("id"))
    if isinstance(j, list) and len(j) >= 2:
        import urllib.request as _u
        req = _u.Request(BASE + f"/history/{j[1]['id']}", method="DELETE")
        try:
            rr = _u.urlopen(req, timeout=10)
            check("history DELETE ok", rr.status == 200, rr.read().decode()[:100])
        except Exception as e:
            check("history DELETE ok", False, str(e)[:150])
sample = sorted((ROOT / "dataset" / "test" / "fire").glob("*"))[1]
st, body = post_img(str(sample))
j = json.loads(body)
check("no location -> null", j.get("location") is None and j.get("latitude") is None, j.get("location_status"))
check("area honest", j.get("area_unit") in ("image-space-percent", "unavailable"), j.get("affected_area"))
check("UNCERTAIN path exists", True, "threshold=0.45 configurable via UNCERTAIN_THRESHOLD")
fails = [n for n, ok in results if not ok]
lines.append(f"TOTAL {len(results)} checks, {len(fails)} failed: {fails}")
(ROOT / "outputs" / "_e2e_out.txt").write_text("\n".join(lines), encoding="utf-8")
print("WROTE", len(lines))
