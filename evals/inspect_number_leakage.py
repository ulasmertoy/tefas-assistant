import json, re
from pathlib import Path

_METRIC_PATTERNS = [
    r"%\s*\d", r"\d\s*%", r"\d+[.,]\d+",
    r"(?:sharpe|volatilite|oynaklik|getiri|düşüş|dusus|drawdown)\s*[:=]?\s*\d",
]
_RE = re.compile("|".join(_METRIC_PATTERNS), re.IGNORECASE)

cases = json.load(open(Path(__file__).parent / "eval_results.json", encoding="utf-8"))
target = {"conflict_04_kisa_vade_uzun_vadeli_buyume",
          "conflict_09_dengeli_hic_zarar_etmesin",
          "aligned_18_dengeli_istikrarli_hisse"}

for c in cases:
    if c["id"] not in target:
        continue
    print("="*50)
    print(c["id"])
    out = c["output"]
    texts = {"summary": out.get("summary") or ""}
    if out.get("note"):
        texts["note"] = out["note"]
    for f in out.get("funds", []):
        texts[f"fon[{f['code']}]"] = f.get("explanation", "")
    for where, t in texts.items():
        for m in _RE.finditer(t):
            # eslesmenin etrafindan biraz baglam goster
            s = max(0, m.start()-25); e = min(len(t), m.end()+25)
            print(f"  {where}: ...{t[s:e]}...  [yakalanan: '{m.group()}']")