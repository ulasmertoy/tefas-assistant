"""
score_eval.py — Asama 3, Katman A: Python (kural) kontrolleri

eval_results.json'u okur, her senaryoya KESIN/deterministik kontroller uygular.
Sadece kurala dokulebilen seyler burada:
  - note dolu mu / bos mu (beklenene gore)
  - yasak metrik sayisi var mi
  - summary var mi
  - her fon aciklanmis mi
"Note celiskiyi DOGRU anlatmis mi", "aciklama mantikli mi" gibi ANLAM
gerektiren kontroller burada DEGIL -> judge katmaninin isi.

Calistirma:
    python evals/score_eval.py
"""
import json
import re
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "eval_results.json"

# Yasak olan: FON METRIGI (Sharpe, volatilite/getiri yuzdesi, ondalik).
# Masum sayilar (BIST 100, "son 1 yil") yasak DEGIL.
_METRIC_PATTERNS = [
    r"%\s*\d",
    r"\d\s*%",
    r"\d+[.,]\d+",
    r"(?:sharpe|volatilite|oynaklik|getiri|düşüş|dusus|drawdown)\s*[:=]?\s*\d",
]
_METRIC_RE = re.compile("|".join(_METRIC_PATTERNS), re.IGNORECASE)


def has_forbidden_number(text: str) -> bool:
    return bool(_METRIC_RE.search(text or ""))


def check_case(case: dict) -> dict:
    if "error" in case:
        return {"id": case["id"], "passed": False, "checks": {"runner_error": case["error"]}}

    exp = case["expected"]
    out = case["output"]
    note = out.get("note")
    summary = out.get("summary") or ""
    funds = out.get("funds") or []

    checks = {}

    note_filled = bool(note and str(note).strip())
    checks["note_state"] = (note_filled == exp["note_should_be_filled"])

    if exp.get("summary_should_exist"):
        checks["summary_exists"] = bool(summary.strip())

    if exp.get("all_funds_explained"):
        checks["all_explained"] = all(f.get("explanation") for f in funds)

    if exp.get("no_numbers_in_text"):
        texts = [summary] + [f.get("explanation", "") for f in funds]
        if note_filled:
            texts.append(str(note))
        checks["no_numbers"] = not any(has_forbidden_number(t) for t in texts)

    passed = all(checks.values())
    return {"id": case["id"], "passed": passed, "checks": checks}


if __name__ == "__main__":
    cases = json.load(open(RESULTS_PATH, encoding="utf-8"))
    results = [check_case(c) for c in cases]

    passed = sum(r["passed"] for r in results)
    total = len(results)

    print(f"\n{'='*60}")
    print(f"PYTHON KURAL KONTROLLERI (Katman A)")
    print(f"PASS RATE: {passed}/{total} = %{passed/total*100:.0f}")
    print(f"{'='*60}\n")

    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['id']}")
        if not r["passed"]:
            for name, ok in r["checks"].items():
                if ok is not True:
                    print(f"        x {name}: {ok}")