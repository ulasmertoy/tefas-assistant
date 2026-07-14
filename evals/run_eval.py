"""
run_eval.py — Eval runner (Asama 2)

Her senaryoyu gercek sisteme sokar (build_profile -> build_response ->
explain_selected -> merge_by_code), ciktiyi (metin + SAYILAR) toplar,
eval_results.json'a yazar. Not vermez.

Calistirma (repo kokunden):
    python evals/run_eval.py
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import recommend as rc
from explainer import explain_selected, merge_by_code

CASES_PATH = Path(__file__).parent / "eval_cases.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"
FEATURES_PATH = Path(__file__).parent.parent / "data" / "processed" / "funds_features.parquet"

TOP_N = 12


def load_cases() -> list[dict]:
    with open(CASES_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_one(case: dict, features) -> dict:
    inp = case["input"]
    profile = rc.build_profile(inp["risk"], inp["vade"], inp["tur"])
    rec = rc.build_response(features, profile=profile, top_n=TOP_N)
    exp = explain_selected(rec, user_note=inp["user_note"])

    # merge_by_code: sayilar (motor) + metin (LLM) tek kartta birlesir.
    # Boylece judge fonun GERCEK sayilarini da gorebilir.
    cards = merge_by_code(rec, exp)

    return {
        "id": case["id"],
        "input": inp,
        "expected": case["expected"],
        "output": {
            "summary": exp.summary,
            "note": exp.note,
            "funds": [{
                "code": c["code"],
                "explanation": c["explanation"],
                "volatility": c["volatility"],
                "sharpe": c["sharpe"],
                "max_drawdown": c["max_drawdown"],
            } for c in cards],
        },
    }


if __name__ == "__main__":
    cases = load_cases()
    features = rc.load_features(str(FEATURES_PATH))
    print(f"{len(cases)} senaryo calistiriliyor...\n")

    results = []
    for c in cases:
        print(f"  -> {c['id']}")
        try:
            results.append(run_one(c, features))
        except Exception as e:
            print(f"    HATA: {e}")
            results.append({"id": c["id"], "input": c["input"],
                            "expected": c["expected"], "error": str(e)})

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{len(results)} sonuc yazildi: {RESULTS_PATH}")