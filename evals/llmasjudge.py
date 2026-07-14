"""
judge_eval.py — Asama 3, Katman B: LLM-as-judge (anlam kontrolleri)

eval_results.json'u okur. Her senaryo icin ikinci bir LLM'e (judge) sistemin
ciktisini gosterir ve ANLAM gerektiren kontrolleri sorar:
  - note dolu ise DOGRU celiskiyi mi anlatiyor?
  - aciklama fonun GERCEK sayilariyla celisiyor mu? (yuksek vol'u "sakin" demek vb.)
  - dayanaksiz iddia var mi?
Judge {pass, reason} JSON dondurur (tool_choice ile zorlanir).

Calistirma:
    python evals/judge_eval.py
"""
import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# .env'deki ANTHROPIC_API_KEY'i yukle (explainer.py ile ayni mantik).
# Judge ayri bir script oldugu icin kendi basina yuklemeli.
load_dotenv(Path(__file__).parent.parent / ".env")

RESULTS_PATH = Path(__file__).parent / "eval_results.json"
JUDGE_OUT_PATH = Path(__file__).parent / "judge_results.json"

MODEL = "claude-sonnet-4-6"  # judge icin daha guclu model (ogrenci haiku'dan farkli olsun)
client = anthropic.Anthropic()

JUDGE_SYSTEM = """Sen bir TEFAS fon asistaninin ciktilarini denetleyen bir \
degerlendiricisin. Sana kullanicinin istegi, sistemin urettigi metin ve fonlarin \
GERCEK sayilari verilecek. Gorevin metnin dogru olup olmadigina karar vermek.

ONEMLI — SISTEMIN NASIL CALISTIGI (bunlari HATA SAYMA):
- Fon SECIMINI, risk/vade/tur FILTRELERINI deterministik bir motor yapar; LLM yapmaz.
  Bu yuzden "su fon neden secilmis", "tur filtresi dogru mu" gibi seyleri DENETLEME.
  Senin isin sadece METNIN dogrulugu.
- "Hisse agirlikli" tur secimi, motorda 'Degisken' VE 'Hisse Senedi' kategorilerini \
  kapsar. Yani listede 'Degisken' fon gormek NORMALDIR, tur uyumsuzlugu DEGILDIR. \
  Bunu asla hata sayma.
- Sayilar kullaniciya AYRI bir tabloda gosterilir. Metnin sayi icermemesi ve niteliksel \
  olmasi BEKLENEN davranistir; "su fonun maxDD'si metinde belirtilmemis" bir hata DEGILDIR.

SADECE su NET ihlalleri ara (kuçuk nuanslari, eksik detaylari KOVALAMA):
1. CELISKI KACIRMA: Kullanicinin notu yapilandirilmis secimleriyle AÇIKÇA celisiyorsa \
   (or. Agresif secip "cok dusuk oynaklik" istemek, kisa vade secip "uzun vadeli buyume" \
   istemek) ve sistem 'note' alanini BOS birakmissa -> HATA.
2. YANLIS CELISKI: Sistem 'note'u doldurmus ama ASIL celiskiden degil, baska bir seyden \
   bahsediyorsa (or. asil sorun vade celiskisiyken risk profilinden bahsetmek) -> HATA.
3. YANLIS ALARM: Gercek bir celiski YOKKEN sistem gereksiz uyari uretmisse -> HATA.
4. BARIZ SAYI CELISKISI: Bir fonun aciklamasi gercek sayisiyla ACIKCA celisiyorsa \
   (or. vol=%30 olan fonu "dusuk oynaklik/sakin" diye sunmak) -> HATA. \
   Sinirda durumlari (vol %15 "orta mi yuksek mi") hata SAYMA, sadece bariz olanlari.
5. DAYANAKSIZ IDDIA: Metin, verilerde KARSILIGI OLMAYAN somut bir iddiada bulunuyorsa \
   (or. getiri/enflasyon verisi yokken "enflasyonu yendi" demek) -> HATA.

Yukaridaki 5 net ihlalden HICBIRI yoksa pass=true. Varsa pass=false ve reason'da \
hangi ihlalin nasil oldugunu KISA yaz. Suphede kalirsan pass=true ver — sadece \
NET hatalari isaretle."""

JUDGE_TOOL = {
    "name": "submit_judgment",
    "description": "Senaryo degerlendirmesini gonder.",
    "input_schema": {
        "type": "object",
        "properties": {
            "passed": {"type": "boolean", "description": "Tum kriterler saglandi mi?"},
            "reason": {"type": "string", "description": "Kisa gerekce (ozellikle fail ise)."},
        },
        "required": ["passed", "reason"],
    },
}


def build_user_message(case: dict) -> str:
    """Judge'a gosterilecek tek mesaj: istek + metin + gercek sayilar."""
    inp = case["input"]
    out = case["output"]
    funds_lines = []
    for f in out.get("funds", []):
        vol = f.get("volatility")
        sharpe = f.get("sharpe")
        dd = f.get("max_drawdown")
        funds_lines.append(
            f"  - {f['code']}: vol={vol:.1%} sharpe={sharpe:.2f} maxDD={dd:.1%}\n"
            f"    aciklama: {f.get('explanation')}"
        )
    funds_block = "\n".join(funds_lines)

    return f"""KULLANICI ISTEGI:
  risk: {inp['risk']}
  vade: {inp['vade']}
  tur: {inp['tur']}
  not: {inp['user_note'] or '(bos)'}

SISTEMIN CIKTISI:
  summary: {out.get('summary')}
  note (uyari kutusu): {out.get('note') or '(bos)'}

FONLAR (gercek sayilar + sistemin aciklamasi):
{funds_block}

Yukaridaki ciktiyi kriterlere gore degerlendir ve submit_judgment'i cagir."""


def judge_one(case: dict) -> dict:
    if "error" in case:
        return {"id": case["id"], "passed": False, "reason": f"runner hatasi: {case['error']}"}

    resp = client.messages.create(
        model=MODEL, max_tokens=500, system=JUDGE_SYSTEM,
        tools=[JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "submit_judgment"},
        messages=[{"role": "user", "content": build_user_message(case)}],
    )
    block = next(b for b in resp.content if b.type == "tool_use")
    return {"id": case["id"],
            "passed": block.input.get("passed", False),
            "reason": block.input.get("reason", "")}


if __name__ == "__main__":
    cases = json.load(open(RESULTS_PATH, encoding="utf-8"))
    print(f"{len(cases)} senaryo judge'a gonderiliyor...\n")

    results = []
    for c in cases:
        r = judge_one(c)
        results.append(r)
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['id']}")
        if not r["passed"]:
            print(f"        -> {r['reason']}")

    with open(JUDGE_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    passed = sum(r["passed"] for r in results)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"JUDGE (Katman B) PASS RATE: {passed}/{total} = %{passed/total*100:.0f}")
    print(f"{'='*60}")