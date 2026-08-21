#!/usr/bin/env python3
"""Sinh các bảng của `submission/REPORT.md` TRỰC TIẾP từ `results/`.

Vì sao có file này. Tiêu chí 4.3 của rubric là "mọi con số trong report khớp với file
trong results/", và grader kiểm tra chéo đúng điều đó. Chép tay bốn bảng từ sáu file
JSON là cách chắc chắn nhất để một chữ số bị lệch — và một chữ số lệch làm hỏng cả mục,
kể cả khi pipeline chạy đúng. Ở đây bảng được sinh ra từ chính artefact, nên hai bên
không thể lệch nhau.

Script này KHÔNG ghi đè `REPORT.md`. Nó in ra (và ghi vào `results/REPORT_BLOCKS.md`)
các khối đã sẵn sàng dán, để phần văn xuôi — thứ thật sự được chấm — vẫn do bạn viết.

    python scripts/fill_report.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MISSING = "—"


def load_json(name: str):
    p = RESULTS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"canh bao: {name} hong: {exc}", file=sys.stderr)
        return None


def load_runs() -> list[dict]:
    p = RESULTS / "runs.csv"
    if not p.exists():
        return []
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def num(value, digits: int = 3) -> str:
    """Format nếu là số, còn không thì trả về dấu thiếu — không bao giờ bịa."""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return MISSING


def gap(name: str) -> str:
    return f"> **Thiếu `results/{name}`** — chạy notebook sinh ra nó rồi chạy lại script này.\n"


def wrong_fields(pred: str, label: dict) -> str:
    """Những trường model đoán SAI — trả lời thẳng câu 'ca thua có mẫu chung không?'.

    Dùng đúng bộ giải mã JSON của `labkit.evaluate`, nên 'sai' ở đây khớp với điểm
    số trong `qualitative.json` thay vì là một phép so sánh thứ hai, lỏng hơn.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from labkit import evaluate as ev

    obj = ev._parse_json_loose(pred)
    if not isinstance(obj, dict):
        return "(không parse được JSON)"
    bad = []
    for k in ev.TRIAGE_KEYS:
        got, want = obj.get(k), label.get(k)
        if want is None:
            continue
        if got is None:
            bad.append(k)
        elif k == "product":                       # so sau khi bỏ dấu, như scorer
            if ev.normalize(str(got)) != ev.normalize(str(want)):
                bad.append(k)
        elif str(got).strip().lower() != str(want).strip().lower():
            bad.append(k)
    return ", ".join(bad) if bad else "—"


def section_setup(out: list[str]) -> None:
    proof = load_json("mask_proof.json")
    stats = load_json("token_stats.json")
    tmpl = load_json("template_check.json")
    frozen = load_json("baselines_frozen.json")
    runs = load_runs()
    correct = next((r for r in runs if r.get("run") == "correct"), {})

    out.append("## §1 — Setup\n\n")
    if not (proof and stats):
        out.append(gap("mask_proof.json / token_stats.json (NB1)"))
        return
    tier = (frozen or {}).get("tier") or correct.get("tier") or MISSING
    model = (frozen or {}).get("model") or correct.get("model") or MISSING
    out.append(
        "| | |\n|---|---|\n"
        "| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |\n"
        "| Train / val | 225 / 25 (seed 42) |\n"
        f"| `max_length` | p95 đo được = **{stats['p95']}** token "
        f"(p50={stats['p50']}, p99={stats['p99']}, max={stats['max']}) → "
        f"`suggested_max_length={stats['suggested_max_length']}` |\n"
        f"| `MASK_MODE` | `{proof['mask_mode']}` |\n"
        f"| Tier / model | `{tier}` / `{model}` |\n"
        f"| max_steps | {correct.get('max_steps', MISSING)} optimizer step |\n"
        f"| Precision | `{correct.get('precision', MISSING)}` |\n")
    if tmpl:
        keeps = "CÓ" if tmpl.get("ok") else "KHÔNG"
        out.append(f"\n**Template giữ khối `<think>`?** {keeps} — `{tmpl.get('verdict', MISSING)}`\n")

    out.append("\n## §2 — Mask proof\n\n")
    out.append(
        "| | |\n|---|---|\n"
        f"| `supervised_fraction` | **{proof['supervised_fraction']}** "
        f"({proof['n_supervised']}/{proof['n_total']} token) |\n"
        f"| Câu trả lời nằm trong loss | `{str(proof['answer_is_supervised']).lower()}` |\n"
        f"| Câu hỏi KHÔNG nằm trong loss | `{str(proof['question_is_masked']).lower()}` |\n")
    if proof.get("supervised_preview"):
        out.append("\n```\n" + proof["supervised_preview"].rstrip() + "\n```\n")


def section_baselines(out: list[str]) -> None:
    out.append("\n## §3 — Ba baseline\n\n")
    frozen = load_json("baselines_frozen.json")
    if not frozen:
        out.append(gap("baselines_frozen.json (NB2)"))
        return
    verdict = load_json("verdict.json") or {}
    ft = next((r for r in verdict.get("comparison", [])
               if "fine-tune" in str(r.get("run", ""))), None)

    out.append("| Run | target | regression | format | latency (ms) |\n|---|---|---|---|---|\n")
    for name, key in (("(a) base + naive prompt", "baseline_a"),
                      ("(b) base + optimized prompt", "baseline_b")):
        s = frozen.get(key, {})
        out.append(f"| {name} | {num(s.get('target'))} | {num(s.get('regression'))} | "
                   f"{num(s.get('format'))} | {num(s.get('latency_ms'), 0)} |\n")
    if ft:
        out.append(f"| (c) LoRA fine-tune | {num(ft.get('target'))} | "
                   f"{num(ft.get('regression'))} | {num(ft.get('format'))} | "
                   f"{num(ft.get('latency_ms'), 0)} |\n")
    else:
        out.append(f"| (c) LoRA fine-tune | {MISSING} | {MISSING} | {MISSING} | {MISSING} |\n\n")
        out.append(gap("verdict.json (NB5)"))

    a = frozen.get("baseline_a", {}).get("target")
    b = frozen.get("baseline_b", {}).get("target")
    if a is not None and b is not None:
        better = "CÓ" if b > a else "**KHÔNG**"
        out.append(f"\n**(b) có mạnh hơn (a) không?** {better} — (a)={num(a)} → (b)={num(b)}.\n")
    out.append(f"\n`optimized_prompt_sha` = `{frozen.get('optimized_prompt_sha', MISSING)}` · "
               f"`smoke_mode` = `{frozen.get('smoke_mode')}` · "
               f"n_target = {frozen.get('n_target', MISSING)}\n")
    if frozen.get("smoke_mode"):
        out.append("\n> CẢNH BÁO: đây là run SMOKE (`EVAL_LIMIT` đang bật). "
                   "Bỏ `EVAL_LIMIT` và chạy lại NB2 + NB5 trước khi nộp.\n")


def section_autopsy(out: list[str]) -> None:
    out.append("\n## §4 — Giải phẫu cấu hình sai\n\n")
    runs = load_runs()
    if not runs:
        out.append(gap("runs.csv (NB3 + NB4)"))
        return
    autopsy = {r["run"]: r for r in (load_json("autopsy.json") or [])}

    out.append("| Run | vị trí | r | trainable | LR | train loss (NB4) | "
               "**target (NB5 §4)** | train s | VRAM GB |\n"
               "|---|---|---|---|---|---|---|---|---|\n")
    for r in runs:
        key = r.get("run", "")
        tgt = autopsy.get(key, {}).get("target")
        raw = r.get("trainable_params", "")
        trainable = f"{int(raw):,}" if str(raw).isdigit() else (raw or MISSING)
        out.append(
            f"| `{key}` | {r.get('placement', MISSING)} | {r.get('r', MISSING)} | "
            f"{trainable} | {r.get('learning_rate', MISSING)} | "
            f"{r.get('final_loss', MISSING)} | "
            f"{num(tgt) if tgt is not None else MISSING} | "
            f"{r.get('train_seconds', MISSING)} | {r.get('peak_vram_gb', MISSING)} |\n")

    by = {r["run"]: r for r in runs}
    if "correct" in by and "attn_only" in by:
        try:
            c = int(by["correct"]["trainable_params"])
            a = int(by["attn_only"]["trainable_params"])
            drift = abs(c - a) / c
            ok = "ĐẠT" if drift < 0.05 else "KHÔNG ĐẠT"
            out.append(f"\n**Ngân sách công bằng (rubric 2.1):** `correct` {c:,} vs "
                       f"`attn_only` {a:,} → lệch **{drift:.3%}** ({ok}, ngưỡng <5%).\n")
        except (ValueError, KeyError, ZeroDivisionError):
            pass

    steps = sorted({r.get("max_steps") for r in runs if r.get("max_steps")})
    same = "ĐẠT — cả bốn run cùng một số step" if len(steps) == 1 else "KHÔNG ĐỒNG NHẤT"
    out.append(f"\n**Cùng ngân sách step (rubric 2.2):** {steps} → {same}\n")

    if autopsy:
        by_loss = [r["run"] for r in sorted(
            (r for r in runs if r.get("final_loss")), key=lambda r: float(r["final_loss"]))]
        by_target = [r["run"] for r in sorted(
            autopsy.values(), key=lambda r: -float(r["target"]))]
        out.append(f"\n**Xếp theo train loss (tốt→tệ):** {' > '.join(by_loss)}\n\n"
                   f"**Xếp theo target NB5 (tốt→tệ):** {' > '.join(by_target)}\n")
        if by_loss != by_target:
            out.append("\n> Hai thứ tự **KHÁC NHAU**. Đây là Lỗi #3 hiện ra bằng số đo của "
                       "chính bạn — nói thẳng điều này ở mục 4.1.\n")
        else:
            out.append("\n> Hai thứ tự **GIỐNG NHAU**. Chỉ số thay thế tình cờ đồng ý lần này; "
                       "nói rõ đó là may, không phải bằng chứng rằng train loss là thang đo đúng.\n")


def section_verdict(out: list[str]) -> None:
    out.append("\n## §5 — Phán quyết\n\n")
    v = load_json("verdict.json")
    if not v:
        out.append(gap("verdict.json (NB5)"))
        return
    d = v.get("verdict", {})
    state = "PASSED" if d.get("passed") else "FAILED"
    out.append(f"**Kết quả cổng hồi quy: `{state}`**\n\n"
               f"`target Δ = {d.get('target_delta', 0):+.4f}` · "
               f"`regression Δ = {d.get('regression_delta', 0):+.4f}` · "
               f"`valid_trace_rate = {v.get('valid_trace_rate', MISSING)}`\n\n")
    for r in d.get("reasons", []):
        out.append(f"* {r}\n")


def section_qualitative(out: list[str]) -> None:
    out.append("\n## §6 — Định tính (2 ca FT THUA + 2 ca FT THẮNG + 1 ca hoà)\n\n")
    ft_rows = load_json("qualitative.json")
    base = load_json("baseline_preds.json")
    if not ft_rows:
        out.append(gap("qualitative.json (NB5)"))
        return
    if not base:
        out.append(gap("baseline_preds.json (NB2)"))
        out.append("> Không có dự đoán của (b) thì cột so sánh của mục 6 không có nguồn.\n")
        return

    by_i = {r["i"]: r for r in base}
    merged = []
    for r in ft_rows:
        b = by_i.get(r["i"])
        if not b:
            continue
        merged.append({**r, "score_b": b["score_b"], "pred_b": b["pred_b"],
                       "label": b["label"], "full_ticket": b["ticket"],
                       # NB5 cat ft_pred con 90 ky tu cho vua bang in ra man hinh;
                       # ft_pred_full moi parse duoc. Van chay voi artefact cu.
                       "pred_ft": r.get("ft_pred_full") or r["ft_pred"],
                       "delta": round(r["ft_score"] - b["score_b"], 2)})
    if not merged:
        out.append("> Không khớp được chỉ số — NB2 và NB5 chạy với `EVAL_LIMIT` khác nhau?\n")
        return

    merged.sort(key=lambda r: r["delta"])
    losses = [r for r in merged if r["delta"] < 0][:2]
    wins = [r for r in merged if r["delta"] > 0][-2:]
    ties = [r for r in merged if r["delta"] == 0][:1]
    picked = losses + wins + ties

    if len(losses) < 2:
        out.append(f"> Chỉ tìm được {len(losses)} ca fine-tune thua. Rubric 3.4 đòi ≥2. "
                   "Nếu thật sự không có, hãy nói rõ điều đó trong report.\n\n")

    out.append("| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Δ | Nhận xét |\n"
               "|---|---|---|---|---|---|---|\n")
    for r in picked:
        if r["delta"] < 0:
            mark = "❌ **FT thua**"
        elif r["delta"] > 0:
            mark = "✅ FT thắng"
        else:
            mark = "hoà"
        ticket = r["full_ticket"][:64].replace("|", "/")
        label = " · ".join(f"{k}={v}" for k, v in r["label"].items())
        out.append(f"| {r['i']} | {ticket}… | {label.replace('|', '/')} | "
                   f"{r['score_b']:.2f} — sai: {wrong_fields(r['pred_b'], r['label'])} | "
                   f"{r['ft_score']:.2f} — sai: {wrong_fields(r['pred_ft'], r['label'])} | "
                   f"{r['delta']:+.2f} | {mark} |\n")
    out.append(f"\n*Chọn tự động từ {len(merged)} mẫu: 2 ca Δ âm nhất, 2 ca Δ dương nhất, "
               "1 ca hoà — không cherry-pick.*\n")

    # "Các ca thua có mẫu chung nào không?" là câu report phải trả lời — đếm hộ.
    tally: dict[str, int] = {}
    n_lost = 0
    for r in merged:
        if r["delta"] >= 0:
            continue
        n_lost += 1
        for f in wrong_fields(r["pred_ft"], r["label"]).split(", "):
            if f and f != "(không parse được JSON)":
                tally[f] = tally.get(f, 0) + 1
    if n_lost:
        ranked = ", ".join(f"`{k}` {v}/{n_lost}" for k, v in
                           sorted(tally.items(), key=lambda kv: -kv[1]))
        out.append(f"\n**Trường nào hỏng trong TẤT CẢ {n_lost} ca fine-tune thua:** {ranked}\n"
                   "\n> Đây là dữ liệu cho câu \"có mẫu chung nào không?\" — nhìn trường đứng "
                   "đầu rồi mở `data/train_seed.jsonl` xem trường đó có bao nhiêu ví dụ và "
                   "phân bố nhãn ra sao.\n")


def main() -> int:
    out: list[str] = [
        "# Khối dán vào submission/REPORT.md\n\n",
        "> Sinh tự động từ `results/` bằng `scripts/fill_report.py`. Số ở đây khớp "
        "artefact theo cấu tạo — đừng gõ lại bằng tay.\n\n"]
    section_setup(out)
    section_baselines(out)
    section_autopsy(out)
    section_verdict(out)
    section_qualitative(out)

    text = "".join(out)
    RESULTS.mkdir(exist_ok=True)
    dest = RESULTS / "REPORT_BLOCKS.md"
    dest.write_text(text, encoding="utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    print(text)
    print(f"\n-> da ghi {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
