import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
tr = json.loads((R / "translate_report.json").read_text())
sim = json.loads((R / "similarity_report.json").read_text())

INK, MUTED, GRID = "#15212b", "#5a6a77", "#e1e7ec"
GEMMA, QWEN, MITRA = "#0e5a63", "#b8742a", "#7a5aa6"
W, H = 760, 626
PW, PX = 300, (60, 410)

panels = [
    ("chrF++", tr, "chrf++", (26.0, 31.5), [27, 28, 29, 30, 31], "{:.2f}", "{:+.2f}"),
    ("Semantic similarity (BGE cosine)", sim, "cosine", (0.705, 0.755), [0.71, 0.72, 0.73, 0.74, 0.75], "{:.3f}", "{:+.3f}"),
]
rows = [("gemma-4-31b", "Gemma-4-31B", GEMMA), ("qwen3.8-27b", "Qwen3.8-27B", QWEN), ("mitra-qwen-9b", "MITRA-Qwen 9B", MITRA)]
pairs = [("gemma-4-31b - qwen3.8-27b", "Gemma - Qwen3.8"), ("gemma-4-31b - mitra-qwen-9b", "Gemma - MITRA"), ("mitra-qwen-9b - qwen3.8-27b", "MITRA - Qwen3.8")]

o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif">',
     f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
     f'<text x="24" y="28" font-size="16" font-weight="700" fill="{INK}">Gemma-4-31B leads on Sanskrit to English; MITRA-Qwen 9B trails both larger models</text>',
     f'<text x="24" y="47" font-size="11.5" fill="{MUTED}">{tr["n"]:,} shared verses, 95% bootstrap interval. Higher is better. Every paired difference excludes zero.</text>']

for (title, rep, key, (lo, hi), ticks, fmt, dfmt), x0 in zip(panels, PX):
    def sx(v): return x0 + (v - lo) / (hi - lo) * PW
    o.append(f'<text x="{x0}" y="86" font-size="13" font-weight="600" fill="{INK}">{title}</text>')
    for t in ticks:
        o.append(f'<line x1="{sx(t):.1f}" y1="100" x2="{sx(t):.1f}" y2="214" stroke="{GRID}"/>')
        o.append(f'<text x="{sx(t):.1f}" y="230" font-size="10.5" fill="{MUTED}" text-anchor="middle">{t}</text>')
    for i, (mk, name, col) in enumerate(rows):
        y = 124 + i * 40
        m = rep["models"][mk]
        v, (a, b) = m[key], m["ci95"]
        o.append(f'<line x1="{sx(a):.1f}" y1="{y}" x2="{sx(b):.1f}" y2="{y}" stroke="{col}" stroke-width="3" stroke-linecap="round"/>')
        o.append(f'<circle cx="{sx(v):.1f}" cy="{y}" r="7" fill="{col}" stroke="#fff" stroke-width="2"/>')
        o.append(f'<text x="{sx(v):.1f}" y="{y-14}" font-size="12" font-weight="600" fill="{INK}" text-anchor="middle">{fmt.format(v)}</text>')
    o.append(f'<rect x="{x0}" y="244" width="{PW}" height="84" rx="8" fill="#eef5f5"/>')
    o.append(f'<text x="{x0+14}" y="262" font-size="11" fill="{MUTED}">Paired differences, 95% CI</text>')
    for j, (pk, pl) in enumerate(pairs):
        d = rep["pairwise"][pk]
        da, db = d["ci95"]
        col = GEMMA if d["diff"] > 0 else MITRA
        o.append(f'<text x="{x0+14}" y="{283+j*18}" font-size="12" fill="{INK}">{pl}</text>')
        o.append(f'<text x="{x0+PW-14}" y="{283+j*18}" font-size="12" font-weight="700" fill="{col}" text-anchor="end">{dfmt.format(d["diff"])}'
                 f'<tspan font-size="10.5" font-weight="400" fill="{MUTED}">  [{dfmt.format(da)}, {dfmt.format(db)}]</tspan></text>')

ext = [
    ("Dense ~30B", "Gemma-4-31B", 1415, "Qwen3.5-27B", 1351),
    ("MoE ~30B", "Gemma-4-26B-A4B", 1253, "Qwen3.5-35B-A3B", 1370),
    ("~10B", "Gemma-4-12B", 1132, "Qwen3.5-9B", 1193),
    ("~4B", "Gemma-4-E4B", 1029, "Qwen3.5-4B", 1016),
]
o.append(f'<line x1="24" y1="348" x2="{W-24}" y2="348" stroke="{GRID}"/>')
o.append(f'<text x="24" y="376" font-size="13" font-weight="600" fill="{INK}">Independent check: Dharmamitra arena, Sanskrit to English Elo</text>')
o.append(f'<text x="24" y="394" font-size="11" fill="{MUTED}">Third-party head-to-head arena, 100 sentences, Gemini judge. Qwen3.5 (not 3.8) is the Qwen on that board. Pairs matched by size.</text>')
lo, hi, ax0, axw = 1000, 1450, 230, 440
def ex(v): return ax0 + (v - lo) / (hi - lo) * axw
for t in (1000, 1100, 1200, 1300, 1400):
    o.append(f'<line x1="{ex(t):.1f}" y1="408" x2="{ex(t):.1f}" y2="530" stroke="{GRID}"/>')
    o.append(f'<text x="{ex(t):.1f}" y="546" font-size="10.5" fill="{MUTED}" text-anchor="middle">{t}</text>')
for i, (grp, gn, gv, qn, qv) in enumerate(ext):
    y = 432 + i * 30
    o.append(f'<text x="24" y="{y+4}" font-size="12" fill="{INK}">{grp}</text>')
    o.append(f'<line x1="{ex(gv):.1f}" y1="{y}" x2="{ex(qv):.1f}" y2="{y}" stroke="{GRID}" stroke-width="3"/>')
    o.append(f'<circle cx="{ex(qv):.1f}" cy="{y}" r="6" fill="{QWEN}" stroke="#fff" stroke-width="2"/>')
    o.append(f'<circle cx="{ex(gv):.1f}" cy="{y}" r="6" fill="{GEMMA}" stroke="#fff" stroke-width="2"/>')
    win = gv > qv
    o.append(f'<text x="{ex(max(gv,qv))+14:.1f}" y="{y+4}" font-size="11" fill="{GEMMA if win else QWEN}" font-weight="600">{"Gemma" if win else "Qwen"} {abs(gv-qv):+d}'.replace("+","+") + '</text>')
o.append(f'<text x="24" y="578" font-size="11" fill="{MUTED}">Gemma wins at ~30B dense (+64 Elo, about 59% head-to-head). Qwen3.5 leads at the MoE and ~10B sizes; ~4B is a tie.</text>')
for i, (_, name, col) in enumerate(rows):
    x = 24 + i * 140
    o.append(f'<circle cx="{x+6}" cy="601" r="5" fill="{col}"/><text x="{x+17}" y="605" font-size="11.5" fill="{INK}">{name}</text>')
o.append(f'<text x="{W-24}" y="605" font-size="10.5" fill="{MUTED}" text-anchor="end">Reference is one translator, so read as relative, not absolute.</text>')
o.append("</svg>")
(R / "model_comparison.svg").write_text("\n".join(o))
