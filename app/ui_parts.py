"""Potongan HTML/CSS untuk tampilan (tanpa dependensi Streamlit)."""
import html as _h

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; max-width: 1200px; }
.hero { background: radial-gradient(1200px 300px at 0% 0%, rgba(45,212,191,.18), transparent 60%),
        linear-gradient(135deg,#111a2e 0%,#0f1b33 100%);
        border: 1px solid #1f2d4a; border-radius: 20px; padding: 28px 32px; margin-bottom: 18px; }
.hero h1 { margin: 0 0 6px 0; font-size: 2rem; font-weight: 800; letter-spacing: -.02em; }
.hero p { margin: 0; color: #9fb0cc; font-size: 1rem; }
.pill { display:inline-block; padding: 3px 12px; border-radius: 999px; font-size: .75rem; font-weight: 600;
        background: rgba(45,212,191,.12); color:#2dd4bf; border:1px solid rgba(45,212,191,.35); margin-right:6px; }
.card { background:#111a2e; border:1px solid #1f2d4a; border-radius:16px; padding:20px 22px; }
.card h4 { margin:0 0 10px 0; font-size:.8rem; letter-spacing:.08em; text-transform:uppercase; color:#8da2c4; font-weight:600; }
.verdict { border-radius:14px; padding:14px 18px; font-weight:700; font-size:1.1rem; margin-top:6px; text-align:center; }
.v-ok { background:rgba(34,197,94,.12); border:1px solid rgba(34,197,94,.4); color:#4ade80; }
.v-warn { background:rgba(245,158,11,.12); border:1px solid rgba(245,158,11,.4); color:#fbbf24; }
.v-bad { background:rgba(239,68,68,.14); border:1px solid rgba(239,68,68,.5); color:#f87171; }
.kpi { background:#111a2e; border:1px solid #1f2d4a; border-radius:14px; padding:14px 16px; }
.kpi .l { color:#8da2c4; font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; font-weight:600; }
.kpi .v { font-size:1.7rem; font-weight:800; margin-top:2px; }
.kpi .s { color:#6f84a8; font-size:.78rem; }
.crow { display:grid; grid-template-columns: 135px 1fr 60px; align-items:center; gap:10px; margin:7px 0; font-size:.85rem; }
.crow .n { color:#cbd8ee; font-weight:600; }
.crow .t { position:relative; height:12px; background:#0b1220; border-radius:6px; }
.crow .b { position:absolute; top:0; height:12px; border-radius:6px; }
.crow .c { text-align:right; color:#8da2c4; font-variant-numeric: tabular-nums; }
.note { color:#8da2c4; font-size:.82rem; }
div[data-testid="stTabs"] button { font-weight:600; }
</style>
"""


def hero(pills) -> str:
    chips = "".join(f'<span class="pill">{_h.escape(str(p))}</span>' for p in pills)
    return (
        '<div class="hero"><h1>🛡️ Credit Card Fraud Detector</h1>'
        '<p>Deteksi transaksi kartu kredit mencurigakan dari data yang sangat tidak seimbang, '
        'lengkap dengan penjelasan alasan keputusan model.</p>'
        f'<div style="margin-top:14px">{chips}</div></div>'
    )


def fmt_pct(p: float) -> str:
    return "<0.01%" if p < 0.0001 else f"{p*100:.2f}%"


def gauge(p: float, thr: float) -> str:
    color = "#22c55e" if p < thr * 0.5 else ("#f59e0b" if p < thr else "#ef4444")
    arc = 282.74
    dash = max(p, 0.004) * arc
    import math
    ang = math.pi * (1 - thr)
    mx, my = 110 + 90 * math.cos(ang), 110 - 90 * math.sin(ang)
    ix, iy = 110 + 74 * math.cos(ang), 110 - 74 * math.sin(ang)
    return (
        '<svg viewBox="0 0 220 135" width="100%" style="max-width:340px;display:block;margin:0 auto">'
        '<path d="M20 110 A90 90 0 0 1 200 110" fill="none" stroke="#0b1220" stroke-width="16" stroke-linecap="round"/>'
        f'<path d="M20 110 A90 90 0 0 1 200 110" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round" '
        f'stroke-dasharray="{dash:.1f} {arc}"/>'
        f'<line x1="{ix:.1f}" y1="{iy:.1f}" x2="{mx:.1f}" y2="{my:.1f}" stroke="#e6edf7" stroke-width="2.5"/>'
        f'<text x="110" y="100" text-anchor="middle" font-size="30" font-weight="800" fill="{color}">{fmt_pct(p)}</text>'
        '<text x="110" y="122" text-anchor="middle" font-size="10" fill="#8da2c4">PROBABILITAS FRAUD</text>'
        '</svg>'
    )


def verdict(p: float, thr: float) -> str:
    if p >= thr:
        return '<div class="verdict v-bad">⚠️ Terindikasi FRAUD, sebaiknya diblokir / diverifikasi</div>'
    if p >= thr * 0.5:
        return '<div class="verdict v-warn">🔎 Perlu perhatian (mendekati batas)</div>'
    return '<div class="verdict v-ok">✅ Transaksi tampak NORMAL</div>'


def kpi(label: str, value: str, sub: str = "") -> str:
    return f'<div class="kpi"><div class="l">{_h.escape(label)}</div><div class="v">{value}</div><div class="s">{_h.escape(sub)}</div></div>'


def contrib_bars(items) -> str:
    """items: [(nama_fitur, nilai_input, kontribusi_logodds)] sudah terurut."""
    if not items:
        return ""
    m = max(abs(c) for _, _, c in items) or 1.0
    rows = []
    for name, val, c in items:
        w = abs(c) / m * 50  # persen dari lebar (maks 50% tiap sisi)
        if c >= 0:
            bar = f'<div class="b" style="left:50%;width:{w:.1f}%;background:#ef4444"></div>'
        else:
            bar = f'<div class="b" style="left:{50-w:.1f}%;width:{w:.1f}%;background:#22c55e"></div>'
        rows.append(
            f'<div class="crow"><div class="n">{_h.escape(name)}<span style="color:#6f84a8;font-weight:400"> = {val:.2f}</span></div>'
            f'<div class="t">{bar}<div style="position:absolute;left:50%;top:-2px;width:1px;height:16px;background:#334866"></div></div>'
            f'<div class="c">{c:+.2f}</div></div>'
        )
    return "".join(rows)
