import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ui_parts as ui  # noqa: E402
from fraud.features import RAW_COLS  # noqa: E402
from predictor import LocalPredictor, PredictorError, RemotePredictor  # noqa: E402

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="🛡️", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Memuat model...")
def get_predictor():
    url = os.getenv("FRAUD_API_URL", "").strip()
    return RemotePredictor(url) if url else LocalPredictor()


predictor = get_predictor()
try:
    model_info = predictor.info()
except PredictorError as e:
    st.error(str(e))
    st.stop()
meta = model_info.get("meta", {}) or {}
metrics = meta.get("metrics_test", {}) or {}

# ---------- Preset contoh ----------
NORMAL = dict(Time=0.0, Amount=149.62, V1=-1.359807, V2=-0.072781, V3=2.536347, V4=1.378155, V5=-0.338321,
              V6=0.462388, V7=0.239599, V8=0.098698, V9=0.363787, V10=0.090794, V11=-0.5516, V12=-0.6178,
              V13=-0.99139, V14=-0.311169, V15=1.468177, V16=-0.470401, V17=0.207971, V18=0.025791,
              V19=0.403993, V20=0.251412, V21=-0.018307, V22=0.277838, V23=-0.110474, V24=0.066928,
              V25=0.128539, V26=-0.189115, V27=0.133558, V28=-0.021053)
SUSPICIOUS = {**{f"V{i}": 0.0 for i in range(1, 29)}, "Time": 406.0, "Amount": 1.0,
              **dict(V1=-3.0, V2=3.0, V3=-6.0, V4=4.0, V5=-3.0, V7=-5.0, V9=-2.0, V10=-6.0, V11=3.0,
                     V12=-8.0, V14=-9.0, V16=-4.0, V17=-7.0, V18=-3.0)}
KEY_FEATURES = ["V14", "V12", "V10", "V17", "V4", "V11"]
OTHER_FEATURES = [f"V{i}" for i in range(1, 29) if f"V{i}" not in KEY_FEATURES]


SLIDER_LIM = 30.0


def load_preset(preset: dict):
    st.session_state["real_info"] = None  # bukan transaksi asli dari dataset
    for k, v in preset.items():
        val = float(v)
        if k in KEY_FEATURES:  # jaga agar tidak keluar dari rentang slider
            val = max(-SLIDER_LIM, min(SLIDER_LIM, val))
        st.session_state[f"in_{k}"] = val


def pick_real(kind: str):
    """Ambil satu transaksi asli dari dataset yang diunggah: 'fraud', 'normal', atau 'index'."""
    df = st.session_state.get("real_df")
    if df is None:
        return
    if kind == "index":
        idx = int(st.session_state.get("pick_idx", 0))
        idx = max(0, min(len(df) - 1, idx))
        row = df.iloc[idx]
    else:
        sub = df[df["Class"] == (1 if kind == "fraud" else 0)]
        if sub.empty:
            return
        row = sub.sample(1).iloc[0]
    vals = {c: float(row[c]) for c in RAW_COLS}
    load_preset(vals)
    st.session_state["real_info"] = {
        "idx": int(df.index.get_loc(row.name)),
        "label": int(row["Class"]) if "Class" in df.columns else None,
        "vals": {k: st.session_state[f"in_{k}"] for k in RAW_COLS},
    }


for _k, _v in NORMAL.items():
    st.session_state.setdefault(f"in_{_k}", float(_v))


def save_current(tx: dict, thr: float):
    try:
        res = predictor.predict_one(tx, threshold=thr, explain=False, save=True)
        st.session_state["saved_msg"] = f"Tersimpan di database (ID #{res.get('transaction_id')})."
    except PredictorError as e:
        st.session_state["saved_msg"] = f"Gagal menyimpan: {e}"


# ---------- Header + sidebar ----------
st.markdown(ui.hero([f"Mode: {predictor.mode}", f"Model: {model_info.get('model_kind', '?')} {model_info.get('model_version', '')}", f"Backend: {model_info.get('backend', '-')}"]), unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")
    thr = st.slider("Ambang batas fraud (threshold)", 0.01, 0.99, min(0.99, max(0.01, float(model_info.get("threshold", 0.5)))), 0.01,
                    help="Probabilitas minimal agar transaksi ditandai fraud. "
                         "Turunkan untuk menangkap lebih banyak fraud (recall naik, false alarm naik).")
    st.caption("Nilai awal = threshold terbaik dari notebook (dipilih di data validation).")
    st.divider()
    st.markdown("### 📈 Performa model (data uji)")
    st.markdown(
        ui.kpi("Recall Fraud", f"{metrics.get('recall', 0):.0%}", "pada data uji") + "<div style='height:8px'></div>" +
        ui.kpi("Precision Fraud", f"{metrics.get('precision', 0):.0%}", "pada data uji") + "<div style='height:8px'></div>" +
        ui.kpi("F1-Score Fraud", f"{metrics.get('f1', 0):.2f}", "pada data uji"),
        unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Cek Satu Transaksi", "📂 Analisis Batch (CSV)", "📊 Tentang Model", "🗄️ Riwayat (Database)"])

# =====================================================================
# TAB 1 — satu transaksi
# =====================================================================
with tab1:
    c1, c2 = st.columns([1.15, 1], gap="large")

    with c1:
        st.markdown("#### Data transaksi")

        with st.expander("📂 Ambil transaksi ASLI dari dataset (disarankan untuk demo)", expanded=True):
            st.caption("Unggah `creditcard.csv` atau `sample_demo.csv`, lalu ambil satu transaksi nyata. "
                       "Hasil tebakan model bisa dibandingkan dengan kunci jawaban (Class) di dataset.")
            up1 = st.file_uploader("File dataset (CSV)", type=["csv"], key="real_up")
            if up1 is not None and st.session_state.get("real_name") != (up1.name, up1.size):
                with st.spinner("Membaca dataset..."):
                    d = pd.read_csv(up1)
                miss = [c for c in RAW_COLS if c not in d.columns]
                if miss:
                    st.error(f"Kolom tidak ditemukan: {', '.join(miss)}")
                else:
                    st.session_state["real_df"] = d
                    st.session_state["real_name"] = (up1.name, up1.size)
                    st.session_state["real_info"] = None
            if up1 is None:
                st.session_state["real_df"] = None
                st.session_state["real_name"] = None
            rd = st.session_state.get("real_df")
            if rd is not None:
                has_cls = "Class" in rd.columns
                r1, r2 = st.columns(2)
                r1.button("🚨 Ambil FRAUD asli (acak)", on_click=pick_real, args=("fraud",), disabled=not has_cls)
                r2.button("🟢 Ambil transaksi NORMAL (acak)", on_click=pick_real, args=("normal",), disabled=not has_cls)
                n1, n2 = st.columns([2, 1])
                n1.number_input(f"Atau pilih nomor baris (0 – {len(rd) - 1:,})", min_value=0, max_value=len(rd) - 1,
                                step=1, key="pick_idx")
                n2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                n2.button("Ambil baris ini", on_click=pick_real, args=("index",))
                st.caption(f"{len(rd):,} transaksi dimuat" + (f", {int((rd['Class'] == 1).sum())} di antaranya fraud."
                                                              if has_cls else ". (Tanpa kolom Class)"))

        st.markdown("<div class='note'>Atau isi manual / pakai contoh simulasi:</div>", unsafe_allow_html=True)
        b1, b2, _ = st.columns([1, 1, 1])
        b1.button("✅ Contoh normal", on_click=load_preset, args=(NORMAL,))
        b2.button("⚠️ Contoh fraud", on_click=load_preset, args=(SUSPICIOUS,))

        a1, a2 = st.columns(2)
        a1.number_input("Jumlah transaksi (Amount, €)", min_value=0.0, step=1.0, format="%.2f", key="in_Amount")
        a2.number_input("Waktu (Time, detik sejak transaksi pertama)", min_value=0.0, step=1.0, format="%.0f",
                        key="in_Time")

        st.markdown("**Fitur paling berpengaruh** (hasil PCA, geser untuk simulasi)")
        s1, s2 = st.columns(2)
        for i, f in enumerate(KEY_FEATURES):
            (s1 if i % 2 == 0 else s2).slider(f, -SLIDER_LIM, SLIDER_LIM, step=0.1, key=f"in_{f}")

        with st.expander("Fitur PCA lainnya (V1–V28)"):
            e1, e2, e3, e4 = st.columns(4)
            for i, f in enumerate(OTHER_FEATURES):
                [e1, e2, e3, e4][i % 4].number_input(f, step=0.1, format="%.4f", key=f"in_{f}")

    tx = {c: float(st.session_state[f"in_{c}"]) for c in RAW_COLS}
    try:
        result = predictor.predict_one(tx, threshold=thr, explain=True)
    except PredictorError as e:
        st.error(str(e))
        st.stop()
    p = float(result["fraud_probability"])

    with c2:
        info = st.session_state.get("real_info")
        if info and all(np.isclose(info["vals"][c], st.session_state[f"in_{c}"]) for c in RAW_COLS):
            pred_fraud = p >= thr
            if info["label"] is None:
                st.markdown(f'<div class="card"><h4>Transaksi asli · baris #{info["idx"]:,}</h4>'
                            '<div class="note">Dataset tidak punya kolom Class, jadi tidak ada kunci jawaban.</div></div>',
                            unsafe_allow_html=True)
            else:
                truth = "FRAUD" if info["label"] == 1 else "NORMAL"
                ok = pred_fraud == (info["label"] == 1)
                cls = "v-ok" if ok else "v-bad"
                msg = "✅ Tebakan model BENAR" if ok else "❌ Tebakan model MELESET"
                st.markdown(f'<div class="card"><h4>Transaksi asli · baris #{info["idx"]:,}</h4>'
                            f'<div style="font-size:1.05rem;margin-bottom:8px">Kunci jawaban di dataset: <b>{truth}</b></div>'
                            f'<div class="verdict {cls}" style="margin-top:0">{msg}</div></div>',
                            unsafe_allow_html=True)
            st.markdown("")
        st.markdown('<div class="card"><h4>Hasil prediksi</h4>' + ui.gauge(p, thr) + '<div class="note" style="text-align:center;margin-bottom:8px">Garis putih = threshold ' + f'{thr:.2f}' + '</div>' + ui.verdict(p, thr) + '</div>',
                    unsafe_allow_html=True)
        st.markdown("")
        factors = result.get("top_factors")
        if factors:
            items = [(f["feature"], float(f["value"]), float(f["contribution"])) for f in factors]
            st.markdown('<div class="card"><h4>Faktor penentu (SHAP)</h4>' + ui.contrib_bars(items) +
                        '<div class="note">Merah mendorong ke arah FRAUD, hijau ke arah NORMAL (satuan log-odds). '
                        'Nilai fitur Amount/Time ditampilkan dalam bentuk terstandarisasi.</div></div>',
                        unsafe_allow_html=True)
        else:
            st.info("Penjelasan SHAP aktif bila library `xgboost` terpasang di sisi model "
                    "(`pip install -U xgboost`). Prediksi tetap berjalan tanpa itu.")
        st.button("💾 Simpan prediksi ini ke database", on_click=save_current, args=(tx, thr))
        if st.session_state.get("saved_msg"):
            st.caption(st.session_state.pop("saved_msg"))

# =====================================================================
# TAB 2 — batch CSV
# =====================================================================
with tab2:
    st.markdown("Unggah CSV dengan kolom **Time, Amount, V1…V28** (format dataset *creditcard.csv*). "
                "Kolom **Class** bersifat opsional; bila ada, performa model dihitung otomatis.")
    up = st.file_uploader("Pilih file CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        missing = [c for c in RAW_COLS if c not in df.columns]
        if missing:
            st.error(f"Kolom berikut tidak ditemukan: {', '.join(missing)}")
        else:
            bar = st.progress(0.0, text=f"Menganalisis {len(df):,} transaksi...")
            try:
                probs = predictor.predict_batch_proba(df, progress=lambda f: bar.progress(f))
            except PredictorError as e:
                bar.empty()
                st.error(str(e))
                st.stop()
            bar.empty()
            res = df.copy()
            res["fraud_probability"] = probs
            res["prediction"] = np.where(probs >= thr, "FRAUD", "NORMAL")
            flagged = res[res["prediction"] == "FRAUD"].sort_values("fraud_probability", ascending=False)

            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(ui.kpi("Total transaksi", f"{len(res):,}"), unsafe_allow_html=True)
            k2.markdown(ui.kpi("Terindikasi fraud", f"{len(flagged):,}", f"{len(flagged)/len(res)*100:.3f}% dari total"),
                        unsafe_allow_html=True)
            k3.markdown(ui.kpi("Probabilitas tertinggi", ui.fmt_pct(float(probs.max()))), unsafe_allow_html=True)
            k4.markdown(ui.kpi("Rata-rata probabilitas", ui.fmt_pct(float(probs.mean()))), unsafe_allow_html=True)

            if "Class" in res.columns:
                y = res["Class"].astype(int).values
                yp = (probs >= thr).astype(int)
                tp = int(((y == 1) & (yp == 1)).sum())
                fp = int(((y == 0) & (yp == 1)).sum())
                fn = int(((y == 1) & (yp == 0)).sum())
                tn = int(((y == 0) & (yp == 0)).sum())
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
                st.markdown("#### Evaluasi terhadap label asli")
                m1, m2, m3, m4 = st.columns(4)
                m1.markdown(ui.kpi("Precision", f"{prec:.2%}"), unsafe_allow_html=True)
                m2.markdown(ui.kpi("Recall", f"{rec:.2%}"), unsafe_allow_html=True)
                m3.markdown(ui.kpi("F1-Score", f"{f1:.3f}"), unsafe_allow_html=True)
                m4.markdown(ui.kpi("TP / FP / FN", f"{tp} / {fp} / {fn}", f"TN {tn:,}"), unsafe_allow_html=True)
                st.caption("Catatan: jika file berisi data yang sama dengan data latih, angka ini terlalu optimistis.")

            st.markdown("#### Transaksi dengan risiko tertinggi")
            show = res.sort_values("fraud_probability", ascending=False).head(200)
            cols = ["fraud_probability", "prediction", "Time", "Amount"] + \
                   (["Class"] if "Class" in res.columns else [])
            st.dataframe(show[cols], hide_index=True)

            st.download_button("⬇️ Unduh transaksi terindikasi fraud (CSV)",
                               flagged.to_csv(index=False).encode("utf-8"),
                               file_name="fraud_terindikasi.csv", mime="text/csv", disabled=flagged.empty)

# =====================================================================
# TAB 3 — tentang
# =====================================================================
with tab3:
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### Bagaimana model bekerja")
        st.markdown(
            "- **Data**: 284.807 transaksi kartu kredit Eropa, hanya **492 (0,17%)** yang fraud.\n"
            "- **Fitur**: `V1–V28` hasil PCA (dianonimkan), ditambah `Amount` dan `Time` yang distandardisasi.\n"
            "- **Imbalance**: ditangani dengan class weight (`scale_pos_weight` pada XGBoost), tanpa membuang data normal.\n"
            "- **Metrik**: fokus pada *precision–recall* (`aucpr`), karena akurasi menyesatkan pada data timpang.\n"
            "- **Penjelasan**: SHAP menunjukkan fitur mana yang mendorong sebuah transaksi dianggap fraud.")
    with b:
        st.markdown("#### Confusion matrix (data uji 20%)")
        cm = pd.DataFrame([[metrics.get("tn", 0), metrics.get("fp", 0)], [metrics.get("fn", 0), metrics.get("tp", 0)]], index=["Aktual Normal", "Aktual Fraud"],
                          columns=["Prediksi Normal", "Prediksi Fraud"])
        st.dataframe(cm)
        st.markdown("<div class='note'>Ubah threshold di sidebar untuk melihat trade-off recall vs precision.</div>",
                    unsafe_allow_html=True)
    st.warning("Aplikasi ini untuk demonstrasi/portofolio. Fitur V1–V28 anonim, jadi input manual hanya simulasi; "
               "untuk penggunaan nyata gunakan data transaksi yang diproses dengan PCA yang sama.")

# =====================================================================
# TAB 4 — riwayat database
# =====================================================================
with tab4:
    st.markdown("Prediksi yang disimpan (lewat tombol **Simpan** atau endpoint `POST /predict`) tercatat di database SQLite.")
    try:
        stt = predictor.stats()
        rows = predictor.history(100)
    except PredictorError as e:
        st.error(str(e))
        stt, rows = None, []
    if stt:
        h1, h2, h3, h4 = st.columns(4)
        h1.markdown(ui.kpi("Total tersimpan", f"{stt['total']:,}"), unsafe_allow_html=True)
        h2.markdown(ui.kpi("Fraud", f"{stt['fraud']:,}"), unsafe_allow_html=True)
        h3.markdown(ui.kpi("Normal", f"{stt['normal']:,}"), unsafe_allow_html=True)
        h4.markdown(ui.kpi("Rata-rata probabilitas", ui.fmt_pct(float(stt["avg_probability"] or 0))),
                    unsafe_allow_html=True)
    if rows:
        hist = pd.DataFrame(rows).drop(columns=["features_json"], errors="ignore")
        st.dataframe(hist, hide_index=True)
    else:
        st.info("Belum ada prediksi tersimpan. Buka tab pertama, lalu klik 'Simpan prediksi ini ke database'.")
