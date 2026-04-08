"""
FinScan — AI-Powered Personal Financial Behavior Optimization System
====================================================================
Stack : Streamlit · Pandas · Scikit-learn (TF-IDF, Isolation Forest) · Plotly
Deploy: Streamlit Community Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FinScan · Financial Anomaly Detector",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# GLOBAL STYLES
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #f0f4fa; }

/* Metric cards */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid rgba(37,99,235,0.12);
    border-radius: 14px;
    padding: 16px 20px;
    box-shadow: 0 1px 4px rgba(37,99,235,0.07);
}
[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 26px !important;
    font-weight: 600 !important;
}

/* Cards */
.fin-card {
    background: white;
    border-radius: 14px;
    border: 1px solid rgba(37,99,235,0.10);
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(37,99,235,0.06);
}

/* Advice box */
.advice-box {
    background: rgba(37,99,235,0.05);
    border-left: 4px solid #2563eb;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 14px;
    color: #1e3a8a;
}
.advice-warn {
    background: rgba(245,158,11,0.07);
    border-left: 4px solid #f59e0b;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 14px;
    color: #92400e;
}
.advice-danger {
    background: rgba(239,68,68,0.07);
    border-left: 4px solid #ef4444;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 14px;
    color: #991b1b;
}

/* Section header */
.section-header {
    font-size: 16px;
    font-weight: 600;
    color: #0f172a;
    margin: 24px 0 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(37,99,235,0.15);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid rgba(37,99,235,0.10);
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: white;
    border-radius: 14px;
    border: 1.5px dashed rgba(37,99,235,0.3);
    padding: 8px;
}

/* Hide default footer */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════

FAMILY_WHITELIST = r"jumpatip|jumpathip"  # ← โอนในครอบครัว ไม่ flag

CATEGORY_MAP = {
    "อาหาร": "🍜 อาหาร",
    "เดินทาง รถ": "🚗 เดินทาง",
    "จ่ายบิล": "📄 จ่ายบิล",
    "เลิฟเว่อ": "💑 ส่วนตัว",
    "trade": "📈 การลงทุน",
    "การเรียนรู้": "📚 การเรียนรู้",
    "ให้ยืม": "🤝 ให้ยืม",
    "ค่ายา หาหมอ": "🏥 สุขภาพ",
    "ค่าห้อง": "🏠 ที่พัก",
    "ของใช้": "🛍 ของใช้",
    "ทำเล็บ": "💅 ความงาม",
    "อื่นๆ": "📦 อื่นๆ",
}

ADVICE_DB = {
    "🍜 อาหาร": {
        "tip": "รายจ่ายอาหารนอกบ้านสูงกว่าปกติ",
        "action": "ลองทำอาหารกินเองสัปดาห์ละ 2–3 วัน ประหยัดได้ 20–30%",
        "level": "warn",
    },
    "🚗 เดินทาง": {
        "tip": "ค่าเดินทางสูงกว่าปกติ",
        "action": "ใช้ BTS/MRT แทน Grab ในชั่วโมงเร่งด่วน ประหยัดได้ถึง 60%",
        "level": "warn",
    },
    "📄 จ่ายบิล": {
        "tip": "ค่าบิลผิดปกติ — อาจถูกเรียกเก็บซ้ำ",
        "action": "ตรวจสอบว่าบิลถูกเรียกเก็บซ้ำหรือมีค่าใช้จ่ายแอบแฝง",
        "level": "danger",
    },
    "📈 การลงทุน": {
        "tip": "รายจ่ายด้านการลงทุนสูงกว่าปกติ",
        "action": "ตรวจสอบว่าเป็นไปตามแผนการลงทุนที่วางไว้",
        "level": "info",
    },
    "📚 การเรียนรู้": {
        "tip": "ค่าเรียน/คอร์สสูงกว่าปกติ",
        "action": "เช็ก refund policy ก่อนลงทะเบียนคอร์สราคาแพง",
        "level": "info",
    },
    "🤝 ให้ยืม": {
        "tip": "โอนเงินให้คนอื่นบ่อยและก้อนใหญ่",
        "action": "บันทึก IOU ไว้เสมอ และตั้งวันคืนให้ชัดเจน",
        "level": "warn",
    },
    "🏥 สุขภาพ": {
        "tip": "ค่ารักษาพยาบาลสูงกว่าปกติ",
        "action": "ตรวจสอบสิทธิ์ประกันสุขภาพ บัตรทอง หรือประกันที่มีอยู่",
        "level": "info",
    },
    "🏠 ที่พัก": {
        "tip": "ค่าเช่าห้องผิดปกติ",
        "action": "ตรวจสอบว่าถูกเรียกเก็บซ้ำหรือมีค่าใช้จ่ายแฝง",
        "level": "danger",
    },
    "🛍 ของใช้": {
        "tip": "รายจ่ายของใช้สูงกว่าปกติ",
        "action": "ซื้อของใช้เป็น bulk ราคาถูกกว่าซื้อทีละชิ้น",
        "level": "info",
    },
}

CAT_COLORS = [
    "#2563eb", "#0ea5e9", "#6366f1", "#10b981",
    "#f59e0b", "#8b5cf6", "#06b6d4", "#ef4444",
    "#f97316", "#84cc16",
]


# ══════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, max_entries=3)
def load_and_clean(file_bytes: bytes) -> pd.DataFrame:
    """
    Step 1: Ingest CSV and apply noise filters.
    - Remove 'Move Money' (internal pocket transfers).
    - Remove Deposits (income, not expense).
    - Remove micro-transactions < ฿5.
    - Parse D/M/YYYY dates correctly (KBank format).
    """
    df = pd.read_csv(
        pd.io.common.BytesIO(file_bytes),
        encoding="utf-8-sig",
    )

    # Standardise column names
    df.columns = [c.strip() for c in df.columns]

    # Parse amount
    df["amount"] = pd.to_numeric(
        df["Txn"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    # Keep only real outflows
    df = df[
        df["Type"].isin(["Payment", "Transfer Withdraw"]) &
        (df["amount"] < 0)
    ].copy()

    df["amount"] = df["amount"].abs()

    # Remove micro-transactions
    df = df[df["amount"] >= 5].copy()

    # Parse date — KBank uses D/M/YYYY
    def parse_kbank_date(s):
        try:
            parts = str(s).strip().split("/")
            if len(parts) == 3:
                return pd.Timestamp(int(parts[2]), int(parts[1]), int(parts[0]))
        except Exception:
            pass
        return pd.NaT

    df["date"] = df["Date"].apply(parse_kbank_date)
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Whitelist: family transfers → not anomaly candidates
    df["is_family"] = (
        df["Note"].fillna("").str.upper().str.contains(FAMILY_WHITELIST.upper()) |
        df["Cloud Pocket Name"].fillna("").str.contains("เลิฟเว่อ", na=False)
    )

    return df


@st.cache_data(show_spinner=False, max_entries=3)
def categorize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: NLP Categorization using TF-IDF features.
    - Use bank's own Category column when available.
    - Fallback: TF-IDF keyword scoring on combined text.
    """
    df = df.copy()

    # Combine text fields for NLP
    df["text_combined"] = (
        df["Note"].fillna("") + " " +
        df["Memo"].fillna("") + " " +
        df["Cloud Pocket Name"].fillna("")
    ).str.lower().str.strip()

    # Use bank category first
    def map_category(row):
        bank_cat = str(row.get("Category", "")).strip()
        txn_type = str(row.get("Type", ""))

        if txn_type == "Transfer Withdraw":
            return "🤝 โอนเงิน"

        if bank_cat and bank_cat not in ["", "nan", "อื่นๆ"]:
            return CATEGORY_MAP.get(bank_cat, f"📦 {bank_cat}")

        # Keyword fallback (acts as lightweight NLP rule-set)
        txt = str(row["text_combined"]).lower()
        if any(k in txt for k in ["grab", "mrt", "bts", "รถ", "taxi", "bolt"]):
            return "🚗 เดินทาง"
        if any(k in txt for k in ["กาแฟ", "cafe", "coffee", "starbucks"]):
            return "🍜 อาหาร"
        if any(k in txt for k in ["ร้าน", "food", "ข้าว", "ก๋วยเตี๋ยว", "sushi"]):
            return "🍜 อาหาร"
        if any(k in txt for k in ["netflix", "spotify", "youtube", "steam"]):
            return "📱 Subscription"
        if any(k in txt for k in ["shop", "ช้อปปิ้ง", "nike", "uniqlo", "lazada", "shopee"]):
            return "🛍 ของใช้"
        return "📦 อื่นๆ"

    df["category"] = df.apply(map_category, axis=1)

    # ── TF-IDF feature extraction (for anomaly model) ──────────
    vectorizer = TfidfVectorizer(
        max_features=30,
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=1,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(df["text_combined"].fillna(""))
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
        )
        df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
    except Exception:
        pass  # not enough text data

    return df


@st.cache_data(show_spinner=False, max_entries=3)
def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3: Isolation Forest anomaly detection.
    - Features: amount, rolling stats, day-of-week, month, TF-IDF.
    - IQR hard filter to reduce false positives.
    - Family whitelist: never flag.
    """
    df = df.copy()

    amounts = df["amount"].values

    # Rolling 7-day stats
    rolling_mean = pd.Series(amounts).rolling(7, min_periods=1).mean().values
    rolling_std  = pd.Series(amounts).rolling(7, min_periods=1).std().fillna(1).values
    z_scores     = (amounts - rolling_mean) / rolling_std

    df["rolling_mean"] = rolling_mean
    df["rolling_std"]  = rolling_std
    df["z_score"]      = z_scores

    # IQR cutoff
    q1, q3 = np.percentile(amounts, 25), np.percentile(amounts, 75)
    iqr = q3 - q1
    iqr_cutoff = q3 + 2.5 * iqr
    df["iqr_cutoff"] = iqr_cutoff

    # Feature matrix for Isolation Forest
    base_features = np.column_stack([
        amounts,
        rolling_mean,
        z_scores,
        df["date"].dt.dayofweek.values,
        df["date"].dt.month.values,
    ])

    # Append TF-IDF features if available
    tfidf_cols = [c for c in df.columns if c.startswith("tfidf_")]
    if tfidf_cols:
        tfidf_arr = df[tfidf_cols].values
        feature_matrix = np.hstack([base_features, tfidf_arr])
    else:
        feature_matrix = base_features

    # Normalise
    feat_min = feature_matrix.min(axis=0)
    feat_max = feature_matrix.max(axis=0)
    feat_range = np.where(feat_max == feat_min, 1, feat_max - feat_min)
    feat_norm = (feature_matrix - feat_min) / feat_range

    # Isolation Forest
    n_samples = len(df)
    sample_size = min(64, n_samples)
    clf = IsolationForest(
        n_estimators=50,
        contamination=0.05,
        random_state=42,
    )
    clf.fit(feat_norm)
    if_scores  = clf.decision_function(feat_norm)   # higher = more normal
    if_labels  = clf.predict(feat_norm)              # -1 = anomaly

    df["if_score"]   = if_scores
    df["if_anomaly"] = if_labels == -1

    # Combined rule: IF anomaly OR IQR outlier (and not family)
    df["iqr_anomaly"] = df["amount"] > iqr_cutoff

    df["is_anomaly"] = (
        ~df["is_family"] &
        (df["if_anomaly"] | df["iqr_anomaly"])
    )

    # Human-readable reason
    def reason(row):
        if row["is_family"]:
            return "ครอบครัว/สมยอม ✓"
        if row["if_anomaly"] and row["iqr_anomaly"]:
            return "ยอดสูง + Isolation Forest"
        if row["iqr_anomaly"]:
            return f"ยอดสูงกว่า IQR cutoff (฿{iqr_cutoff:,.0f})"
        if row["if_anomaly"]:
            return "รูปแบบผิดปกติ (Isolation Forest)"
        return ""

    df["anomaly_reason"] = df.apply(reason, axis=1)

    return df, iqr_cutoff


# ══════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════

def chart_timeline(df: pd.DataFrame) -> go.Figure:
    """Stacked bar: normal vs anomaly spending over time."""
    normal  = df[~df["is_anomaly"]]
    anomaly = df[df["is_anomaly"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=normal["date"], y=normal["amount"],
        name="ปกติ", marker_color="rgba(37,99,235,0.55)",
        hovertemplate="%{x|%d %b}<br>฿%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=anomaly["date"], y=anomaly["amount"],
        name="Anomaly", marker_color="rgba(239,68,68,0.85)",
        hovertemplate="%{x|%d %b}<br>฿%{y:,.0f}<br><b>ANOMALY</b><extra></extra>",
    ))
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
        xaxis=dict(showgrid=False, tickformat="%d %b"),
        yaxis=dict(gridcolor="rgba(37,99,235,0.06)", tickprefix="฿"),
        font=dict(family="sans-serif", size=12, color="#64748b"),
    )
    return fig


def chart_category_donut(df: pd.DataFrame) -> go.Figure:
    """Donut chart: spending by category."""
    cat_totals = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    fig = go.Figure(go.Pie(
        labels=cat_totals.index,
        values=cat_totals.values,
        hole=0.65,
        marker=dict(colors=CAT_COLORS),
        textinfo="label+percent",
        hovertemplate="%{label}<br>฿%{value:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=8, b=0),
        showlegend=False,
        font=dict(family="sans-serif", size=12, color="#64748b"),
    )
    return fig


def chart_monthly_bar(df: pd.DataFrame) -> go.Figure:
    """Monthly spending: normal vs anomaly stacked."""
    df2 = df.copy()
    df2["month"] = df2["date"].dt.to_period("M").astype(str)

    monthly = df2.groupby(["month", "is_anomaly"])["amount"].sum().reset_index()
    normal  = monthly[~monthly["is_anomaly"]].rename(columns={"amount": "normal"})
    anom    = monthly[monthly["is_anomaly"]].rename(columns={"amount": "anomaly"})
    merged  = normal.merge(anom[["month","anomaly"]], on="month", how="left").fillna(0)
    merged["total"] = merged["normal"] + merged["anomaly"]

    # % change MoM
    merged["pct_change"] = merged["total"].pct_change() * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=merged["month"], y=merged["normal"],
        name="ปกติ", marker_color="rgba(37,99,235,0.55)",
        hovertemplate="%{x}<br>ปกติ: ฿%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=merged["month"], y=merged["anomaly"],
        name="Anomaly", marker_color="rgba(239,68,68,0.8)",
        hovertemplate="%{x}<br>Anomaly: ฿%{y:,.0f}<extra></extra>",
    ))

    # MoM annotation
    for _, row in merged.iterrows():
        if pd.notna(row["pct_change"]) and row["pct_change"] != 0:
            color = "#ef4444" if row["pct_change"] > 0 else "#10b981"
            sign  = "+" if row["pct_change"] > 0 else ""
            fig.add_annotation(
                x=row["month"], y=row["total"],
                text=f"{sign}{row['pct_change']:.0f}%",
                showarrow=False, yshift=10,
                font=dict(size=10, color=color),
            )

    fig.update_layout(
        barmode="stack",
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=0, r=0, t=24, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(37,99,235,0.06)", tickprefix="฿"),
        font=dict(family="sans-serif", size=12, color="#64748b"),
    )
    return fig


def chart_calendar_heatmap(df: pd.DataFrame) -> go.Figure:
    """Daily spending heatmap (calendar-style)."""
    daily = df.groupby("date")["amount"].sum().reset_index()
    daily["week"]    = daily["date"].dt.isocalendar().week.astype(int)
    daily["weekday"] = daily["date"].dt.weekday  # Mon=0
    daily["label"]   = daily["date"].dt.strftime("%d %b") + "<br>฿" + daily["amount"].apply(lambda x: f"{x:,.0f}")

    fig = go.Figure(go.Heatmap(
        x=daily["week"],
        y=daily["weekday"],
        z=daily["amount"],
        text=daily["label"],
        hovertemplate="%{text}<extra></extra>",
        colorscale=[[0, "rgba(37,99,235,0.05)"], [1, "rgba(37,99,235,0.85)"]],
        showscale=True,
        colorbar=dict(tickprefix="฿", len=0.8),
    ))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=0, r=0, t=8, b=0),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(7)),
            ticktext=["จ", "อ", "พ", "พฤ", "ศ", "ส", "อา"],
            showgrid=False,
        ),
        xaxis=dict(title="สัปดาห์", showgrid=False),
        font=dict(family="sans-serif", size=12, color="#64748b"),
        height=220,
    )
    return fig


def chart_anomaly_scatter(df: pd.DataFrame, iqr_cutoff: float) -> go.Figure:
    """IF score vs amount scatter — shows decision boundary."""
    normal  = df[~df["is_anomaly"]]
    anomaly = df[df["is_anomaly"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal["amount"], y=normal["if_score"],
        mode="markers", name="ปกติ",
        marker=dict(color="rgba(37,99,235,0.45)", size=6),
        hovertemplate="฿%{x:,.0f}<br>IF Score: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=anomaly["amount"], y=anomaly["if_score"],
        mode="markers", name="Anomaly",
        marker=dict(color="rgba(239,68,68,0.85)", size=9, symbol="x"),
        text=anomaly["anomaly_reason"],
        hovertemplate="฿%{x:,.0f}<br>IF Score: %{y:.3f}<br>%{text}<extra></extra>",
    ))
    # IQR cutoff line
    fig.add_vline(
        x=iqr_cutoff, line_dash="dash",
        line_color="rgba(245,158,11,0.7)", line_width=1.5,
        annotation_text=f"IQR Cutoff ฿{iqr_cutoff:,.0f}",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#92400e"),
    )
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
        xaxis=dict(title="จำนวนเงิน (฿)", tickprefix="฿", gridcolor="rgba(37,99,235,0.06)"),
        yaxis=dict(title="Isolation Forest Score", gridcolor="rgba(37,99,235,0.06)"),
        font=dict(family="sans-serif", size=12, color="#64748b"),
    )
    return fig


def chart_category_bar(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar: total spending per category."""
    cat_totals = (
        df.groupby("category")["amount"]
        .sum()
        .sort_values()
        .reset_index()
    )
    fig = go.Figure(go.Bar(
        x=cat_totals["amount"],
        y=cat_totals["category"],
        orientation="h",
        marker=dict(color=CAT_COLORS[:len(cat_totals)], opacity=0.85),
        hovertemplate="%{y}<br>฿%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=0, r=0, t=8, b=0),
        xaxis=dict(tickprefix="฿", gridcolor="rgba(37,99,235,0.06)"),
        yaxis=dict(showgrid=False),
        font=dict(family="sans-serif", size=12, color="#64748b"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# PRESCRIPTIVE ADVICE ENGINE
# ══════════════════════════════════════════════════════════════════

def render_advice(anomaly_df: pd.DataFrame):
    """Generate context-aware financial recommendations."""
    if anomaly_df.empty:
        st.success("✅ ไม่พบรายการผิดปกติในข้อมูลนี้")
        return

    cat_counts = anomaly_df.groupby("category").agg(
        count=("amount", "count"),
        total=("amount", "sum"),
    ).sort_values("total", ascending=False)

    st.markdown('<div class="section-header">💡 คำแนะนำเฉพาะบุคคล (Prescriptive AI)</div>', unsafe_allow_html=True)

    for cat, row in cat_counts.iterrows():
        advice = ADVICE_DB.get(cat, {
            "tip": f"รายจ่าย {cat} ผิดปกติ",
            "action": "ตรวจสอบและตั้งงบรายเดือนให้ชัดเจน",
            "level": "info",
        })
        css_class = {
            "danger": "advice-danger",
            "warn":   "advice-warn",
            "info":   "advice-box",
        }.get(advice["level"], "advice-box")

        icon = "🔴" if advice["level"] == "danger" else "🟡" if advice["level"] == "warn" else "🔵"

        st.markdown(f"""
        <div class="{css_class}">
            {icon} <strong>{cat}</strong> · {row['count']} รายการ · ฿{row['total']:,.0f}<br>
            <span style="opacity:0.8">{advice['tip']}</span><br>
            <strong>คำแนะนำ:</strong> {advice['action']}
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## 📊 FinScan")
        st.markdown("**AI-Powered Financial Behavior Optimization**")
        st.divider()

        st.markdown("### 🔬 Model Info")
        st.markdown("""
        **NLP Layer**  
        TF-IDF (char n-gram 2–4)  
        Max 50 features  

        **Anomaly Detection**  
        Isolation Forest  
        `n_estimators = 100`  
        `contamination = 5%`  

        **IQR Filter**  
        `Q3 + 2.5 × IQR`  
        ลด false positive  
        """)

        st.divider()
        st.markdown("### ⚙️ Pipeline")
        st.markdown("""
        1. 📥 Ingest CSV  
        2. 🔇 Filter noise  
        3. 🏷 NLP Categorize  
        4. 🌲 Isolation Forest  
        5. 📊 Visualize  
        6. 💡 Prescribe  
        """)

        st.divider()
        st.caption("KBank Cloud Pocket CSV · D/M/YYYY")


# ══════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════

def main():
    render_sidebar()

    # ── Header ────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:8px">
        <div style="width:48px;height:48px;border-radius:14px;background:linear-gradient(135deg,#2563eb,#0ea5e9);
                    display:flex;align-items:center;justify-content:center;font-size:22px">📊</div>
        <div>
            <h1 style="margin:0;font-size:26px;font-weight:700;color:#0f172a">FinScan</h1>
            <p style="margin:0;font-size:13px;color:#64748b">AI-Powered Personal Financial Behavior Optimization System</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Upload ─────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "อัปโหลดไฟล์ CSV จากธนาคาร (KBank Cloud Pocket)",
        type=["csv"],
        help="รองรับ format: Cloud Pocket Name, Type, Txn, CP Bal, Account Bal, Category, Memo, Date, Time, Note",
    )

    if uploaded is None:
        # Landing state
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="fin-card" style="text-align:center">
                <div style="font-size:32px">🌲</div>
                <h3 style="color:#2563eb;margin:8px 0 4px">Isolation Forest</h3>
                <p style="color:#64748b;font-size:13px">Unsupervised ML ตรวจจับรายการผิดปกติโดยไม่ต้อง label ข้อมูล</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="fin-card" style="text-align:center">
                <div style="font-size:32px">🔤</div>
                <h3 style="color:#2563eb;margin:8px 0 4px">TF-IDF NLP</h3>
                <p style="color:#64748b;font-size:13px">จัดหมวดหมู่รายจ่ายจาก Note/Memo ภาษาไทย-อังกฤษ</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="fin-card" style="text-align:center">
                <div style="font-size:32px">💡</div>
                <h3 style="color:#2563eb;margin:8px 0 4px">Prescriptive AI</h3>
                <p style="color:#64748b;font-size:13px">คำแนะนำการเงินเฉพาะบุคคลตาม pattern ที่พบ</p>
            </div>
            """, unsafe_allow_html=True)

        st.info("👆 เริ่มต้นโดยอัปโหลดไฟล์ CSV จาก KBank ด้านบน")
        return

    # ── Pipeline ───────────────────────────────────────────────────
    file_bytes = uploaded.read()

    with st.spinner("📥 Step 1/3 — กำลังโหลดและ clean ข้อมูล..."):
        df_clean = load_and_clean(file_bytes)
    if df_clean.empty:
        st.error("❌ ไม่พบรายการจ่ายเงินในไฟล์นี้ กรุณาตรวจสอบ format")
        return

    with st.spinner("🔤 Step 2/3 — NLP Categorization (TF-IDF)..."):
        df_cat = categorize_transactions(df_clean)

    with st.spinner("🌲 Step 3/3 — Isolation Forest Anomaly Detection..."):
        df_final, iqr_cutoff = detect_anomalies(df_cat)

    # ── KPI Metrics ────────────────────────────────────────────────
    anomalies  = df_final[df_final["is_anomaly"]]
    family_txn = df_final[df_final["is_family"]]
    total_amt  = df_final["amount"].sum()
    avg_amt    = df_final["amount"].mean()
    anom_amt   = anomalies["amount"].sum()

    st.markdown('<div class="section-header">📈 ภาพรวม</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("รายจ่ายทั้งหมด",  f"฿{total_amt:,.0f}", f"{len(df_final)} รายการ")
    c2.metric("Anomaly พบ",       str(len(anomalies)),   f"{len(anomalies)/len(df_final)*100:.1f}% ของทั้งหมด")
    c3.metric("ยอด Anomaly รวม",  f"฿{anom_amt:,.0f}",  f"{anom_amt/total_amt*100:.1f}% ของรายจ่าย")
    c4.metric("เฉลี่ย / ครั้ง",  f"฿{avg_amt:,.0f}")
    c5.metric("IQR Cutoff",       f"฿{iqr_cutoff:,.0f}", "เกณฑ์ผิดปกติ")

    st.divider()

    # ── Tabs ───────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🏷 Categories",
        "📅 Trends",
        "🌲 ML Explorer",
        "💡 Advice",
    ])

    # ── Tab 1: Overview ────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown('<div class="section-header">📅 Timeline รายจ่าย</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_timeline(df_final), use_container_width=True)

        with col_b:
            st.markdown('<div class="section-header">🍩 Category Breakdown</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_category_donut(df_final), use_container_width=True)

        st.markdown('<div class="section-header">🚨 รายการ Anomaly ทั้งหมด</div>', unsafe_allow_html=True)

        if anomalies.empty:
            st.success("✅ ไม่พบรายการผิดปกติ")
        else:
            display_cols = ["date", "Note", "category", "amount", "anomaly_reason", "if_score"]
            anom_display = anomalies[display_cols].copy()
            anom_display.columns = ["วันที่", "รายละเอียด", "หมวดหมู่", "จำนวนเงิน (฿)", "เหตุผล", "IF Score"]
            anom_display["จำนวนเงิน (฿)"] = anom_display["จำนวนเงิน (฿)"].apply(lambda x: f"฿{x:,.2f}")
            anom_display["IF Score"] = anom_display["IF Score"].apply(lambda x: f"{x:.4f}")
            anom_display["วันที่"] = anom_display["วันที่"].dt.strftime("%d %b %Y")
            st.dataframe(anom_display, use_container_width=True, hide_index=True)

        if not family_txn.empty:
            with st.expander(f"👨‍👩‍👧 รายการครอบครัว/สมยอม ({len(family_txn)} รายการ) — ไม่ถูก flag"):
                fam_display = family_txn[["date", "Note", "category", "amount"]].copy()
                fam_display["amount"] = fam_display["amount"].apply(lambda x: f"฿{x:,.2f}")
                fam_display["date"] = fam_display["date"].dt.strftime("%d %b %Y")
                st.dataframe(fam_display, use_container_width=True, hide_index=True)

    # ── Tab 2: Categories ──────────────────────────────────────────
    with tab2:
        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown('<div class="section-header">💰 รายจ่ายรวมต่อหมวด</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_category_bar(df_final), use_container_width=True)

        with col_d:
            st.markdown('<div class="section-header">🚨 Anomaly Amount ต่อหมวด</div>', unsafe_allow_html=True)
            anom_by_cat = (
                anomalies.groupby("category")["amount"]
                .sum().sort_values().reset_index()
            )
            if not anom_by_cat.empty:
                fig_acat = go.Figure(go.Bar(
                    x=anom_by_cat["amount"],
                    y=anom_by_cat["category"],
                    orientation="h",
                    marker_color="rgba(239,68,68,0.75)",
                    hovertemplate="%{y}<br>฿%{x:,.0f}<extra></extra>",
                ))
                fig_acat.update_layout(
                    paper_bgcolor="white", plot_bgcolor="white",
                    margin=dict(l=0, r=0, t=8, b=0),
                    xaxis=dict(tickprefix="฿", gridcolor="rgba(239,68,68,0.08)"),
                    yaxis=dict(showgrid=False),
                    font=dict(size=12, color="#64748b"),
                )
                st.plotly_chart(fig_acat, use_container_width=True)
            else:
                st.info("ไม่พบ anomaly ในข้อมูลนี้")

        # Category filter table
        st.markdown('<div class="section-header">🔍 กรองตามหมวดหมู่</div>', unsafe_allow_html=True)
        cats = ["ทั้งหมด"] + sorted(df_final["category"].unique().tolist())
        selected_cat = st.selectbox("เลือกหมวดหมู่", cats, label_visibility="collapsed")

        filtered = df_final if selected_cat == "ทั้งหมด" else df_final[df_final["category"] == selected_cat]
        tbl = filtered[["date", "Note", "category", "amount", "is_anomaly"]].copy()
        tbl["amount"] = tbl["amount"].apply(lambda x: f"฿{x:,.2f}")
        tbl["is_anomaly"] = tbl["is_anomaly"].map({True: "🔴 ANOMALY", False: "✅ ปกติ"})
        tbl["date"] = tbl["date"].dt.strftime("%d %b %Y")
        tbl.columns = ["วันที่", "รายละเอียด", "หมวดหมู่", "จำนวนเงิน", "สถานะ"]
        st.dataframe(tbl.sort_values("จำนวนเงิน", ascending=False), use_container_width=True, hide_index=True)

    # ── Tab 3: Trends ──────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">📅 รายจ่ายรายเดือน (MoM Comparison)</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_monthly_bar(df_final), use_container_width=True)

        st.markdown('<div class="section-header">🗓 Calendar Heatmap — ความถี่รายวัน</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_calendar_heatmap(df_final), use_container_width=True)

        # Category stacked monthly
        st.markdown('<div class="section-header">📊 สัดส่วนหมวดหมู่รายเดือน</div>', unsafe_allow_html=True)
        df_stack = df_final.copy()
        df_stack["month"] = df_stack["date"].dt.to_period("M").astype(str)
        pivot = df_stack.groupby(["month", "category"])["amount"].sum().reset_index()

        fig_stack = px.bar(
            pivot, x="month", y="amount", color="category",
            color_discrete_sequence=CAT_COLORS,
            labels={"amount": "฿", "month": "เดือน", "category": "หมวดหมู่"},
        )
        fig_stack.update_layout(
            paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(l=0, r=0, t=8, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1, font=dict(size=10)),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(37,99,235,0.06)", tickprefix="฿"),
            font=dict(size=12, color="#64748b"),
        )
        st.plotly_chart(fig_stack, use_container_width=True)

    # ── Tab 4: ML Explorer ─────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">🌲 Isolation Forest — Decision Boundary</div>', unsafe_allow_html=True)
        st.caption("แกน X = จำนวนเงิน · แกน Y = Isolation Forest Score (ยิ่งต่ำ = ผิดปกติมาก) · เส้นประ = IQR Cutoff")
        st.plotly_chart(chart_anomaly_scatter(df_final, iqr_cutoff), use_container_width=True)

        col_e, col_f = st.columns(2)

        with col_e:
            st.markdown('<div class="section-header">📐 Feature Engineering</div>', unsafe_allow_html=True)
            st.markdown("""
            | Feature | คำอธิบาย |
            |---|---|
            | `amount` | จำนวนเงินจริง |
            | `rolling_mean_7` | ค่าเฉลี่ย 7 วันย้อนหลัง |
            | `rolling_std_7` | Std dev 7 วัน |
            | `z_score` | (amount - mean) / std |
            | `day_of_week` | วันในสัปดาห์ |
            | `month` | เดือน |
            | `tfidf_0..49` | TF-IDF char n-gram features |
            """)

        with col_f:
            st.markdown('<div class="section-header">🔢 Model Stats</div>', unsafe_allow_html=True)
            total_n = len(df_final)
            anom_n  = len(anomalies)
            fam_n   = len(family_txn)

            st.markdown(f"""
            | Metric | Value |
            |---|---|
            | Total transactions | {total_n:,} |
            | Anomalies detected | {anom_n} ({anom_n/total_n*100:.1f}%) |
            | Family whitelisted | {fam_n} |
            | IQR Q3 cutoff | ฿{iqr_cutoff:,.0f} |
            | IF contamination | 5% |
            | IF n_estimators | 100 |
            | TF-IDF features | 50 |
            | Combined features | {5 + min(50, total_n)} |
            """)

        # Raw score distribution
        st.markdown('<div class="section-header">📊 Anomaly Score Distribution</div>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df_final[~df_final["is_anomaly"]]["if_score"],
            name="ปกติ", marker_color="rgba(37,99,235,0.55)",
            nbinsx=30, opacity=0.8,
        ))
        fig_hist.add_trace(go.Histogram(
            x=df_final[df_final["is_anomaly"]]["if_score"],
            name="Anomaly", marker_color="rgba(239,68,68,0.75)",
            nbinsx=30, opacity=0.8,
        ))
        fig_hist.update_layout(
            barmode="overlay",
            paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(l=0, r=0, t=8, b=0),
            xaxis=dict(title="IF Score", gridcolor="rgba(37,99,235,0.06)"),
            yaxis=dict(title="Count", gridcolor="rgba(37,99,235,0.06)"),
            legend=dict(orientation="h", yanchor="bottom", y=1),
            font=dict(size=12, color="#64748b"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Tab 5: Advice ──────────────────────────────────────────────
    with tab5:
        render_advice(anomalies)

        # Top 10 spenders
        st.markdown('<div class="section-header">🏆 Top 10 รายการสูงสุด</div>', unsafe_allow_html=True)
        top10 = df_final.nlargest(10, "amount")[["date", "Note", "category", "amount", "is_anomaly"]]
        top10 = top10.copy()
        top10["is_anomaly"] = top10["is_anomaly"].map({True: "🔴", False: "✅"})
        top10["amount"] = top10["amount"].apply(lambda x: f"฿{x:,.2f}")
        top10["date"] = top10["date"].dt.strftime("%d %b %Y")
        top10.columns = ["วันที่", "รายละเอียด", "หมวดหมู่", "จำนวนเงิน", ""]
        st.dataframe(top10, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
