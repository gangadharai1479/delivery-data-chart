# Optimized NSE dashboard: Enhanced performance and AMD insights
# Performance improvements:
#  - Vectorized operations for faster processing
#  - Optimized chart rendering with reduced complexity
#  - Streamlined data processing pipeline
# Enhanced AMD insights with visual indicators and summaries

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from nselib import capital_market as cm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, warnings, re
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from io import BytesIO, StringIO
from zipfile import ZipFile
import numpy as np

warnings.filterwarnings("ignore")

# ===== Class for Fast NSE Data Fetching with Caching =====
class NSEVisualizer:
    def __init__(self, cache_dir="nse_cache"):
        self.data = None
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, symbol, start_date, end_date):
        return os.path.join(self.cache_dir, f"{symbol}_{start_date}_to_{end_date}.csv")

    def fetch_data(self, symbol, start_date, end_date):
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        # Persistent per-symbol cache that grows over time
        master_cache = os.path.join(self.cache_dir, f"{symbol}_master.csv")

        cached_df = None
        if os.path.exists(master_cache):
            try:
                cached_df = pd.read_csv(master_cache, parse_dates=["DATE"])
                cached_df["DATE"] = pd.to_datetime(cached_df["DATE"]).dt.tz_localize(None)
            except Exception:
                cached_df = None

        # Determine business dates we need
        all_days = pd.date_range(start=start_date, end=end_date, freq="B")
        missing_days = []
        if cached_df is not None and not cached_df.empty:
            known_dates = set(pd.to_datetime(cached_df["DATE"]).dt.normalize())
            for d in all_days:
                if d.normalize() not in known_dates:
                    missing_days.append(d)
        else:
            missing_days = list(all_days)

        # Fetch missing in parallel (bounded threads)
        fetched_frames = []
        if missing_days:
            with ThreadPoolExecutor(max_workers=min(8, max(2, os.cpu_count() or 2))) as ex:
                futures = [ex.submit(self._fetch_one_day_symbol, symbol, d) for d in missing_days]
                for fut in as_completed(futures):
                    try:
                        df_day = fut.result()
                        if df_day is not None and not df_day.empty:
                            fetched_frames.append(df_day)
                    except Exception:
                        continue

        if fetched_frames:
            new_data = pd.concat(fetched_frames, ignore_index=True)
            if cached_df is not None and not cached_df.empty:
                combined = pd.concat([cached_df, new_data], ignore_index=True)
            else:
                combined = new_data
            combined = (
                combined.sort_values("DATE")
                .drop_duplicates(subset=["DATE", "SYMBOL"], keep="last")
                .reset_index(drop=True)
            )
            combined.to_csv(master_cache, index=False)
            cached_df = combined

        # Slice to requested window
        if cached_df is None or cached_df.empty:
            result = pd.DataFrame()
        else:
            mask = (cached_df["DATE"] >= pd.to_datetime(start_date)) & (
                cached_df["DATE"] <= pd.to_datetime(end_date)
            )
            result = cached_df.loc[mask].copy()

        self.data = result
        return result

    def _fetch_range(self, symbol, start_dt, end_dt):
        # Kept for compatibility if called elsewhere; now unused by fetch_data
        frames = []
        for d in pd.date_range(start=start_dt, end=end_dt, freq="B"):
            df = self._fetch_one_day_symbol(symbol, d)
            if df is not None and not df.empty:
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _fetch_one_day_symbol(self, symbol: str, date: pd.Timestamp):
        try:
            ds = pd.Timestamp(date).strftime("%d-%m-%Y")
            # Primary: capital market bhavcopy_with_delivery
            bhav = None
            try:
                bhav = cm.bhavcopy_with_delivery(ds)
            except Exception:
                bhav = None
            if bhav is None or getattr(bhav, "empty", True):
                bhav = self._fetch_from_nse_sec_bhav(pd.Timestamp(date).to_pydatetime())
            if bhav is None or bhav.empty:
                return None
            if "SYMBOL" not in bhav.columns:
                return None
            bhav["SYMBOL"] = bhav["SYMBOL"].astype(str).str.strip().str.upper()
            if "SERIES" in bhav.columns:
                bhav["SERIES"] = bhav["SERIES"].astype(str).str.strip().str.upper()
            sdata = bhav[bhav["SYMBOL"] == symbol.upper()]
            if "SERIES" in bhav.columns:
                sdata = sdata[sdata["SERIES"] == "EQ"]
            if sdata.empty:
                return None
            sdata = sdata.copy()
            sdata["DATE"] = pd.to_datetime(date)

            # Derive delivery % if needed
            qty_col = "TTL_TRD_QNTY" if "TTL_TRD_QNTY" in sdata.columns else ("TOT_TRD_QTY" if "TOT_TRD_QTY" in sdata.columns else None)
            if "DELIV_PER" in sdata.columns:
                sdata["DELIVERY_PCT"] = pd.to_numeric(sdata["DELIV_PER"], errors="coerce").round(2)
            elif qty_col and "DELIV_QTY" in sdata.columns:
                sdata["DELIVERY_PCT"] = (
                    pd.to_numeric(sdata["DELIV_QTY"], errors="coerce") / pd.to_numeric(sdata[qty_col], errors="coerce") * 100
                ).round(2)
            else:
                sdata["DELIVERY_PCT"] = pd.NA
            return sdata
        except Exception:
            return None

    def _fetch_from_nse_sec_bhav(self, dt: datetime) -> pd.DataFrame | None:
        # Try daily sec_bhavdata_full CSV
        url_daily = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{dt:%d%m%Y}.csv"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
            )
        }
        try:
            resp = requests.get(url_daily, headers=headers, timeout=20)
            if resp.status_code == 200 and resp.text.strip():
                df = pd.read_csv(StringIO(resp.text))
                return self._normalize_sec_bhav_columns(df)
        except Exception:
            pass

        # Try historical daily equity bhav zip (without delivery; only if sec file missing)
        # Note: delivery% not available here; we'll return None to skip rather than partial data
        return None

    def _normalize_sec_bhav_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Normalize headers
        def norm(c: str) -> str:
            c = c.strip().upper()
            c = re.sub(r"[^A-Z0-9]+", "_", c)
            return c

        df = df.rename(columns={c: norm(c) for c in df.columns})
        # Common mappings
        rename_map = {
            "OPEN_PRICE": "OPEN",
            "HIGH_PRICE": "HIGH",
            "LOW_PRICE": "LOW",
            "CLOSE_PRICE": "CLOSE",
            "LAST_PRICE": "LAST",
            "TOTAL_TRADES": "NO_OF_TRADES",
            "TOTAL_TRADE_QUANTITY": "TOT_TRD_QTY",
            "TOTAL_TRD_QTY": "TOT_TRD_QTY",
            "DELIVERABLE_QUANTITY": "DELIV_QTY",
            "DELIVERABLE_QTY": "DELIV_QTY",
            "DELIVERABLE__QTY": "DELIV_QTY",
            "DELIVERABLE_PERCENT": "DELIV_PER",
            "DELIVERABLE_PERC": "DELIV_PER",
        }
        for k, v in list(rename_map.items()):
            if k in df.columns and v not in df.columns:
                df.rename(columns={k: v}, inplace=True)
        # Ensure expected basic columns
        return df


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_symbol_list() -> list[str]:
    # NSE equity list (EQUITY_L.csv) – includes all tradable symbols
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
        )
    }
    try:
        df = pd.read_csv(url, headers=headers)
    except Exception:
        # Fallback via requests if pandas cannot pass headers param
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
    symbols = (
        df.get("SYMBOL", pd.Series(dtype=str)).dropna().astype(str).str.upper().unique().tolist()
    )
    symbols.sort()
    return symbols


# ===== Enhanced AMD Analysis Functions =====
@st.cache_data
def analyze_amd_patterns(df, price_col):
    """Vectorized AMD pattern analysis for faster processing"""
    if df.empty or price_col not in df.columns:
        return df, {}
    
    # Prepare data (vectorized operations)
    df = df.sort_values("DATE").reset_index(drop=True)
    df["Price"] = pd.to_numeric(df[price_col], errors="coerce")
    df["Delivery_PCT"] = pd.to_numeric(df["DELIVERY_PCT"], errors="coerce").fillna(0)
    df["Price_Change_%"] = df["Price"].pct_change() * 100
    df["Volume"] = pd.to_numeric(df.get("TOT_TRD_QTY", 0), errors="coerce")
    
    # Vectorized pattern classification
    conditions = [
        (df["Delivery_PCT"] >= 50) & (df["Price_Change_%"] >= 0),  # Accumulation
        (df["Delivery_PCT"] >= 50) & (df["Price_Change_%"] < 0),   # Distribution
        (df["Price_Change_%"].abs() > 5) & ((df["Delivery_PCT"] < 30) | (df["Delivery_PCT"] > 80))  # Manipulation
    ]
    choices = ["Accumulation", "Distribution", "Manipulation"]
    df["Market_Behavior"] = np.select(conditions, choices, default="Neutral")
    
    # Calculate AMD statistics
    pattern_counts = df["Market_Behavior"].value_counts()
    total_days = len(df)
    
    amd_stats = {
        "accumulation_days": pattern_counts.get("Accumulation", 0),
        "distribution_days": pattern_counts.get("Distribution", 0),
        "manipulation_days": pattern_counts.get("Manipulation", 0),
        "neutral_days": pattern_counts.get("Neutral", 0),
        "accumulation_pct": (pattern_counts.get("Accumulation", 0) / total_days * 100) if total_days > 0 else 0,
        "distribution_pct": (pattern_counts.get("Distribution", 0) / total_days * 100) if total_days > 0 else 0,
        "manipulation_pct": (pattern_counts.get("Manipulation", 0) / total_days * 100) if total_days > 0 else 0,
        "avg_delivery": df["Delivery_PCT"].mean(),
        "trend_direction": "Bullish" if df["Price_Change_%"].mean() > 0 else "Bearish",
        "volatility_level": "High" if df["Price_Change_%"].std() > 5 else "Moderate" if df["Price_Change_%"].std() > 2 else "Low"
    }
    
    return df, amd_stats


# ===== Enhanced Custom CSS =====
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        text-align: center;
        margin: 0;
    }
    
    /* Control panel styling */
    .control-panel {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.2);
    }
    
    .control-panel .stSelectbox > label,
    .control-panel .stNumberInput > label {
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* AMD Insights Panel */
    .amd-panel {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.3);
        color: white;
    }
    
    .amd-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .amd-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .amd-card {
        background: rgba(255, 255, 255, 0.15);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .amd-label {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .amd-value {
        font-size: 1.6rem;
        font-weight: 700;
    }
    
    .amd-value.accumulation { color: #2ecc71; }
    .amd-value.distribution { color: #e74c3c; }
    .amd-value.manipulation { color: #f39c12; }
    .amd-value.trend { color: #ecf0f1; }
    
    /* Metric cards */
    .metric-container {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(252, 182, 159, 0.3);
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
    }
    
    .metric-value.price { color: #27ae60; }
    .metric-value.percentage { color: #e67e22; }
    .metric-value.volatility { color: #e74c3c; }
    
    /* Chart container */
    .chart-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(168, 237, 234, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(79, 172, 254, 0.3);
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.4);
    }
    
    /* Input styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(10px);
    }
    
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(10px);
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    </style>
    """, unsafe_allow_html=True)

def create_header():
    st.markdown("""
    <div class="main-header">
        <h1>📈 NSE Stock Analysis Dashboard</h1>
        <p>Enhanced AMD Analysis with Real-time Price vs Delivery Insights</p>
    </div>
    """, unsafe_allow_html=True)

def create_control_panel(symbols):
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        symbol = st.selectbox(
            "🏢 Select NSE Symbol",
            options=symbols,
            index=(symbols.index("RELIANCE") if "RELIANCE" in symbols else 0),
            help="Choose from all NSE listed equity symbols"
        )
    
    with col2:
        days = st.number_input(
            "📅 Days to Analyze", 
            min_value=1, 
            max_value=365, 
            value=30,
            help="Number of business days to fetch"
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 Analyze Stock", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return symbol, days, analyze_btn

def create_metrics_display(avg_price, avg_delivery, high_price, low_price, volatility):
    st.markdown("""
    <div class="metric-container">
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Average Price</div>
                <div class="metric-value price">₹{:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Average Delivery %</div>
                <div class="metric-value percentage">{:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Highest Price</div>
                <div class="metric-value price">₹{:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Lowest Price</div>
                <div class="metric-value price">₹{:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Volatility (σ)</div>
                <div class="metric-value volatility">₹{:.2f}</div>
            </div>
        </div>
    </div>
    """.format(avg_price, avg_delivery, high_price, low_price, volatility), unsafe_allow_html=True)

def create_amd_insights_panel(amd_stats):
    """Enhanced AMD insights with visual indicators"""
    st.markdown(f"""
    <div class="amd-panel">
        <div class="amd-title">🎯 AMD Market Behavior Analysis</div>
        <div class="amd-grid">
            <div class="amd-card">
                <div class="amd-label">Accumulation Days</div>
                <div class="amd-value accumulation">{amd_stats['accumulation_days']}</div>
                <div class="amd-label">{amd_stats['accumulation_pct']:.1f}%</div>
            </div>
            <div class="amd-card">
                <div class="amd-label">Distribution Days</div>
                <div class="amd-value distribution">{amd_stats['distribution_days']}</div>
                <div class="amd-label">{amd_stats['distribution_pct']:.1f}%</div>
            </div>
            <div class="amd-card">
                <div class="amd-label">Manipulation Days</div>
                <div class="amd-value manipulation">{amd_stats['manipulation_days']}</div>
                <div class="amd-label">{amd_stats['manipulation_pct']:.1f}%</div>
            </div>
            <div class="amd-card">
                <div class="amd-label">Trend Direction</div>
                <div class="amd-value trend">{amd_stats['trend_direction']}</div>
            </div>
            <div class="amd-card">
                <div class="amd-label">Volatility Level</div>
                <div class="amd-value trend">{amd_stats['volatility_level']}</div>
            </div>
            <div class="amd-card">
                <div class="amd-label">Avg Delivery %</div>
                <div class="amd-value trend">{amd_stats['avg_delivery']:.1f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_optimized_chart(df, symbol, price_col):
    """Optimized chart rendering for faster performance"""
    # Reduce data points for faster rendering if dataset is large
    if len(df) > 100:
        # Sample every nth point to keep chart responsive while maintaining trend visibility
        step = max(1, len(df) // 80)
        df_chart = df.iloc[::step].copy()
    else:
        df_chart = df.copy()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Simplified price line (no fill for better performance)
    price_values = pd.to_numeric(df_chart[price_col], errors="coerce")
    fig.add_trace(
        go.Scatter(
            x=df_chart["DATE"],
            y=price_values,
            mode="lines",
            name="Close Price",
            line=dict(color="#667eea", width=3),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Close: ₹%{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    
    # Enhanced delivery % bars with pattern-based coloring
    delivery_values = pd.to_numeric(df_chart["DELIVERY_PCT"], errors="coerce")
    behavior_colors = {
        "Accumulation": "#2ecc71",
        "Distribution": "#e74c3c", 
        "Manipulation": "#f39c12",
        "Neutral": "#95a5a6"
    }
    bar_colors = [behavior_colors.get(behavior, "#95a5a6") for behavior in df_chart["Market_Behavior"]]
    
    fig.add_trace(
        go.Bar(
            x=df_chart["DATE"],
            y=delivery_values,
            name="Delivery %",
            marker=dict(color=bar_colors, opacity=0.7),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Delivery: %{y:.2f}%<br>Pattern: %{customdata}<extra></extra>",
            customdata=df_chart["Market_Behavior"]
        ),
        secondary_y=True,
    )

    # Optimized layout
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b> - Price vs Delivery % with AMD Patterns",
            font=dict(size=20, color="#2c3e50"),
            x=0.5
        ),
        template="plotly_white",
        legend=dict(
            x=0.02, y=0.98, 
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.1)",
            borderwidth=1
        ),
        hovermode="x unified",
        bargap=0.3,
        height=500,  # Reduced height for faster rendering
        showlegend=True
    )
    
    fig.update_yaxes(title_text="<b>Close Price (₹)</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Delivery %</b>", secondary_y=True, range=[0, 100])
    
    return fig

# ===== Streamlit UI =====
st.set_page_config(
    page_title="NSE Stock Analysis Dashboard", 
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom styling
apply_custom_css()

# Create beautiful header
create_header()

# Load symbols with spinner
with st.spinner("🔄 Loading NSE symbols..."):
    symbols = load_symbol_list()

# Control panel
symbol, days, analyze_btn = create_control_panel(symbols)

# Analysis section
if analyze_btn:
    with st.spinner(f"📊 Fetching and analyzing data for {symbol}..."):
        vis = NSEVisualizer()
        end_dt = datetime.now()
        # Use business-day offset for more accurate trading window
        start_dt = (pd.Timestamp(end_dt) - pd.tseries.offsets.BDay(int(days))).to_pydatetime()
        df = vis.fetch_data(symbol, start_dt, end_dt)

    if df.empty:
        st.error("❌ No data found for the selected symbol and date range. Please try different parameters.")
        st.info("Tip: try a shorter 'Days to Analyze' (e.g., 10), or test with another symbol. If network or NSE archive is unavailable, data may be missing.")
    else:
        st.success(f"✅ Successfully fetched {len(df)} trading days for **{symbol}**")

        # Determine price column and perform AMD analysis
        price_col = "CLOSE" if "CLOSE" in df.columns else ("CLOSE_PRICE" if "CLOSE_PRICE" in df.columns else None)
        if price_col is None:
            st.error("❌ Price column not found in data returned by NSE.")
        else:
            # Enhanced AMD analysis with caching for performance
            df_analyzed, amd_stats = analyze_amd_patterns(df, price_col)
            
            # Calculate summary metrics
            price_series = pd.to_numeric(df_analyzed[price_col], errors="coerce")
            avg_price = price_series.mean()
            avg_delivery = pd.to_numeric(df_analyzed["DELIVERY_PCT"], errors="coerce").mean()
            high_price = price_series.max()
            low_price = price_series.min()
            volatility = price_series.std()

            # Display enhanced AMD insights panel
            create_amd_insights_panel(amd_stats)

            # Display beautiful metrics
            create_metrics_display(avg_price, avg_delivery, high_price, low_price, volatility)

            # Optimized interactive chart
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            fig = create_optimized_chart(df_analyzed, symbol, price_col)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

            # AMD Pattern Summary with insights
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 🔍 Pattern Insights")
                if amd_stats['accumulation_pct'] > 40:
                    st.success(f"🟢 **Strong Accumulation** ({amd_stats['accumulation_pct']:.1f}%) - Institutional buying likely")
                elif amd_stats['distribution_pct'] > 40:
                    st.warning(f"🟡 **High Distribution** ({amd_stats['distribution_pct']:.1f}%) - Possible profit booking")
                elif amd_stats['manipulation_pct'] > 30:
                    st.error(f"🔴 **Manipulation Detected** ({amd_stats['manipulation_pct']:.1f}%) - Exercise caution")
                else:
                    st.info(f"ℹ️ **Neutral Market** - Balanced trading activity")
            
            with col2:
                st.markdown("### 📊 Market Summary")
                trend_emoji = "📈" if amd_stats['trend_direction'] == "Bullish" else "📉"
                vol_emoji = "🌊" if amd_stats['volatility_level'] == "High" else "〰️" if amd_stats['volatility_level'] == "Moderate" else "📏"
                
                st.write(f"{trend_emoji} **Trend**: {amd_stats['trend_direction']}")
                st.write(f"{vol_emoji} **Volatility**: {amd_stats['volatility_level']}")
                st.write(f"📦 **Avg Delivery**: {amd_stats['avg_delivery']:.1f}%")

            # Enhanced data table with pattern highlighting
            st.subheader("📊 Detailed Data Analysis")

            # Prepare optimized display table
            display_cols = ["DATE", "SYMBOL", price_col, "DELIVERY_PCT", "Price_Change_%", "Market_Behavior"]
            display_df = df_analyzed[display_cols].copy()
            
            # Format columns for better display
            display_df["DATE"] = pd.to_datetime(display_df["DATE"]).dt.strftime("%Y-%m-%d")
            display_df[price_col] = pd.to_numeric(display_df[price_col], errors="coerce").round(2)
            display_df["DELIVERY_PCT"] = pd.to_numeric(display_df["DELIVERY_PCT"], errors="coerce").round(2)
            display_df["Price_Change_%"] = pd.to_numeric(display_df["Price_Change_%"], errors="coerce").round(2)

            # Enhanced table display with styling
            def highlight_patterns(row):
                if row["Market_Behavior"] == "Accumulation":
                    return ['background-color: rgba(46, 204, 113, 0.2)'] * len(row)
                elif row["Market_Behavior"] == "Distribution":
                    return ['background-color: rgba(231, 76, 60, 0.2)'] * len(row)
                elif row["Market_Behavior"] == "Manipulation":
                    return ['background-color: rgba(243, 156, 18, 0.2)'] * len(row)
                else:
                    return [''] * len(row)

            # Display with enhanced formatting
            styled_df = display_df.style.apply(highlight_patterns, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=400)

            # Download section
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                csv = df_analyzed.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Enhanced Data with AMD Analysis",
                    csv, 
                    f"{symbol}_amd_analysis_{datetime.now().strftime('%Y%m%d')}.csv", 
                    "text/csv",
                    use_container_width=True
                )

            # Additional AMD Pattern Distribution Chart
            st.markdown("### 🥧 AMD Pattern Distribution")
            
            pattern_counts = df_analyzed["Market_Behavior"].value_counts()
            colors = ["#2ecc71", "#e74c3c", "#f39c12", "#95a5a6"]  # Green, Red, Orange, Gray
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=pattern_counts.index,
                values=pattern_counts.values,
                hole=0.4,
                marker=dict(colors=colors, line=dict(color='white', width=2)),
                textinfo='label+percent',
                textfont_size=12,
                hovertemplate="<b>%{label}</b><br>Days: %{value}<br>Percentage: %{percent}<extra></extra>"
            )])
            
            fig_pie.update_layout(
                title=dict(
                    text=f"<b>{symbol}</b> - Market Behavior Distribution",
                    font=dict(size=18, color="#2c3e50"),
                    x=0.5
                ),
                height=400,
                showlegend=True,
                legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.1)
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)

# Footer
st.markdown("""
---
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><i>Soli Deo Gloria</i> • Enhanced with AMD Intelligence</p>
</div>
""", unsafe_allow_html=True)
