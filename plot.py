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
        fetched_frames: list[pd.DataFrame] = []
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


# ===== Custom CSS for Beautiful UI =====
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
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
    
    .metric-value.price {
        color: #27ae60;
    }
    
    .metric-value.percentage {
        color: #e67e22;
    }
    
    .metric-value.volatility {
        color: #e74c3c;
    }
    
    /* Chart container */
    .chart-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(168, 237, 234, 0.3);
    }
    
    /* Success/Error messages */
    .stAlert > div {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
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
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        <p>Real-time Price vs Delivery Percentage Analysis</p>
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
        st.markdown("<br>", unsafe_allow_html=True)  # spacing
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
    with st.spinner(f"📊 Fetching data for {symbol}..."):
        vis = NSEVisualizer()
        end_dt = datetime.now()
        # Use business-day offset for more accurate trading window
        start_dt = (pd.Timestamp(end_dt) - pd.tseries.offsets.BDay(int(days))).to_pydatetime()
        df = vis.fetch_data(symbol, start_dt, end_dt)

    if df.empty:
        st.error("❌ No data found for the selected symbol and date range. Please try different parameters.")
    else:
        st.success(f"✅ Successfully fetched {len(df)} trading days for **{symbol}**")

        # Determine price column and summary stats (capital market bhav uses CLOSE)
        price_col = "CLOSE" if "CLOSE" in df.columns else ("CLOSE_PRICE" if "CLOSE_PRICE" in df.columns else None)
        if price_col is None:
            st.error("❌ Price column not found in data returned by NSE.")
        else:
            price_series = pd.to_numeric(df[price_col], errors="coerce")
            avg_price = price_series.mean()
            avg_delivery = pd.to_numeric(df["DELIVERY_PCT"], errors="coerce").mean()
            high_price = price_series.max()
            low_price = price_series.min()
            volatility = price_series.std()

        # Display beautiful metrics
        create_metrics_display(avg_price, avg_delivery, high_price, low_price, volatility)

        # Interactive dual-axis Plotly chart with enhanced styling
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Enhanced price line with gradient colors
        if price_col is not None:
            fig.add_trace(
                go.Scatter(
                    x=df["DATE"],
                    y=pd.to_numeric(df[price_col], errors="coerce"),
                    mode="lines+markers",
                    name="Close Price",
                    line=dict(
                        color="#667eea", 
                        width=4,
                        shape="spline",
                        smoothing=0.3
                    ),
                    marker=dict(
                        size=6,
                        color="#764ba2",
                        line=dict(width=2, color="white")
                    ),
                    hovertemplate="<b>%{x|%d %b %Y}</b><br>Close: ₹%{y:.2f}<extra></extra>",
                    fill="tonexty" if len(df) > 1 else None,
                    fillcolor="rgba(102, 126, 234, 0.1)"
                ),
                secondary_y=False,
            )
        
        # Enhanced delivery % bars with conditional coloring
        delivery_values = pd.to_numeric(df["DELIVERY_PCT"], errors="coerce")
        bar_colors = []
        for val in delivery_values:
            if pd.isna(val):
                bar_colors.append("#cccccc")  # Gray for missing data
            elif val > 70:
                bar_colors.append("#2ecc71")  # Bright Green for >70%
            elif val >= 50:
                bar_colors.append("#f39c12")  # Orange-Yellow for 50-70%
            else:
                bar_colors.append("#e74c3c")  # Red for <50%
        
        fig.add_trace(
            go.Bar(
                x=df["DATE"],
                y=delivery_values,
                name="Delivery %",
                marker=dict(
                    color=bar_colors,
                    opacity=0.8,
                    line=dict(width=1, color="white")
                ),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Delivery: %{y:.2f}%<extra></extra>",
            ),
            secondary_y=True,
        )

        # Enhanced layout with modern styling
        price_values = pd.to_numeric(df[price_col], errors="coerce") if price_col else pd.Series([])
        if not price_values.empty:
            price_min = price_values.min()
            price_max = price_values.max()
            price_range = price_max - price_min
            price_buffer = max(price_range * 0.05, 10)  # 5% buffer or minimum 10 rupees
        else:
            price_min, price_max, price_buffer = 0, 100, 10
        
        fig.update_layout(
            title=dict(
                text=f"<b>{symbol}</b> - Price vs Delivery % Analysis",
                font=dict(size=24, color="#2c3e50"),
                x=0.5
            ),
            template="plotly_white",
            legend=dict(
                x=0.02, y=0.98, 
                orientation="v",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.1)",
                borderwidth=1,
                font=dict(size=12)
            ),
            hovermode="x unified",
            bargap=0.2,
            xaxis=dict(
                title="<b>Date</b>", 
                showgrid=True,
                gridcolor="rgba(0, 0, 0, 0.1)"
            ),
            uirevision="keep",  # keep zoom/selection when rerunning
            plot_bgcolor="rgba(255, 255, 255, 0.8)",
            paper_bgcolor="rgba(255, 255, 255, 0.9)",
            height=600
        )
        
        fig.update_yaxes(
            title_text="<b>Close Price (₹)</b>", 
            secondary_y=False,
            showgrid=True,
            gridcolor="rgba(102, 126, 234, 0.2)",
            range=[price_min - price_buffer, price_max + price_buffer]
        )
        fig.update_yaxes(
            title_text="<b>Delivery %</b>", 
            secondary_y=True,
            showgrid=False,
            range=[0, 100]  # Keep delivery % range fixed at 0-100%
        )

        # Remove gaps for weekends and NSE holidays not present in data
        try:
            observed = pd.to_datetime(df["DATE"]).dt.normalize().unique()
            all_days = pd.date_range(pd.to_datetime(df["DATE"]).min().normalize(), pd.to_datetime(df["DATE"]).max().normalize(), freq="D")
            missing = [d for d in all_days if d.to_datetime64() not in set(observed)]
            fig.update_xaxes(rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
                dict(values=missing),           # hide other missing dates (holidays)
            ])
        except Exception:
            pass

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Enhanced download section
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Data as CSV",
                csv, 
                f"{symbol}_price_delivery_{datetime.now().strftime('%Y%m%d')}.csv", 
                "text/csv",
                use_container_width=True
            )

# Footer
st.markdown("""
---
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><i>Soli Deo Gloria</i></p>
</div>
""", unsafe_allow_html=True)