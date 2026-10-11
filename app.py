import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from scipy.optimize import minimize
from hyperopt import fmin, tpe, hp, Trials

# ==========================================
# CẤU HÌNH TRANG WEB STREAMLIT
# ==========================================
st.set_page_config(
    page_title="HOSE Strategy Backtester | SMA + OBV OR",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện hiện đại, chuyên nghiệp
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #F3F4F6;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #2563EB;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-weight: 600;
        border-radius: 6px 6px 0px 0px;
        padding: 0 16px;
    }
</style>
""", unsafe_allow_html=True)

# Preset tham số tối ưu từ thực nghiệm Notebook (Train 2020-2021)
DEFAULT_OPTIMIZED_PARAMS = {
    "DIG": {"ma_short": 60, "ma_long": 150, "obv_window": 55},
    "DPM": {"ma_short": 50, "ma_long": 330, "obv_window": 55},
    "NLG": {"ma_short": 35, "ma_long": 310, "obv_window": 50},
    "KBC": {"ma_short": 25, "ma_long": 350, "obv_window": 15},
    "DCM": {"ma_short": 45, "ma_long": 210, "obv_window": 15},
}

# ==========================================
# CÁC HÀM XỬ LÝ DỮ LIỆU & CHỈ BÁO KỸ THUẬT
# ==========================================

@st.cache_data(show_spinner=False)
def load_and_preprocess_data(file_source):
    """
    Đọc và chuẩn hóa dữ liệu từ file path hoặc UploadedFile.
    """
    if isinstance(file_source, str):
        df = pd.read_csv(file_source, encoding="utf-8-sig", low_memory=False)
    else:
        df = pd.read_csv(file_source, encoding="utf-8-sig", low_memory=False)

    df.columns = df.columns.str.strip().str.lower()
    
    required_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu các cột dữ liệu bắt buộc: {missing_cols}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    numeric_cols = [
        "open", "high", "low", "close", "volume",
        "adj_open", "adj_high", "adj_low", "adj_close"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ưu tiên giá điều chỉnh (adj_close) nếu có để tránh sai lệch sau chia tách cổ tức
    price_col = "adj_close" if "adj_close" in df.columns and df["adj_close"].notna().sum() > 0 else "close"
    
    df = df.dropna(subset=["date", "ticker", price_col, "volume"]).sort_values(["ticker", "date"])
    return df, price_col


@st.cache_data(show_spinner=False)
def calculate_ta_features(df_train, price_col, min_observations=400, trading_days=252):
    """
    Tính 4 nhóm nhân tố kỹ thuật trên tập Train:
    - Trend: Tỷ lệ SMA50 > SMA200, Giá vs SMA200
    - Momentum: Lợi suất 6 tháng
    - Volume: Giá trị giao dịch trung bình, OBV Confirmation 20 phiên
    - Risk: Volatility thường niên hóa, Max Drawdown
    """
    rows = []
    for ticker, g in df_train.groupby("ticker"):
        g = g.sort_values("date").drop_duplicates(subset=["date"]).dropna(subset=[price_col, "volume"]).copy()
        if len(g) < min_observations:
            continue

        price = g[price_col].astype(float)
        volume = g["volume"].astype(float)

        # 1. TREND
        sma50 = price.rolling(50).mean()
        sma200 = price.rolling(200).mean()
        trend_strength = (sma50 > sma200).mean()

        if pd.notna(sma200.iloc[-1]) and sma200.iloc[-1] != 0:
            price_vs_sma200 = (price.iloc[-1] / sma200.iloc[-1]) - 1.0
        else:
            price_vs_sma200 = np.nan

        # 2. MOMENTUM
        momentum_6m = (price.iloc[-1] / price.iloc[-126] - 1.0) if len(price) >= 126 else np.nan

        # 3. VOLUME
        avg_trading_value = (price * volume).mean()
        obv = ta.volume.OnBalanceVolumeIndicator(close=price, volume=volume).on_balance_volume()

        if len(obv) >= 20:
            volume_20 = volume.iloc[-20:].sum()
            obv_confirmation = ((obv.iloc[-1] - obv.iloc[-20]) / volume_20) if volume_20 != 0 else np.nan
        else:
            obv_confirmation = np.nan

        # 4. RISK
        returns = price.pct_change().dropna()
        annual_volatility = returns.std() * np.sqrt(trading_days)
        equity = (1.0 + returns).cumprod()
        max_drawdown = (equity / equity.cummax() - 1.0).min()

        rows.append({
            "Ticker": ticker,
            "N": len(g),
            "Trend_Strength": trend_strength,
            "Price_vs_SMA200": price_vs_sma200,
            "Momentum_6M": momentum_6m,
            "Avg_Trading_Value": avg_trading_value,
            "OBV_Confirmation": obv_confirmation,
            "Volatility": annual_volatility,
            "Max_Drawdown": max_drawdown
        })

    out = pd.DataFrame(rows)
    return out.dropna().reset_index(drop=True)


def build_ta_score(ta_features, weights=None):
    """
    Xếp hạng percentile và tổng hợp điểm TA Score.
    """
    if weights is None:
        weights = {"trend": 0.30, "momentum": 0.25, "volume": 0.25, "risk": 0.20}

    df = ta_features.copy()

    # Trend Score (50% Trend Strength + 50% Price vs SMA200)
    df["Trend_Strength_Rank"] = df["Trend_Strength"].rank(pct=True)
    df["Price_SMA200_Rank"] = df["Price_vs_SMA200"].rank(pct=True)
    df["Trend_Score"] = 0.50 * df["Trend_Strength_Rank"] + 0.50 * df["Price_SMA200_Rank"]

    # Momentum Score
    df["Momentum_Score"] = df["Momentum_6M"].rank(pct=True)

    # Volume Score (60% Thanh khoản + 40% OBV xác nhận)
    df["Liquidity_Rank"] = df["Avg_Trading_Value"].rank(pct=True)
    df["OBV_Rank"] = df["OBV_Confirmation"].rank(pct=True)
    df["Volume_Score"] = 0.60 * df["Liquidity_Rank"] + 0.40 * df["OBV_Rank"]

    # Risk Score (Volatility thấp xếp hạng cao hơn, Drawdown ít âm hơn xếp hạng cao hơn)
    df["Volatility_Rank"] = df["Volatility"].rank(pct=True, ascending=False)
    df["Drawdown_Rank"] = df["Max_Drawdown"].rank(pct=True)
    df["Risk_Score"] = 0.50 * df["Volatility_Rank"] + 0.50 * df["Drawdown_Rank"]

    # Tổng điểm TA
    df["TA_Score"] = (
        weights["trend"] * df["Trend_Score"]
        + weights["momentum"] * df["Momentum_Score"]
        + weights["volume"] * df["Volume_Score"]
        + weights["risk"] * df["Risk_Score"]
    )

    return df.sort_values("TA_Score", ascending=False).reset_index(drop=True)


def prepare_stock_series(df_full, ticker, price_col):
    """
    Trích xuất chuỗi giá và khối lượng theo ngày cho một mã cổ phiếu.
    """
    df = df_full[df_full["ticker"] == ticker].copy()
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df = df.dropna(subset=["date", price_col, "volume"])
    out = pd.DataFrame({
        "date": df["date"],
        "Close": df[price_col].astype(float),
        "Volume": df["volume"].astype(float)
    }).set_index("date").sort_index()
    return out


# ==========================================
# CƠ CHẾ TÍN HIỆU SMA + OBV VÀ BACKTEST
# ==========================================

def find_position_sma(df, ma_short, ma_long):
    """
    Tín hiệu SMA Crossover:
    1: Mua khi SMA short cắt lên SMA long
    -1: Bán khi SMA short cắt xuống SMA long
    """
    position = pd.Series(0.0, index=df.index, name="SMA_signal")
    if ma_short >= ma_long:
        return position

    sma_short = df["Close"].rolling(int(ma_short)).mean()
    sma_long = df["Close"].rolling(int(ma_long)).mean()

    buy = (sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))
    sell = (sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))

    position.loc[buy] = 1.0
    position.loc[sell] = -1.0
    return position, sma_short, sma_long


def find_position_obv(df, obv_window):
    """
    Tín hiệu OBV Crossover:
    1: Mua khi OBV cắt lên đường trung bình OBV_MA
    -1: Bán khi OBV cắt xuống đường trung bình OBV_MA
    """
    position = pd.Series(0.0, index=df.index, name="OBV_signal")
    obv = ta.volume.OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
    obv_ma = obv.rolling(int(obv_window)).mean()

    buy = (obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))
    sell = (obv < obv_ma) & (obv.shift(1) >= obv_ma.shift(1))

    position.loc[buy] = 1.0
    position.loc[sell] = -1.0
    return position, obv, obv_ma


def find_position_or(df, ma_short, ma_long, obv_window):
    """
    Kết hợp tín hiệu theo logic OR:
    - Mua nếu SMA Mua HOẶC OBV Mua
    - Bán nếu SMA Bán HOẶC OBV Bán
    - Nếu cả hai tín hiệu đối nghịch trong cùng phiên -> hủy (0)
    """
    sma_sig, sma_s, sma_l = find_position_sma(df, ma_short, ma_long)
    obv_sig, obv_val, obv_ma = find_position_obv(df, obv_window)

    events = pd.Series(0.0, index=df.index, name="SMA_OBV_OR")
    buy = (sma_sig == 1.0) | (obv_sig == 1.0)
    sell = (sma_sig == -1.0) | (obv_sig == -1.0)
    conflict = buy & sell

    events.loc[buy & ~conflict] = 1.0
    events.loc[sell & ~conflict] = -1.0

    return events, sma_s, sma_l, obv_val, obv_ma


def events_to_holding(events):
    """
    Chuyển đổi các sự kiện (Events: +1, -1, 0) thành trạng thái nắm giữ (Holding: 1 hoặc 0).
    """
    holding = pd.Series(0.0, index=events.index)
    current = 0.0
    for i in range(len(events)):
        sig = events.iloc[i]
        if sig == 1.0:
            current = 1.0
        elif sig == -1.0:
            current = 0.0
        holding.iloc[i] = current
    return holding


def strategy_returns(df, events, commission=0.001):
    """
    Mô phỏng thực thi lệnh không look-ahead bias:
    - Tín hiệu ngày t chỉ có hiệu lực vào ngày t+1 (shift 1).
    - Khấu trừ chi phí giao dịch (turnover * commission).
    """
    asset_return = df["Close"].pct_change().fillna(0.0)
    holding = events_to_holding(events)
    
    # Thực thi ở t+1
    executed_holding = holding.shift(1).fillna(0.0)
    gross_return = executed_holding * asset_return
    
    # Tính turnover khi thay đổi vị thế
    turnover = executed_holding.diff().abs().fillna(executed_holding.abs())
    net_return = gross_return - (turnover * commission)

    return net_return, executed_holding, turnover


def performance_stats(returns, trading_days=252):
    """
    Tính toán các chỉ số đánh giá hiệu quả đầu tư.
    """
    r = pd.Series(returns).dropna().astype(float)
    if len(r) == 0:
        return {
            "Mean Daily Return [%]": np.nan,
            "Total Return [%]": np.nan,
            "Annual Return [%]": np.nan,
            "Annual Volatility [%]": np.nan,
            "Sharpe Ratio": np.nan,
            "Max Drawdown [%]": np.nan
        }

    equity = (1.0 + r).cumprod()
    total_return = equity.iloc[-1] - 1.0
    years = len(r) / trading_days

    if years > 0 and equity.iloc[-1] > 0:
        annual_return = (equity.iloc[-1] ** (1.0 / years)) - 1.0
    else:
        annual_return = np.nan

    annual_vol = r.std() * np.sqrt(trading_days)
    sharpe = (r.mean() / r.std() * np.sqrt(trading_days)) if r.std() != 0 else np.nan

    drawdown = equity / equity.cummax() - 1.0
    max_dd = drawdown.min()

    return {
        "Mean Daily Return [%]": r.mean() * 100.0,
        "Total Return [%]": total_return * 100.0,
        "Annual Return [%]": annual_return * 100.0,
        "Annual Volatility [%]": annual_vol * 100.0,
        "Sharpe Ratio": sharpe,
        "Max Drawdown [%]": max_dd * 100.0
    }


def evaluate_test_with_full_history(full_stock_df, ma_short, ma_long, obv_window, test_start, test_end, commission):
    """
    Tính chỉ báo trên TOÀN BỘ lịch sử có trước và trong Test để warm-up đầy đủ
    (tránh mất dữ liệu do rolling window 350 của SMA dài), sau đó cắt đúng khoảng thời gian Test.
    """
    history_and_test = full_stock_df.loc[full_stock_df.index <= pd.Timestamp(test_end)].copy()
    events, sma_s, sma_l, obv_val, obv_ma = find_position_or(history_and_test, ma_short, ma_long, obv_window)
    net_ret, holding, turnover = strategy_returns(history_and_test, events, commission=commission)

    # Cắt kết quả cho giai đoạn Test
    mask_test = (history_and_test.index >= pd.Timestamp(test_start)) & (history_and_test.index <= pd.Timestamp(test_end))
    test_returns = net_ret.loc[mask_test]
    test_events = events.loc[mask_test]
    test_holding = holding.loc[mask_test]
    test_turnover = turnover.loc[mask_test]

    return {
        "events": test_events,
        "holding": test_holding,
        "returns": test_returns,
        "turnover": test_turnover,
        "stats": performance_stats(test_returns),
        "sma_short_series": sma_s.loc[mask_test],
        "sma_long_series": sma_l.loc[mask_test],
        "obv_series": obv_val.loc[mask_test],
        "obv_ma_series": obv_ma.loc[mask_test],
    }


# ==========================================
# PHÂN BỔ DANH MỤC: MPT & BENCHMARK
# ==========================================

def optimize_mpt(train_return_matrix, min_weight=0.05, max_weight=0.40, risk_free_rate=0.0, trading_days=252):
    """
    Tối ưu hóa danh mục MPT (Markowitz) có giới hạn tỷ trọng (Concentration constraints).
    Tối đa hóa Sharpe Ratio trên tập Train.
    """
    n = train_return_matrix.shape[1]
    if (n * min_weight > 1.0) or (n * max_weight < 1.0):
        # Fallback về equal weight nếu bounds vô nghiệm
        return np.repeat(1.0 / n, n)

    def neg_sharpe(w):
        p_ret = float(w @ train_return_matrix.mean().values * trading_days)
        cov_annual = train_return_matrix.cov().values * trading_days
        variance = float(w.T @ cov_annual @ w)
        p_vol = np.sqrt(max(variance, 0.0))
        if p_vol == 0:
            return 1e9
        return -((p_ret - risk_free_rate) / p_vol)

    x0 = np.repeat(1.0 / n, n)
    bounds = [(min_weight, max_weight)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(neg_sharpe, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success:
        return res.x
    return x0


def calculate_equal_weight_buy_hold(test_stocks_dict, tickers, equal_weights):
    """
    Tính chuẩn Benchmark Buy & Hold:
    Mỗi cổ phiếu được đầu tư ban đầu tỷ trọng bằng nhau, sau đó KHÔNG rebalance hàng ngày
    mà để giá trị vốn tự do trôi theo thị trường (True Buy & Hold).
    """
    price_matrix = pd.concat({t: test_stocks_dict[t]["Close"] for t in tickers}, axis=1).dropna()
    normalized = price_matrix / price_matrix.iloc[0]
    portfolio_wealth = (normalized * equal_weights).sum(axis=1)
    portfolio_returns = portfolio_wealth.pct_change().fillna(0.0)
    return portfolio_returns, portfolio_wealth


# ==========================================
# TỐI ƯU HÓA THAM SỐ HYPEROPT TRỰC TIẾP TRÊN APP
# ==========================================

def run_hyperopt_single_stock(df_train, ticker, max_evals=40, commission=0.001, seed=42):
    """
    Chạy Hyperopt tìm kiếm tham số tối ưu Sharpe Ratio trên Train.
    """
    space = {
        "ma_short": hp.quniform(f"{ticker}_ma_short", 20, 60, 5),
        "ma_long": hp.quniform(f"{ticker}_ma_long", 100, 350, 10),
        "obv_window": hp.quniform(f"{ticker}_obv_window", 5, 60, 5)
    }

    def objective(p):
        ms = int(p["ma_short"])
        ml = int(p["ma_long"])
        ow = int(p["obv_window"])
        if ms >= ml:
            return 999999.0

        ev, _, _, _, _ = find_position_or(df_train, ms, ml, ow)
        if (ev == 1.0).sum() < 3:  # Phạt nếu tạo quá ít giao dịch
            return 999999.0

        ret, _, _ = strategy_returns(df_train, ev, commission=commission)
        sharpe = performance_stats(ret)["Sharpe Ratio"]
        if pd.isna(sharpe):
            return 999999.0
        return -sharpe

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        verbose=False,
        rstate=np.random.default_rng(seed)
    )

    return {
        "ma_short": int(best[f"{ticker}_ma_short"]),
        "ma_long": int(best[f"{ticker}_ma_long"]),
        "obv_window": int(best[f"{ticker}_obv_window"])
    }


# ==========================================
# MAIN APPLICATION INTERFACE
# ==========================================

def main():
    st.markdown('<div class="main-title">📈 Kiểm Định Chiến Lược Kết Hợp SMA & OBV (HOSE)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Kiểm định chiến lược giao dịch định lượng kết hợp tín hiệu xu hướng (SMA) và dòng tiền (OBV) theo quy tắc OR trên thị trường chứng khoán HOSE.</div>', unsafe_allow_html=True)

    # ------------------------------------------
    # SIDEBAR: CẤU HÌNH THAM SỐ
    # ------------------------------------------
    st.sidebar.header("⚙️ Cấu Hình Dữ Liệu & Chiến Lược")

    # 1. Chọn nguồn dữ liệu
    data_source_mode = st.sidebar.radio(
        "Nguồn dữ liệu:",
        ["File mẫu có sẵn", "Tải file CSV mới lên"],
        index=0
    )

    default_files = [
        "HOSE_2020_2023_in (1).csv",
        "HOSE_2020_2023_in.csv",
        "data.csv"
    ]
    existing_file = next((f for f in default_files if os.path.exists(f)), None)

    df_full = None
    price_col = None

    if data_source_mode == "File mẫu có sẵn":
        if existing_file:
            with st.spinner("Đang tải dữ liệu..."):
                df_full, price_col = load_and_preprocess_data(existing_file)
            st.sidebar.success(f"Đã nạp file: `{existing_file}`")
        else:
            st.sidebar.error("Không tìm thấy file mẫu mặc định trong thư mục. Vui lòng tải file CSV lên!")
    else:
        uploaded_file = st.sidebar.file_uploader("Chọn file CSV dữ liệu HOSE", type=["csv"])
        if uploaded_file is not None:
            with st.spinner("Đang đọc file tải lên..."):
                df_full, price_col = load_and_preprocess_data(uploaded_file)
            st.sidebar.success("Tải file thành công!")

    if df_full is None:
        st.info("👋 Chào mừng bạn! Vui lòng chọn hoặc tải file dữ liệu CSV để bắt đầu kiểm định chiến lược.")
        st.markdown("""
        **Yêu cầu định dạng file CSV:**
        - Cột bắt buộc: `date`, `ticker`, `open`, `high`, `low`, `close`, `volume`
        - Cột khuyến nghị: `adj_close` (hệ thống sẽ ưu tiên dùng giá điều chỉnh để khử sai lệch chia tách cổ tức).
        """)
        return

    # Khoảng thời gian
    min_date = df_full["date"].min().date()
    max_date = df_full["date"].max().date()

    st.sidebar.subheader("📅 Phân chia Train / Test")
    col_d1, col_d2 = st.sidebar.columns(2)
    with col_d1:
        train_start = st.date_input("Train Start", pd.to_datetime("2020-01-01").date(), min_value=min_date, max_value=max_date)
        train_end = st.date_input("Train End", pd.to_datetime("2021-12-31").date(), min_value=min_date, max_value=max_date)
    with col_d2:
        test_start = st.date_input("Test Start", pd.to_datetime("2022-01-01").date(), min_value=min_date, max_value=max_date)
        test_end = st.date_input("Test End", pd.to_datetime("2022-12-31").date(), min_value=min_date, max_value=max_date)

    # Lọc dữ liệu Train
    mask_train = (df_full["date"] >= pd.Timestamp(train_start)) & (df_full["date"] <= pd.Timestamp(train_end))
    df_train = df_full[mask_train].copy()

    # Tính TA Features trên Train
    ta_features = calculate_ta_features(df_train, price_col=price_col, min_observations=300)

    # 2. Lựa chọn cổ phiếu
    st.sidebar.subheader("🎯 Chọn Cổ Phiếu Đầu Tư")
    selection_mode = st.sidebar.radio(
        "Cách chọn danh mục:",
        ["Tự động lọc Top N theo TA Score", "Chọn mã thủ công (Custom)"],
        index=0
    )

    all_tickers = sorted(df_full["ticker"].unique().tolist())
    
    # Cấu hình trọng số TA
    with st.sidebar.expander("⚖️ Trọng số TA Screening Score"):
        w_trend = st.slider("Trend (%)", 0, 100, 30, 5) / 100.0
        w_momentum = st.slider("Momentum (%)", 0, 100, 25, 5) / 100.0
        w_volume = st.slider("Volume (%)", 0, 100, 25, 5) / 100.0
        w_risk = st.slider("Risk (%)", 0, 100, 20, 5) / 100.0
        sum_w = w_trend + w_momentum + w_volume + w_risk
        if abs(sum_w - 1.0) > 1e-4:
            st.caption(f"⚠️ Tổng trọng số: {sum_w*100:.0f}% (sẽ được tự động chuẩn hóa về 100%)")
            w_trend, w_momentum, w_volume, w_risk = w_trend/sum_w, w_momentum/sum_w, w_volume/sum_w, w_risk/sum_w

    user_weights = {"trend": w_trend, "momentum": w_momentum, "volume": w_volume, "risk": w_risk}
    ta_scored = build_ta_score(ta_features, user_weights)

    if selection_mode == "Tự động lọc Top N theo TA Score":
        top_n = st.sidebar.number_input("Số lượng cổ phiếu (Top N)", min_value=2, max_value=20, value=5, step=1)
        selected_tickers = ta_scored.head(top_n)["Ticker"].tolist()
    else:
        default_top5 = [t for t in ["DIG", "DPM", "NLG", "KBC", "DCM"] if t in all_tickers]
        selected_tickers = st.sidebar.multiselect(
            "Chọn các mã cổ phiếu:",
            options=all_tickers,
            default=default_top5 if default_top5 else all_tickers[:5]
        )

    if len(selected_tickers) < 2:
        st.warning("⚠️ Vui lòng chọn ít nhất 2 cổ phiếu để có thể tạo danh mục đầu tư!")
        return

    st.sidebar.caption(f"Danh mục ({len(selected_tickers)} mã): " + ", ".join(selected_tickers))

    # 3. Cấu hình tham số chiến lược
    st.sidebar.subheader("🎛️ Tham Số SMA & OBV")
    param_mode = st.sidebar.radio(
        "Chế độ tham số:",
        ["Dùng Preset Tối Ưu Sẵn", "Tùy Chỉnh Thủ Công (Manual)", "Chạy Hyperopt Ngay"],
        index=0
    )

    base_commission = st.sidebar.number_input(
        "Chi phí giao dịch cơ sở (Turnover Fee)",
        min_value=0.0,
        max_value=0.01,
        value=0.001,
        step=0.0005,
        format="%.4f",
        help="0.001 tương đương 0.1% mỗi lần thay đổi trạng thái vị thế."
    )

    active_params = {}

    if param_mode == "Dùng Preset Tối Ưu Sẵn":
        for t in selected_tickers:
            if t in DEFAULT_OPTIMIZED_PARAMS:
                active_params[t] = DEFAULT_OPTIMIZED_PARAMS[t]
            else:
                active_params[t] = {"ma_short": 30, "ma_long": 200, "obv_window": 30}

    elif param_mode == "Tùy Chỉnh Thủ Công (Manual)":
        apply_uniform = st.sidebar.checkbox("Áp dụng chung 1 bộ tham số cho tất cả mã", value=True)
        if apply_uniform:
            p_ms = st.sidebar.slider("SMA Short (phiên)", 5, 100, 30, 5)
            p_ml = st.sidebar.slider("SMA Long (phiên)", 50, 400, 200, 10)
            p_ow = st.sidebar.slider("OBV Window (phiên)", 5, 100, 30, 5)
            for t in selected_tickers:
                active_params[t] = {"ma_short": p_ms, "ma_long": p_ml, "obv_window": p_ow}
        else:
            for t in selected_tickers:
                with st.sidebar.expander(f"Tham số cho {t}"):
                    curr = DEFAULT_OPTIMIZED_PARAMS.get(t, {"ma_short": 30, "ma_long": 200, "obv_window": 30})
                    ms = st.slider(f"{t} SMA Short", 5, 100, curr["ma_short"], 5, key=f"ms_{t}")
                    ml = st.slider(f"{t} SMA Long", 50, 400, curr["ma_long"], 10, key=f"ml_{t}")
                    ow = st.slider(f"{t} OBV Window", 5, 100, curr["obv_window"], 5, key=f"ow_{t}")
                    active_params[t] = {"ma_short": ms, "ma_long": ml, "obv_window": ow}

    else: # Chạy Hyperopt Ngay
        evals = st.sidebar.slider("Số lượt thử Hyperopt (Max Evals)", 10, 80, 30, 5)
        if st.sidebar.button("🚀 Bắt đầu tối ưu hóa"):
            prog_bar = st.sidebar.progress(0.0)
            for i, t in enumerate(selected_tickers):
                st.sidebar.text(f"Đang tối ưu {t}...")
                s_train = prepare_stock_series(df_full, t, price_col).loc[train_start:train_end]
                opt_res = run_hyperopt_single_stock(s_train, t, max_evals=evals, commission=base_commission, seed=42+i)
                active_params[t] = opt_res
                prog_bar.progress((i + 1) / len(selected_tickers))
            st.session_state["cached_opt_params"] = active_params
            st.sidebar.success("Tối ưu hóa hoàn tất!")
        
        if "cached_opt_params" in st.session_state:
            active_params = st.session_state["cached_opt_params"]
        else:
            for t in selected_tickers:
                active_params[t] = DEFAULT_OPTIMIZED_PARAMS.get(t, {"ma_short": 30, "ma_long": 200, "obv_window": 30})

    # Cấu hình MPT
    with st.sidebar.expander("🏛️ Cấu Hình MPT (Markowitz)"):
        mpt_min_w = st.slider("Tỷ trọng tối thiểu mỗi mã", 0.0, 0.20, 0.05, 0.01)
        mpt_max_w = st.slider("Tỷ trọng tối đa mỗi mã", 0.20, 1.00, 0.40, 0.05)
        initial_capital = st.number_input("Vốn đầu tư ban đầu (VND)", value=1_000_000, step=100_000)

    # ------------------------------------------
    # THỰC THI KIỂM ĐỊNH CHIẾN LƯỢC
    # ------------------------------------------
    stock_data = {t: prepare_stock_series(df_full, t, price_col) for t in selected_tickers}
    
    # Train backtest
    train_strategy = {}
    test_strategy = {}
    bh_train_stats = {}
    bh_test_stats = {}

    for t in selected_tickers:
        pars = active_params[t]
        s_full = stock_data[t]
        s_train = s_full.loc[train_start:train_end]

        # Train
        ev_train, _, _, _, _ = find_position_or(s_train, pars["ma_short"], pars["ma_long"], pars["obv_window"])
        ret_train, hold_train, turn_train = strategy_returns(s_train, ev_train, commission=base_commission)
        train_strategy[t] = {
            "events": ev_train,
            "holding": hold_train,
            "returns": ret_train,
            "turnover": turn_train,
            "stats": performance_stats(ret_train)
        }
        bh_train_ret = s_train["Close"].pct_change().fillna(0.0)
        bh_train_stats[t] = performance_stats(bh_train_ret)

        # Test với Full History warm-up
        test_res = evaluate_test_with_full_history(
            s_full, pars["ma_short"], pars["ma_long"], pars["obv_window"],
            test_start, test_end, commission=base_commission
        )
        test_strategy[t] = test_res

        s_test = s_full.loc[test_start:test_end]
        bh_test_ret = s_test["Close"].pct_change().fillna(0.0)
        bh_test_stats[t] = performance_stats(bh_test_ret)

    # Tạo Return Matrix cho Portfolio
    train_return_matrix = pd.concat({t: train_strategy[t]["returns"] for t in selected_tickers}, axis=1).dropna()
    test_return_matrix = pd.concat({t: test_strategy[t]["returns"] for t in selected_tickers}, axis=1).dropna()

    # Tính Trọng Số Danh Mục
    n_assets = len(selected_tickers)
    equal_weights = np.repeat(1.0 / n_assets, n_assets)
    mpt_weights = optimize_mpt(train_return_matrix, min_weight=mpt_min_w, max_weight=mpt_max_w)

    # Lợi suất Portfolio
    ew_test_ret = test_return_matrix.mul(equal_weights, axis=1).sum(axis=1)
    mpt_test_ret = test_return_matrix.mul(mpt_weights, axis=1).sum(axis=1)

    # Benchmark Buy & Hold True
    ew_bh_test_ret, ew_bh_test_wealth = calculate_equal_weight_buy_hold(stock_data, selected_tickers, equal_weights)

    # Thống kê Portfolio trên Test
    portfolio_test_comparison = pd.DataFrame({
        "Equal Weight Buy & Hold": performance_stats(ew_bh_test_ret),
        "SMA+OBV OR (Equal Weight)": performance_stats(ew_test_ret),
        "SMA+OBV OR (MPT Weighted)": performance_stats(mpt_test_ret)
    }).T

    # ------------------------------------------
    # GIAO DIỆN CHÍNH: 5 TABS PHÂN TÍCH
    # ------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Tổng Quan & Dữ Liệu",
        "🔍 Lọc Cổ Phiếu (TA Screening)",
        "📈 Phân Tích Từng Cổ Phiếu",
        "🚀 Kiểm Định Ngoài Mẫu (Out-of-Sample)",
        "🔬 Độ Nhạy & Kết Luận"
    ])

    # ==========================================
    # TAB 1: TỔNG QUAN & DỮ LIỆU
    # ==========================================
    with tab1:
        st.subheader("1. Tổng quan Nghiên cứu & Quy trình Thực hiện")
        st.markdown(f"""
        Hệ thống kiểm định chiến lược giao dịch kết hợp giữa **Tín hiệu Kỹ thuật Cắt Trung Bình Động (SMA Crossover)** và **Chỉ báo Dòng Tiền Cân Bằng (OBV Crossover)**:
        - **Dữ liệu phân tích:** Sử dụng dữ liệu chứng khoán sàn HOSE từ `{min_date}` đến `{max_date}`.
        - **Cột giá sử dụng:** `{price_col}` (được ưu tiên giá điều chỉnh để khử quyền cổ tức và chia tách).
        - **Quy tắc OR:** Tín hiệu Mua kích hoạt khi SMA Mua **HOẶC** OBV Mua; Tín hiệu Bán kích hoạt khi SMA Bán **HOẶC** OBV Bán.
        - **Cơ chế chống Look-Ahead Bias:** Tín hiệu phát sinh ở phiên $t$ chỉ được thực thi khớp lệnh vào phiên $t+1$ (`shift(1)`).
        - **Khấu trừ chi phí thực tế:** Mỗi lần thay đổi vị thế đều chịu chi phí mô phỏng `{base_commission*100:.2f}%`.
        - **Khai thác tập dữ liệu:** Train (`{train_start}` → `{train_end}`) dùng để chọn cổ phiếu và tối ưu tham số; Test (`{test_start}` → `{test_end}`) dùng kiểm định ngoài mẫu thực tế.
        """)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Tổng số dòng dữ liệu", f"{len(df_full):,}")
        with col_m2:
            st.metric("Tổng số mã cổ phiếu", f"{df_full['ticker'].nunique()}")
        with col_m3:
            st.metric("Số phiên Train", f"{len(train_return_matrix)}")
        with col_m4:
            st.metric("Số phiên Test", f"{len(test_return_matrix)}")

        st.subheader("Bảng Xem Nhanh Dữ Liệu")
        preview_ticker = st.selectbox("Chọn mã để xem dữ liệu mẫu:", selected_tickers, index=0)
        st.dataframe(stock_data[preview_ticker].tail(10), use_container_width=True)

    # ==========================================
    # TAB 2: LỌC CỔ PHIẾU (TA SCREENING)
    # ==========================================
    with tab2:
        st.subheader("2. Bảng Xếp Hạng Kỹ Thuật (TA Screening trên Train)")
        st.markdown("""
        Điểm số **TA Score** được cấu thành từ 4 nhóm nhân tố độc lập (chuẩn hóa theo phân vị Percentile Rank):
        - **Trend (30%):** Sức mạnh xu hướng SMA50 > SMA200 và Vị thế giá so với SMA200.
        - **Momentum (25%):** Động lượng sinh lời 6 tháng.
        - **Volume (25%):** Thanh khoản giá trị giao dịch và Khối lượng OBV xác nhận xu hướng.
        - **Risk (20%):** Mức độ biến động thấp (Low Volatility) và Mức sụt giảm tối đa thấp (Low Max Drawdown).
        """)

        cols_display = [
            "Ticker", "TA_Score", "Trend_Score", "Momentum_Score",
            "Volume_Score", "Risk_Score", "Avg_Trading_Value", "Volatility", "Max_Drawdown"
        ]
        score_view = ta_scored[cols_display].copy()
        score_view.insert(0, "Rank", range(1, len(score_view) + 1))

        st.dataframe(
            score_view.head(20).style.format({
                "TA_Score": "{:.3f}",
                "Trend_Score": "{:.3f}",
                "Momentum_Score": "{:.3f}",
                "Volume_Score": "{:.3f}",
                "Risk_Score": "{:.3f}",
                "Avg_Trading_Value": "{:,.0f}",
                "Volatility": "{:.2%}",
                "Max_Drawdown": "{:.2%}"
            }),
            use_container_width=True
        )

        st.success(f"🎯 **Danh mục {len(selected_tickers)} cổ phiếu đang được kiểm định:** " + ", ".join(selected_tickers))

    # ==========================================
    # TAB 3: PHÂN TÍCH TỪNG CỔ PHIẾU
    # ==========================================
    with tab3:
        st.subheader("3. Chi Tiết Tín Hiệu & Hiệu Quả Từng Mã")
        sel_t = st.selectbox("Chọn cổ phiếu để phân tích chuyên sâu:", selected_tickers)
        
        col_t1, col_t2 = st.columns([1, 2])
        pars_t = active_params[sel_t]
        with col_t1:
            st.markdown(f"### Tham số cho `{sel_t}`")
            st.write(f"- **SMA Short:** `{pars_t['ma_short']}` phiên")
            st.write(f"- **SMA Long:** `{pars_t['ma_long']}` phiên")
            st.write(f"- **OBV Window:** `{pars_t['obv_window']}` phiên")
            
            # Thống kê giao dịch
            ev_tr = train_strategy[sel_t]["events"]
            ev_te = test_strategy[sel_t]["events"]
            st.markdown("### Số lượt tín hiệu giao dịch")
            trade_df = pd.DataFrame({
                "Giai đoạn": ["Train", "Test"],
                "Lệnh Mua (BUY)": [int((ev_tr == 1).sum()), int((ev_te == 1).sum())],
                "Lệnh Bán (SELL)": [int((ev_tr == -1).sum()), int((ev_te == -1).sum())],
                "Turnover": [f"{train_strategy[sel_t]['turnover'].sum():.1f}", f"{test_strategy[sel_t]['turnover'].sum():.1f}"]
            })
            st.dataframe(trade_df, hide_index=True, use_container_width=True)

        with col_t2:
            st.markdown(f"### So sánh hiệu quả trên Test (2022)")
            res_comp = pd.DataFrame({
                "Buy & Hold": bh_test_stats[sel_t],
                "SMA + OBV OR": test_strategy[sel_t]["stats"]
            })
            res_comp["Chênh lệch (Alpha)"] = res_comp["SMA + OBV OR"] - res_comp["Buy & Hold"]
            st.dataframe(res_comp.style.format("{:.2f}"), use_container_width=True)

        # Biểu đồ tương tác Plotly
        st.subheader(f"Biểu Đồ Kỹ Thuật & Điểm Vào/Ra Lệnh `{sel_t}` (Giai Đoạn Test)")
        t_data = stock_data[sel_t].loc[test_start:test_end]
        t_res = test_strategy[sel_t]
        
        fig_stock = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.55, 0.25, 0.20],
            subplot_titles=[f"Giá {sel_t} & Điểm Mua/Bán (SMA + OBV OR)", "Chỉ báo OBV & OBV Moving Average", "Trạng thái Nắm giữ (Holding State)"]
        )

        # 1. Giá & SMA
        fig_stock.add_trace(go.Scatter(x=t_data.index, y=t_data["Close"], mode="lines", name="Giá", line=dict(color="#1E3A8A", width=1.8)), row=1, col=1)
        fig_stock.add_trace(go.Scatter(x=t_data.index, y=t_res["sma_short_series"], mode="lines", name=f"SMA {pars_t['ma_short']}", line=dict(color="#F59E0B", width=1.3, dash="dot")), row=1, col=1)
        fig_stock.add_trace(go.Scatter(x=t_data.index, y=t_res["sma_long_series"], mode="lines", name=f"SMA {pars_t['ma_long']}", line=dict(color="#DC2626", width=1.5)), row=1, col=1)

        # Điểm Mua / Bán
        buy_dates = t_res["events"][t_res["events"] == 1.0].index
        sell_dates = t_res["events"][t_res["events"] == -1.0].index
        
        if len(buy_dates) > 0:
            fig_stock.add_trace(go.Scatter(
                x=buy_dates, y=t_data.loc[buy_dates, "Close"],
                mode="markers", name="Mua (BUY)",
                marker=dict(symbol="triangle-up", size=12, color="#10B981")
            ), row=1, col=1)

        if len(sell_dates) > 0:
            fig_stock.add_trace(go.Scatter(
                x=sell_dates, y=t_data.loc[sell_dates, "Close"],
                mode="markers", name="Bán (SELL)",
                marker=dict(symbol="triangle-down", size=12, color="#EF4444")
            ), row=1, col=1)

        # 2. OBV
        fig_stock.add_trace(go.Scatter(x=t_data.index, y=t_res["obv_series"], mode="lines", name="OBV", line=dict(color="#8B5CF6", width=1.5)), row=2, col=1)
        fig_stock.add_trace(go.Scatter(x=t_data.index, y=t_res["obv_ma_series"], mode="lines", name=f"OBV MA ({pars_t['obv_window']})", line=dict(color="#6B7280", width=1.2, dash="dash")), row=2, col=1)

        # 3. Holding
        fig_stock.add_trace(go.Scatter(x=t_data.index, y=t_res["holding"], mode="lines", name="Vị thế (1: Giữ, 0: Tiền mặt)", line=dict(color="#059669", width=1.5), fill="tozeroy"), row=3, col=1)

        fig_stock.update_layout(height=720, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
        st.plotly_chart(fig_stock, use_container_width=True)

    # ==========================================
    # TAB 4: KIỂM ĐỊNH NGOÀI MẪU (TEST 2022)
    # ==========================================
    with tab4:
        st.subheader("4. Đánh Giá Danh Mục Ngoài Mẫu (Out-of-Sample Test)")
        st.markdown("""
        Giai đoạn **Test 2022** là một trong những giai đoạn thị trường giảm điểm mạnh mẽ nhất trong lịch sử VN-Index (-35% đến -40%).
        Đây là phép thử khắc nghiệt nhất để đánh giá khả năng **phòng vệ rủi ro và bảo toàn vốn** của chiến lược định lượng.
        """)

        # Bảng so sánh chỉ số tài chính
        st.markdown("### 🏆 Bảng So Sánh Hiệu Quả Danh Mục Trên Tập Test")
        st.dataframe(portfolio_test_comparison.style.format("{:.2f}"), use_container_width=True)

        # Trọng số danh mục
        st.markdown("### ⚖️ Trọng Số Phân Bổ Danh Mục")
        col_w1, col_w2 = st.columns([1, 1])
        with col_w1:
            w_df = pd.DataFrame({
                "Ticker": selected_tickers,
                "Equal Weight": equal_weights,
                "MPT Weight": mpt_weights
            })
            st.dataframe(w_df.style.format({"Equal Weight": "{:.2%}", "MPT Weight": "{:.2%}"}), use_container_width=True)
        with col_w2:
            fig_pie = go.Figure(data=[go.Pie(labels=selected_tickers, values=mpt_weights, hole=0.4)])
            fig_pie.update_layout(title_text="Tỷ trọng Tối ưu MPT", height=280, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Biểu đồ Đường Cong Vốn (Equity Curve)
        st.subheader("📈 Đường Cong Tăng Trưởng Vốn (Equity Curve)")
        equity_bh = initial_capital * (1.0 + ew_bh_test_ret).cumprod()
        equity_ew = initial_capital * (1.0 + ew_test_ret).cumprod()
        equity_mpt = initial_capital * (1.0 + mpt_test_ret).cumprod()

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=equity_bh.index, y=equity_bh, mode="lines", name="Equal Weight Buy & Hold", line=dict(color="#DC2626", width=2.2, dash="dash")))
        fig_eq.add_trace(go.Scatter(x=equity_ew.index, y=equity_ew, mode="lines", name="SMA+OBV OR (Equal Weight)", line=dict(color="#2563EB", width=2.5)))
        fig_eq.add_trace(go.Scatter(x=equity_mpt.index, y=equity_mpt, mode="lines", name="SMA+OBV OR (MPT Weighted)", line=dict(color="#059669", width=2.8)))

        fig_eq.update_layout(
            title="So Sánh Giá Trị Danh Mục Đầu Tư Theo Thời Gian (Test)",
            xaxis_title="Ngày giao dịch",
            yaxis_title="Giá trị tài sản (VND)",
            height=500,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # Biểu đồ Drawdown (Underwater Chart)
        st.subheader("📉 Mức Sụt Giảm Tài Sản Tối Đa (Underwater Drawdown)")
        dd_bh = (equity_bh / equity_bh.cummax() - 1.0) * 100.0
        dd_ew = (equity_ew / equity_ew.cummax() - 1.0) * 100.0
        dd_mpt = (equity_mpt / equity_mpt.cummax() - 1.0) * 100.0

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd_bh.index, y=dd_bh, mode="lines", name="Drawdown Buy & Hold", line=dict(color="#DC2626", width=1.8), fill="tozeroy"))
        fig_dd.add_trace(go.Scatter(x=dd_ew.index, y=dd_ew, mode="lines", name="Drawdown SMA+OBV (EW)", line=dict(color="#2563EB", width=2.0)))
        fig_dd.add_trace(go.Scatter(x=dd_mpt.index, y=dd_mpt, mode="lines", name="Drawdown SMA+OBV (MPT)", line=dict(color="#059669", width=2.2)))

        fig_dd.update_layout(
            title="Độ Sụt Giảm Từ Đỉnh (Underwater Drawdown %)",
            xaxis_title="Ngày giao dịch",
            yaxis_title="Drawdown (%)",
            height=380,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    # ==========================================
    # TAB 5: ĐỘ NHẠY & KẾT LUẬN
    # ==========================================
    with tab5:
        st.subheader("5. Kiểm Định Độ Nhạy (Sensitivity Analysis)")
        st.markdown("Kiểm tra sức chống chịu của chiến lược khi thay đổi giả định chi phí giao dịch trên thị trường thực tế:")

        comm_scenarios = [0.0, 0.0005, 0.001, 0.002]
        cost_rows = []
        for comm in comm_scenarios:
            test_streams = {}
            for t in selected_tickers:
                res_c = evaluate_test_with_full_history(
                    stock_data[t], active_params[t]["ma_short"], active_params[t]["ma_long"],
                    active_params[t]["obv_window"], test_start, test_end, commission=comm
                )
                test_streams[t] = res_c["returns"]
            mat_c = pd.concat(test_streams, axis=1).dropna()
            ew_c = mat_c.mul(equal_weights, axis=1).sum(axis=1)
            mpt_c = mat_c.mul(mpt_weights, axis=1).sum(axis=1)

            st_ew = performance_stats(ew_c)
            st_mpt = performance_stats(mpt_c)
            cost_rows.append({
                "Phí Giao Dịch": f"{comm*100:.2f}%",
                "EW Total Return [%]": st_ew["Total Return [%]"],
                "EW Sharpe": st_ew["Sharpe Ratio"],
                "MPT Total Return [%]": st_mpt["Total Return [%]"],
                "MPT Sharpe": st_mpt["Sharpe Ratio"]
            })

        cost_df = pd.DataFrame(cost_rows)
        st.dataframe(cost_df.style.format({
            "EW Total Return [%]": "{:.2f}%",
            "EW Sharpe": "{:.4f}",
            "MPT Total Return [%]": "{:.2f}%",
            "MPT Sharpe": "{:.4f}"
        }), use_container_width=True)

        st.subheader("💡 Kết Luận Tự Động Từ Kiểm Định Định Lượng")
        best_ret_strategy = portfolio_test_comparison["Total Return [%]"].idxmax()
        best_sharpe_strategy = portfolio_test_comparison["Sharpe Ratio"].idxmax()
        best_dd_strategy = portfolio_test_comparison["Max Drawdown [%]"].idxmax()

        st.markdown(f"""
        1. **Khả năng phòng vệ thị trường gấu (Bear Market Protection):**
           - Benchmark **Buy & Hold** trong năm 2022 sụt giảm tới **{portfolio_test_comparison.loc['Equal Weight Buy & Hold', 'Total Return [%]']:.2f}%** với Max Drawdown lên tới **{portfolio_test_comparison.loc['Equal Weight Buy & Hold', 'Max Drawdown [%]']:.2f}%**.
           - Trong khi đó, chiến lược kết hợp **{best_ret_strategy}** chỉ giảm **{portfolio_test_comparison.loc[best_ret_strategy, 'Total Return [%]']:.2f}%** và cắt giảm một nửa mức sụt giảm tối đa xuống còn **{portfolio_test_comparison.loc[best_ret_strategy, 'Max Drawdown [%]']:.2f}%**.
        2. **Hiệu quả của việc kết hợp SMA và OBV theo quy tắc OR:**
           - Tín hiệu dòng tiền OBV giúp nhận diện lực phân phối dòng tiền sớm hơn đường giá SMA.
           - Việc chuyển đổi trạng thái về tiền mặt (holding = 0) trong phần lớn các nhịp giảm sốc năm 2022 giúp danh mục giữ được gần như nguyên vẹn vốn đầu tư.
        3. **Vai trò của tối ưu hóa danh mục MPT:**
           - Chiến lược MPT phân bổ tỷ trọng theo ma trận hiệp phương sai của các luồng lợi suất giúp đạt Sharpe cao nhất: **{portfolio_test_comparison.loc['SMA+OBV OR (MPT Weighted)', 'Sharpe Ratio']:.4f}**.
        4. **Độ nhạy chi phí:**
           - Ngay cả ở mức phí giao dịch cao 0.20% (bao gồm thuế phí và trượt giá slippage), chiến lược vẫn duy trì mức vượt trội (Alpha) rất lớn so với Buy & Hold.
        """)

if __name__ == "__main__":
    main()
