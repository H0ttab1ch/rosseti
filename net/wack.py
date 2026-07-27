import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import norm

try:
    import holidays
    RU_HOLIDAYS = holidays.RU()
except ImportError:
    RU_HOLIDAYS = None

# =========================================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ И ФИРМЕННЫЙ СТИЛЬ «РОССЕТИ»
# =========================================================
st.set_page_config(
    page_title="Аналитический дашборд энергопотребления",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROSSETI_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@300;400;700&display=swap');
    :root {
        --background-color: #FFFFFF !important;
        --secondary-background-color: #F8F9FA !important;
        --text-color: #1A1A1A !important;
    }
    html, body, [class*="css"], .stApp {
        font-family: 'PF Din Text Cond Pro', 'Roboto Condensed', 'Arial Narrow', sans-serif !important;
        background-color: #F8F9FA !important;
        color: #1A1A1A !important;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, div { color: #1A1A1A; }

    /* --- КНОПКИ И НАВИГАЦИЯ --- */
    .stButton > button, 
    .stButton > button *, 
    .stButton > button p, 
    .stButton > button span,
    .nav-btn button, 
    .nav-btn button *, 
    .nav-btn button p, 
    .nav-btn button span {
        color: #FFFFFF !important; 
    }
    .stButton > button { 
        background-color: #005A9B !important; 
        border: none !important; 
        border-radius: 4px !important; 
        font-weight: 600 !important; 
        text-transform: uppercase !important; 
        padding: 8px 16px !important; 
        transition: all 0.2s ease-in-out; 
    }
    .stButton > button:hover, 
    .stButton > button:hover *, 
    .stButton > button:hover p, 
    .stButton > button:hover span,
    .nav-btn button:hover, 
    .nav-btn button:hover *, 
    .nav-btn button:hover p, 
    .nav-btn button:hover span { 
        background-color: #003B66 !important; 
        color: #FFFFFF !important; 
        box-shadow: 0 4px 8px rgba(0, 90, 155, 0.3) !important; 
    }

    .rosseti-header {
        background: linear-gradient(90deg, #005A9B 0%, #003B66 100%);
        color: #FFFFFF !important;
        padding: 20px 25px;
        border-radius: 4px;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0, 90, 155, 0.15);
    }
    .rosseti-header * { color: #FFFFFF !important; }
    .rosseti-title { font-size: 24px; font-weight: 700; letter-spacing: 0.5px; margin: 0; text-transform: uppercase; }
    .rosseti-subtitle { font-size: 14px; opacity: 0.9; margin-top: 4px; }
    section[data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #E0E0E0; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label { color: #005A9B !important; font-weight: 700; }
    div[data-testid="stMetric"] { background-color: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-left: 5px solid #005A9B !important; padding: 15px 20px !important; border-radius: 4px !important; box-shadow: 0 2px 6px rgba(0,0,0,0.03) !important; }
    div[data-testid="stMetricLabel"] { font-size: 13px !important; color: #555555 !important; text-transform: uppercase !important; font-weight: 600 !important; }
    div[data-testid="stMetricValue"] { font-size: 22px !important; color: #005A9B !important; font-weight: 700 !important; }
    .stDataFrame { background-color: #FFFFFF !important; border-radius: 4px; border: 1px solid #E2E8F0; }
    .stDataFrame [data-testid="stTable"] { background-color: #FFFFFF !important; color: #1A1A1A !important; }
    div[data-baseweb="select"] > div { background-color: #FFFFFF !important; color: #1A1A1A !important; border-color: #CCCCCC !important; }
    header[data-testid="stHeader"] { background: transparent !important; }
</style>
"""
st.markdown(ROSSETI_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# 1. ЗАГРУЗКА ДАННЫХ ИЗ ТРЁХ ТАБЛИЦ ИСТОЧНИКОВ
# ---------------------------------------------------------
@st.cache_data(ttl=600)
def load_and_preprocess_energy_data():
    db_url = os.getenv("DB_URL")
    if not db_url:
        st.error("Строка подключения к БД отсутствует. Настройте DB_URL.")
        st.stop()

    engine = create_engine(db_url)
    
    # Загрузка данных power_ext (3 целевых объекта АЗС)
    query_ext = "SELECT * FROM power_ext ORDER BY station_id, date ASC;"
    df_ext = pd.read_sql(query_ext, engine)
    
    excluded_consumers = tuple(df_ext['station_id'].unique().tolist()) if not df_ext.empty else ()
    
    # Формируем WHERE, чтобы исключить объекты из power_ext
    if excluded_consumers:
        excl_str = f"('{excluded_consumers[0]}')" if len(excluded_consumers) == 1 else str(excluded_consumers)
        where_clause = f"WHERE source_sheet NOT IN {excl_str}"
    else:
        where_clause = ""

    # Загрузка основной таблицы power
    query_power = f"""
    SELECT 
        source_sheet AS series_name,
        date AS ts,
        SUM(power) AS value_kwh
    FROM power 
    {where_clause}
    GROUP BY source_sheet, date
    ORDER BY source_sheet, date ASC;
    """
    df_power = pd.read_sql(query_power, engine)

    # Базовая подготовка power
    if not df_power.empty:
        df_power['ts'] = pd.to_datetime(df_power['ts'])
        df_power['series_name'] = df_power['series_name'].astype(str)
        df_power = (
            df_power.set_index('ts')
            .groupby('series_name')
            .resample('D')
            .agg({'value_kwh': 'sum'})
            .reset_index()
        ).sort_values(by=['series_name', 'ts']).reset_index(drop=True)

    # Базовая подготовка power_ext
    if not df_ext.empty:
        df_ext['date'] = pd.to_datetime(df_ext['date'])

    return df_power, df_ext


@st.cache_data(ttl=600)
def load_power_setpoints():
    """Загрузка таблицы уставок мощности и параметров объектов."""
    db_url = st.secrets.get("DB_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        return pd.DataFrame()

    try:
        engine = create_engine(db_url)
        query = "SELECT * FROM power_setpoints;"
        df_setpoints = pd.read_sql(query, engine)
        return df_setpoints
    except Exception as e:
        st.warning(f"Не удалось загрузить данные из таблицы power_setpoints: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------
# 2. МОДЕЛЬ 1: XGBoost ДЛЯ ОСНОВНЫХ ОБЪЕКТОВ
# ---------------------------------------------------------
def preprocess_daily_advanced(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
        
    df = df.copy()
    df['target'] = df['value_kwh']
    df['series_name_cat'] = df['series_name'].astype('category')
    df['dayofweek'] = df['ts'].dt.dayofweek
    df['month'] = df['ts'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    if RU_HOLIDAYS is not None:
        df['is_holiday'] = df['ts'].dt.date.apply(lambda x: int(x in RU_HOLIDAYS))
    else:
        df['is_holiday'] = 0

    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    grouped = df.groupby('series_name')['target']
    df['lag_1d'] = grouped.shift(1)
    df['lag_2d'] = grouped.shift(2)
    df['lag_7d'] = grouped.shift(7)
    df['lag_14d'] = grouped.shift(14)
    df['rolling_mean_7d'] = grouped.transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    df['rolling_std_7d'] = grouped.transform(lambda x: x.shift(1).rolling(7, min_periods=1).std()).fillna(0)
    df['rolling_mean_30d'] = grouped.transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    df['ratio_7d_30d'] = (df['rolling_mean_7d'] / (df['rolling_mean_30d'] + 1e-5)).fillna(1.0)
    return df

@st.cache_data(ttl=600)
def walk_forward_global_model(df: pd.DataFrame, retrain_step_days: int = 7) -> pd.DataFrame:
    if df.empty: return df

    features = [
        'series_name_cat', 'dayofweek', 'is_weekend', 'is_holiday', 'month',
        'day_sin', 'day_cos', 'month_sin', 'month_cos', 'lag_1d', 'lag_2d', 
        'lag_7d', 'lag_14d', 'rolling_mean_7d', 'rolling_std_7d', 
        'rolling_mean_30d', 'ratio_7d_30d'
    ]

    df_clean = df.dropna(subset=features + ['target']).copy().sort_values('ts').reset_index(drop=True)
    unique_dates = df_clean['ts'].drop_duplicates().sort_values().tolist()
    
    warmup_days = 30 
    df['predicted_kwh'] = np.nan
    df['real_kwh'] = df['target']

    if len(unique_dates) <= warmup_days: return df

    all_predictions = []
    model = XGBRegressor(n_estimators=150, learning_rate=0.04, max_depth=5, subsample=0.8,
                         colsample_bytree=0.8, enable_categorical=True, random_state=42, n_jobs=-1)

    for i in range(warmup_days, len(unique_dates), retrain_step_days):
        train_cutoff = unique_dates[i]
        test_dates = unique_dates[i : i + retrain_step_days]

        train_data = df_clean[df_clean['ts'] < train_cutoff]
        test_data = df_clean[df_clean['ts'].isin(test_dates)]
        if train_data.empty or test_data.empty: continue

        model.fit(train_data[features], train_data['target'])
        preds = model.predict(test_data[features])
        
        test_res = test_data[['series_name', 'ts']].copy()
        test_res['predicted_kwh'] = np.clip(preds, a_min=0, a_max=None)
        all_predictions.append(test_res)

    if not all_predictions: return df

    df_preds = pd.concat(all_predictions, ignore_index=True)
    df['series_name'] = df['series_name'].astype(str)
    df_preds['series_name'] = df_preds['series_name'].astype(str)

    df_merged = pd.merge(df, df_preds, on=['series_name', 'ts'], how='left', suffixes=('', '_pred'))
    if 'predicted_kwh_pred' in df_merged.columns:
        df_merged['predicted_kwh'] = df_merged['predicted_kwh_pred']
        df_merged.drop(columns=['predicted_kwh_pred'], inplace=True)

    df_merged['real_kwh'] = df_merged['target']
    df_merged['ts_predicted'] = df_merged['ts']
    return df_merged


# ---------------------------------------------------------
# 3. МОДЕЛЬ 2: CatBoost ДЛЯ ТРЁХ ОБЪЕКТОВ ИЗ power_ext
# ---------------------------------------------------------
@st.cache_data(ttl=600)
def process_catboost_model(df_ext: pd.DataFrame) -> pd.DataFrame:
    if df_ext.empty:
        return pd.DataFrame()
        
    df = df_ext.copy()
    
    # Сопоставление структуры
    df.rename(columns={
        'station_id': 'series_name', 
        'date': 'ts', 
        'power_consumption': 'real_kwh'
    }, inplace=True)
    
    # Циклические признаки для CatBoost
    df['month'] = df['ts'].dt.month
    df['day_of_year'] = df['ts'].dt.dayofyear
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    df['day_of_year_sin'] = np.sin(2*np.pi*df['day_of_year']/365)
    df['day_of_year_cos'] = np.cos(2*np.pi*df['day_of_year']/365)

    # Признаки из таблицы power_ext
    feature_cols = [
        'series_name', 'day_of_week', 'month',
        'is_weekend', 'is_holiday', 'is_pre_holiday',
        'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos',
        'temp_avg_c', 'temp_min_c', 'temp_max_c', 'precip_mm'
    ]
    cat_features = ['series_name', 'month']

    X = df[feature_cols].reset_index(drop=True)
    y = df['real_kwh'].reset_index(drop=True)

    # Кросс-валидация со стратификацией по объекту
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))

    for train_idx, test_idx in skf.split(X, df['series_name']):
        train_pool = Pool(X.iloc[train_idx], y.iloc[train_idx], cat_features=cat_features)
        test_pool = Pool(X.iloc[test_idx], cat_features=cat_features)

        # Для внутренней метрики CatBoost используется встроенное название 'MAE'
        model = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=4,
                                   loss_function='MAE', verbose=0, random_seed=42)
        model.fit(train_pool)
        pred = model.predict(test_pool)
        oof_preds[test_idx] = pred

    df['predicted_kwh'] = oof_preds
    df['ts_predicted'] = df['ts']
    
    return df[['series_name', 'ts', 'real_kwh', 'predicted_kwh', 'ts_predicted']]


# ---------------------------------------------------------
# 4. ДЕТЕКТОР АНОМАЛИЙ И ИНТЕРФЕЙС
# ---------------------------------------------------------
def anomaly_tester(df: pd.DataFrame, std_threshold: float = 2.75):
    df = df.copy()
    df['error'] = df['real_kwh'] - df['predicted_kwh']
    df['abs_error'] = df['error'].abs()
    df['is_anomaly'] = False

    valid_preds = df.dropna(subset=['predicted_kwh'])
    metrics = {}

    for consumer, group in valid_preds.groupby('series_name'):
        mae = group['abs_error'].mean()
        rmse = np.sqrt((group['error'] ** 2).mean())
        denom = group['real_kwh'].replace(0, np.nan)
        mape = (group['abs_error'] / denom).mean() * 100

        mean_err = group['error'].mean()
        std_err = group['error'].std()

        if pd.isna(std_err) or std_err == 0:
            metrics[consumer] = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Anomalies": 0}
            continue

        upper_bound = mean_err + (std_threshold * std_err)
        lower_bound = mean_err - (std_threshold * std_err)

        anomalies_mask = (group['error'] > upper_bound) | (group['error'] < lower_bound)
        df.loc[group[anomalies_mask].index, 'is_anomaly'] = True

        metrics[consumer] = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Anomalies": int(anomalies_mask.sum())}

    return df, metrics


def main():
    st.markdown("""
    <div class="rosseti-header">
        <div class="rosseti-title">АНАЛИТИЧЕСКИЙ ДАШБОРД</div>
        <div class="rosseti-subtitle">Система мониторинга электропотребления и выявления аномалий</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Загрузка данных и инициализация моделей (XGBoost + CatBoost)..."):
        # Извлекаем данные
        df_power, df_ext = load_and_preprocess_energy_data()
        df_setpoints = load_power_setpoints()
        
        # Запуск пайплайна XGBoost
        df_processed_xgb = preprocess_daily_advanced(df_power)
        df_final_xgb = walk_forward_global_model(df_processed_xgb, retrain_step_days=7)
        
        # Запуск пайплайна CatBoost
        df_final_cat = process_catboost_model(df_ext)
        
        # Слияние потоков предсказаний
        df_final = pd.concat([df_final_xgb, df_final_cat], ignore_index=True)

    if df_final.empty:
        st.warning("Данные не получены из базы данных.")
        st.stop()

        # --- Настройки панели управления (Sidebar) ---
    confidence_pct = st.sidebar.slider(
        "Порог норматива (%)", 
        min_value=80.0, 
        max_value=120.0, 
        value=98.8, 
        step=0.1,
        help="Доля ожидаемого штатного электропотребления. Все значения за пределами этого интервала считаются аномалиями."
    )

    # 2. Convert confidence percentage to sigma threshold (z-score)
    alpha = 1.0 - (confidence_pct / 100.0)
    std_thresh = float(norm.ppf(1.0 - alpha / 2.0))

    # Optional display showing the calculated sigma for transparency
    st.sidebar.caption(f"Эквивалент в стандартных отклонениях: **±{std_thresh:.2f}σ**")


    df_evaluated, metrics = anomaly_tester(df_final, std_threshold=std_thresh)
    all_consumers = sorted(df_final['series_name'].unique().tolist())

    # Динамическая привязка координат из power_setpoints при наличии
    default_coords = {
        "02073": {"lat": 54.7621750, "lon": 56.3952460},
        "02136": {"lat": 54.6225180, "lon": 55.9280060},
        "02134": {"lat": 54.6471700, "lon": 55.9201900},
        "арадан": {"lat": 52.5777780, "lon": 93.4430560},
        "петрунь": {"lat": 66.4722220, "lon": 60.7416670},
        "северо-курильск": {"lat": 50.6758, "lon": 156.1244},
        "эвенск": {"lat": 61.9166670, "lon": 159.2333330}
    }

    if 'consumer_coords' not in st.session_state:
        st.session_state.consumer_coords = {}
        for c in all_consumers:
            matched = False
            c_lower = c.lower()
            
            # Пробуем найти координаты в загруженной power_setpoints
            if not df_setpoints.empty:
                match_sp = df_setpoints[df_setpoints['location'].astype(str).str.strip().str.lower().apply(lambda loc: loc in c_lower or c_lower in loc)]
                if not match_sp.empty:
                    row_sp = match_sp.iloc[0]
                    st.session_state.consumer_coords[c] = {"lat": float(row_sp['latitude']), "lon": float(row_sp['longitude'])}
                    matched = True

            if not matched:
                for key, coords in default_coords.items():
                    if key in c_lower:
                        st.session_state.consumer_coords[c] = coords
                        matched = True
                        break
            if not matched:
                st.session_state.consumer_coords[c] = {"lat": 54.75, "lon": 56.0}

    pages_list = ["Интерактивная карта", "Общая сводка", "Графики объектов"]
    if 'navigation_page' not in st.session_state or st.session_state.navigation_page not in pages_list:
        st.session_state.navigation_page = "Интерактивная карта"
    if 'selected_consumer_target' not in st.session_state:
        st.session_state.selected_consumer_target = all_consumers[0]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Разделы дашборда")
    selected_page = st.sidebar.radio(
        "Навигация", options=pages_list, index=pages_list.index(st.session_state.navigation_page), label_visibility="collapsed"
    )
    if selected_page != st.session_state.navigation_page:
        st.session_state.navigation_page = selected_page
        st.rerun()

    # --- СТРАНИЦА 1: КАРТА ---
    if st.session_state.navigation_page == "Интерактивная карта":
        st.subheader("Географическое расположение объектов сети")
        st.caption("Выберите объект на карте для детального анализа профиля энергопотребления.")

        map_rows = []
        for c in all_consumers:
            coords = st.session_state.consumer_coords.get(c, {"lat": 54.75, "lon": 56.0})
            anom_count = metrics.get(c, {}).get("Anomalies", 0)
            map_rows.append({"series_name": c, "lat": coords["lat"], "lon": coords["lon"], "anomalies": anom_count, "status": "Отклонение" if anom_count > 0 else "Штатный режим"})
        
        fig_map = px.scatter_mapbox(
            pd.DataFrame(map_rows), lat="lat", lon="lon", color="status",
            color_discrete_map={"Штатный режим": "#005A9B", "Отклонение": "#D62728"},
            size_max=16, zoom=3, center={"lat": 60.0, "lon": 90.0}, hover_name="series_name",
            hover_data={"lat": False, "lon": False, "anomalies": True, "status": True}, custom_data=["series_name"]
        )
        fig_map.update_layout(
            mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=520, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            legend=dict(title=dict(text="Статус объекта", font=dict(color="#1A1A1A")), bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#E0E0E0", borderwidth=1, font=dict(color="#1A1A1A"), yanchor="top", y=0.98, xanchor="left", x=0.01)
        )
        fig_map.update_traces(marker=dict(size=22))
        map_event = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", selection_mode="points", key="plotly_map_selection")

        if map_event and "selection" in map_event and "points" in map_event["selection"]:
            selected_points = map_event["selection"]["points"]
            if selected_points and selected_points[0].get("customdata"):
                st.session_state.selected_consumer_target = selected_points[0].get("customdata")[0]
                st.session_state.navigation_page = "Графики объектов"
                st.rerun()

        st.markdown("---")
        col_m1, col_m2 = st.columns([3, 1], vertical_alignment="bottom")
        selected_from_map = col_m1.selectbox("Быстрый выбор объекта сети:", options=all_consumers, key="map_target_select")
        if col_m2.button("Перейти к графику", use_container_width=True):
            st.session_state.selected_consumer_target = selected_from_map
            st.session_state.navigation_page = "Графики объектов"
            st.rerun()

    # --- СТРАНИЦА 2: ОБЩАЯ СВОДКА ---
    elif st.session_state.navigation_page == "Общая сводка":
        col_head, col_back = st.columns([4, 1], vertical_alignment="center")
        col_head.subheader("Сводные показатели по энергосетевому комплексу")
        if col_back.button("К карте", use_container_width=True):
            st.session_state.navigation_page = "Интерактивная карта"
            st.rerun()

        valid_metrics = [m for m in metrics.values() if not pd.isna(m.get("MAPE"))]
        col1, col2, col3 = st.columns(3)
        col1.metric("Всего подстанций / объектов", len(all_consumers))
        col2.metric("Выявлено аномалий", sum(m["Anomalies"] for m in metrics.values()), delta_color="inverse")
        col3.metric("Средняя погрешность модели (MAPE)", f"{np.mean([m['MAPE'] for m in valid_metrics]) if valid_metrics else 0.0:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Детализация по объектам сети")
        
        summary_rows = [
            {
                "Объект сети": c, 
                "Кол-во аномалий": metrics[c]["Anomalies"], 
                "САО (MAE), кВт·ч": round(metrics[c]["MAE"], 2),
                "СКО (RMSE), кВт·ч": round(metrics[c]["RMSE"], 2), 
                "MAPE, %": round(metrics[c]["MAPE"], 1)
            }
            for c in all_consumers if c in metrics
        ]
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # --- СТРАНИЦА 3: ИНДИВИДУАЛЬНЫЕ ГРАФИКИ ---
    else:
        col_head, col_back = st.columns([4, 1], vertical_alignment="center")
        col_head.subheader("Анализ профиля нагрузок объекта")
        if col_back.button("К карте", use_container_width=True):
            st.session_state.navigation_page = "Интерактивная карта"
            st.rerun()

        target_key = st.session_state.get('selected_consumer_target', None)
        selected_consumer = st.selectbox("Выберите объект сети:", options=all_consumers, index=all_consumers.index(target_key) if target_key in all_consumers else 0)
        st.session_state.selected_consumer_target = selected_consumer

        if selected_consumer in metrics:
            m = metrics[selected_consumer]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Зафиксировано аномалий", m["Anomalies"])
            c2.metric("САО (MAE)", f"{m['MAE']:.2f} кВт·ч", help="Средняя абсолютная ошибка")
            c3.metric("СКО (RMSE)", f"{m['RMSE']:.2f} кВт·ч", help="Среднеквадратическая ошибка")
            c4.metric("MAPE", f"{m['MAPE']:.1f}%", help="Средняя абсолютная процентная ошибка")

        sub = df_evaluated[df_evaluated['series_name'] == selected_consumer].sort_values('ts')
        fig = go.Figure()

        if not sub.dropna(subset=['real_kwh']).empty:
            fig.add_trace(go.Scatter(x=sub['ts'], y=sub['real_kwh'], mode='lines', name='Фактическое потребление', line=dict(width=2.5, color='#005A9B')))
        if not sub.dropna(subset=['predicted_kwh']).empty:
            fig.add_trace(go.Scatter(x=sub['ts_predicted'], y=sub['predicted_kwh'], mode='lines', name='Расчетный прогноз', line=dict(dash='dash', width=2, color='#E65100'), opacity=0.9))
        
        anomalies = sub[sub['is_anomaly']]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(x=anomalies['ts'], y=anomalies['real_kwh'], mode='markers', name='Выявленные отклонения', marker=dict(color='#D62728', size=16, symbol='x', line=dict(width=3, color='#800000'))))

        fig.update_layout(
            title=dict(text=f"Профиль электропотребления — Объект {selected_consumer}", font=dict(size=18, color='#005A9B', family='PF Din Text Cond Pro, Roboto Condensed')),
            xaxis_title="Дата", yaxis_title="Мощность / Потребление (кВт·ч)", template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            hovermode="x unified", height=500, margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(title_font=dict(color="#1A1A1A"), tickfont=dict(color="#1A1A1A"), gridcolor="#E2E8F0"),
            yaxis=dict(title_font=dict(color="#1A1A1A"), tickfont=dict(color="#1A1A1A"), gridcolor="#E2E8F0"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#1A1A1A"), bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0", borderwidth=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------------------------
        # ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ОБ ОБЪЕКТЕ ИЗ power_setpoints
        # ---------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Паспортные характеристики и уставки объекта")

        if not df_setpoints.empty:
            # Сопоставляем selected_consumer с полем location из таблицы power_setpoints
            selected_lower = str(selected_consumer).strip().lower()
            matched_sp = df_setpoints[
                df_setpoints['location'].astype(str).str.strip().str.lower().apply(
                    lambda loc: loc in selected_lower or selected_lower in loc
                )
            ]

            if not matched_sp.empty:
                sp_row = matched_sp.iloc[0]
                
                sp_col1, sp_col2, sp_col3, sp_col4 = st.columns(4)
                
                sp_col1.metric(
                    label="Источник питания", 
                    value=str(sp_row.get('power_source', 'Н/Д'))
                )
                sp_col2.metric(
                    label="Уставка на объект", 
                    value=f"{sp_row.get('setpoint_reserve_15_kw', 0)} кВт"
                )
                sp_col3.metric(
                    label="Максимальная мощность", 
                    value=f"{sp_row.get('max_capacity_kw', 0)} кВт"
                )
                
                lat_val = sp_row.get('latitude', None)
                lon_val = sp_row.get('longitude', None)
                coords_display = f"{lat_val:.4f}, {lon_val:.4f}" if lat_val and lon_val else str(sp_row.get('coordinates', 'Н/Д'))
                
                sp_col4.metric(
                    label="Географические координаты", 
                    value=coords_display
                )
            else:
                st.info("Паспортные данные и уставки мощности для данного объекта не найдена в базе `power_setpoints`.")
        else:
            st.info("Таблица уставок `power_setpoints` недоступна или пуста.")

if __name__ == "__main__":
    main()