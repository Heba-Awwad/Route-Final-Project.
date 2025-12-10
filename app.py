# app.py  -- Air Quality Dashboard (O3 AQI, UMAP, KMeans + Hierarchical + DBSCAN)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from xgboost import XGBRegressor

# Optional imports: ARIMA & UMAP
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False

try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# -----------------------------
# 0) PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Air Quality Dashboard",
    layout="wide"
)

st.title("🌍 United States Air Quality Dashboard (2000–2016)")
st.markdown(
    "This dashboard uses a **60,000 record sample** from the US EPA air "
    "pollution dataset. It supports visualization, forecasting, "
    "pattern exploration (KMeans + Hierarchical + DBSCAN + UMAP), "
    "health advisory, and data export."
)

# Small helper for seasons (same idea as notebook)
def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"


# -----------------------------
# 1) LOAD & PREPARE DATA
# -----------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("pollution_us_2000_2016.csv")

    # Keep only relevant columns
    df = df[[
        "Date Local", "State", "City",
        "NO2 Mean", "O3 Mean", "SO2 Mean", "CO Mean",
        "NO2 AQI", "O3 AQI", "SO2 AQI", "CO AQI"
    ]]

    # Drop rows where all pollutant means are NaN
    df = df.dropna(subset=["NO2 Mean", "O3 Mean", "SO2 Mean", "CO Mean"])

    # Sample 60,000 records for performance
    if len(df) > 60000:
        df = df.sample(n=60000, random_state=42)
    df = df.reset_index(drop=True)

    # Parse date
    df["Date Local"] = pd.to_datetime(df["Date Local"], errors="coerce")
    df = df.dropna(subset=["Date Local"])

    # Temporal features
    df["Year"] = df["Date Local"].dt.year
    df["Month"] = df["Date Local"].dt.month
    df["DayOfYear"] = df["Date Local"].dt.dayofyear
    df["DayOfWeek"] = df["Date Local"].dt.dayofweek

    return df


data = load_data()

# -----------------------------
# SIDEBAR GLOBAL FILTERS
# -----------------------------

st.sidebar.header("🔧 Global Filters")

min_date = data["Date Local"].min().date()
max_date = data["Date Local"].max().date()

date_filter = st.sidebar.slider(
    "Select date range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
)

state_options = ["All"] + sorted(data["State"].unique())
state_filter = st.sidebar.selectbox("Filter by State", state_options)

# Apply global date/state filters
mask = (
    (data["Date Local"].dt.date >= date_filter[0]) &
    (data["Date Local"].dt.date <= date_filter[1])
)
if state_filter != "All":
    mask &= (data["State"] == state_filter)

filtered = data[mask].copy()

if filtered.empty:
    st.warning("⚠️ No data found for this filter. Try expanding the date range or choose another state.")
    st.stop()


# -----------------------------
# TABS LAYOUT
# -----------------------------

tab_map, tab_forecast, tab_patterns, tab_health, tab_export = st.tabs(
    ["🗺️ Map", "📈 Forecasting", "🔍 Patterns & UMAP", "❤️ Health Advisory", "📤 Export"]
)


# -----------------------------
# TAB 1: INTERACTIVE MAP
# -----------------------------

with tab_map:
    st.subheader("🗺️ Interactive State-level Pollution Map (O3 AQI)")

    US_STATE_ABBREV = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
        'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT',
        'Delaware': 'DE', 'District of Columbia': 'DC', 'Florida': 'FL',
        'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
        'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY',
        'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA',
        'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
        'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH',
        'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
        'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
        'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
        'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
        'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
        'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
        'Wisconsin': 'WI', 'Wyoming': 'WY'
    }

    map_mode = st.radio(
        "Map aggregation mode:",
        ["Latest day in selected range", "Average over selected range"],
        horizontal=True
    )

    if map_mode == "Latest day in selected range":
        latest_date = filtered["Date Local"].max().date()
        st.markdown(f"🔍 Showing **latest available day** in filter: **{latest_date}**")
        map_source = filtered[filtered["Date Local"].dt.date == latest_date]
    else:
        st.markdown("🔍 Showing **average AQI over the selected date range**.")
        map_source = filtered.copy()

    map_df = (
        map_source
        .groupby("State", as_index=False)
        .agg(
            O3_AQI_mean=("O3 AQI", "mean"),
            NO2_AQI_mean=("NO2 AQI", "mean"),
            SO2_AQI_mean=("SO2 AQI", "mean"),
            CO_AQI_mean=("CO AQI", "mean")
        )
    )

    map_df["state_code"] = map_df["State"].map(US_STATE_ABBREV)
    map_df = map_df.dropna(subset=["state_code"])

    if not map_df.empty:
        fig_map = px.choropleth(
            map_df,
            locations="state_code",
            locationmode="USA-states",
            color="O3_AQI_mean",
            hover_name="State",
            hover_data=["O3_AQI_mean", "NO2_AQI_mean", "SO2_AQI_mean", "CO_AQI_mean"],
            color_continuous_scale="YlOrRd",
            scope="usa",
            labels={"O3_AQI_mean": "Mean O3 AQI"}
        )
        fig_map.update_layout(height=500, margin=dict(r=10, l=10, t=30, b=10))
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No state-level data available for this aggregation.")


# -----------------------------
# TAB 2: FORECASTING (O3 AQI)
# -----------------------------

with tab_forecast:
    st.subheader("📈 O₃ Air Quality Forecasting (O3 AQI)")

    # National daily O3 AQI series
    ts_df = (
        data.groupby("Date Local", as_index=False)
        .agg(O3_AQI=("O3 AQI", "mean"))
        .sort_values("Date Local")
    )

    ts_df["Year"] = ts_df["Date Local"].dt.year
    ts_df["Month"] = ts_df["Date Local"].dt.month
    ts_df["DayOfYear"] = ts_df["Date Local"].dt.dayofyear
    ts_df["DayOfWeek"] = ts_df["Date Local"].dt.dayofweek

    features = ["Year", "Month", "DayOfYear", "DayOfWeek"]
    X = ts_df[features]
    y = ts_df["O3_AQI"]

    split_idx = int(len(ts_df) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

    model_choice = st.selectbox(
        "Select forecasting model",
        ["XGBoost (tree-based)", "ARIMA (time-series)"]
    )

    horizon = st.slider(
        "Forecast horizon (days into the future)",
        min_value=3,
        max_value=30,
        value=7
    )

    last_date = ts_df["Date Local"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    if model_choice.startswith("XGBoost"):
        xgb_model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)

        future_feat = pd.DataFrame({"Date Local": future_dates})
        future_feat["Year"] = future_feat["Date Local"].dt.year
        future_feat["Month"] = future_feat["Date Local"].dt.month
        future_feat["DayOfYear"] = future_feat["Date Local"].dt.dayofyear
        future_feat["DayOfWeek"] = future_feat["Date Local"].dt.dayofweek

        future_feat["O3_AQI_Pred"] = xgb_model.predict(future_feat[features])

        fig_forecast = px.line(
            ts_df,
            x="Date Local",
            y="O3_AQI",
            title="Historical National Mean O3 AQI + XGBoost Forecast"
        )
        fig_forecast.add_scatter(
            x=future_feat["Date Local"],
            y=future_feat["O3_AQI_Pred"],
            mode="lines+markers",
            name="XGBoost Forecast"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

    else:
        if not HAS_ARIMA:
            st.error("ARIMA is not available. Please install `statsmodels` in your environment.")
        else:
            try:
                arima_model = ARIMA(y, order=(2, 1, 2)).fit()
                arima_forecast = arima_model.forecast(steps=horizon)
                arima_df = pd.DataFrame({
                    "Date Local": future_dates,
                    "O3_AQI_Pred": arima_forecast.values
                })

                fig_forecast = px.line(
                    ts_df,
                    x="Date Local",
                    y="O3_AQI",
                    title="Historical National Mean O3 AQI + ARIMA Forecast"
                )
                fig_forecast.add_scatter(
                    x=arima_df["Date Local"],
                    y=arima_df["O3_AQI_Pred"],
                    mode="lines+markers",
                    name="ARIMA Forecast"
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
            except Exception as e:
                st.error(f"ARIMA model failed to fit: {e}")


# -----------------------------
# TAB 3: PATTERNS (KMEANS + HIER + DBSCAN + UMAP)
# -----------------------------

with tab_patterns:
    st.subheader("🔍 Pollution Pattern Explorer")

    st.markdown(
        "This section applies **KMeans (k=8)**, **Hierarchical Clustering**, and "
        "**DBSCAN outlier detection** on multivariate pollution + temporal features, "
        "and visualizes them using **UMAP**."
    )

    # --- Build clustering dataframe consistent with notebook logic ---

    cluster_df = data.copy()

    # Temporal features already exist (Year, Month, DayOfYear, DayOfWeek)
    # Add Season
    cluster_df["Season"] = cluster_df["Month"].apply(get_season)
    cluster_df = pd.get_dummies(cluster_df, columns=["Season"], drop_first=True)

    # Spatial frequency encoding
    city_freq = cluster_df["City"].value_counts()
    state_freq = cluster_df["State"].value_counts()
    cluster_df["City_freq"] = cluster_df["City"].map(city_freq)
    cluster_df["State_freq"] = cluster_df["State"].map(state_freq)

    # Feature list
    season_cols = [c for c in cluster_df.columns if c.startswith("Season_")]

    cluster_features = [
        "NO2 Mean", "O3 Mean", "SO2 Mean", "CO Mean",
        "NO2 AQI", "O3 AQI", "SO2 AQI", "CO AQI",
        "Month", "DayOfYear", "DayOfWeek",
        "City_freq", "State_freq"
    ] + season_cols

    cluster_df = cluster_df.dropna(subset=cluster_features)
    X_clust = cluster_df[cluster_features].values

    scaler = StandardScaler()
    X_clust_scaled = scaler.fit_transform(X_clust)

    # --- KMeans ---
    kmeans = KMeans(n_clusters=8, random_state=42, n_init="auto")
    cluster_df["kmeans_cluster"] = kmeans.fit_predict(X_clust_scaled)

    # --- Hierarchical (on sample to save time/RAM) ---
    
    st.subheader("Hierarchical Clustering (Sampled)")
    # Choose number of clusters
    hier_n_clusters = st.sidebar.slider("Hierarchical Clusters (k)", 2, 12, 6)
    
    sample_size = min(5000, len(cluster_df))

# Sample indices directly from the matrix to avoid index mismatch
    sample_idx = np.random.choice(len(cluster_df), size=sample_size, replace=False)

    hier_sample = cluster_df.iloc[sample_idx].copy()
    X_sample = X_clust_scaled[sample_idx]


    hier_model = AgglomerativeClustering(
        n_clusters=hier_n_clusters,
        metric="euclidean",
        linkage="ward"
    )
    
    
    hier_labels = hier_model.fit_predict(X_sample)
    hier_sample["hier_cluster"] = hier_labels
    
    

    # --- DBSCAN ---
    st.markdown("#### DBSCAN Parameters")
    col_eps, col_min = st.columns(2)
    with col_eps:
        eps = st.slider("eps (radius)", 0.5, 5.0, 1.5, step=0.1)
    with col_min:
        min_samples = st.slider("min_samples", 5, 200, 50, step=5)

    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_df["dbscan_label"] = dbscan_model.fit_predict(X_clust_scaled)
    outlier_mask = cluster_df["dbscan_label"] == -1

    st.markdown(
        f"Detected **{outlier_mask.sum()}** DBSCAN outliers "
        f"({100*outlier_mask.mean():.2f}% of points)."
    )

    # --- UMAP Visualization ---
    if HAS_UMAP:
        umap_model = umap.UMAP(
            n_neighbors=30,
            min_dist=0.3,
            n_components=2,
            random_state=42
        )
        umap_emb = umap_model.fit_transform(X_clust_scaled)
        cluster_df["umap_x"] = umap_emb[:, 0]
        cluster_df["umap_y"] = umap_emb[:, 1]

        # KMeans view
        st.markdown("#### UMAP projection colored by **KMeans (k=8)**")
        fig_umap_km = px.scatter(
            cluster_df,
            x="umap_x",
            y="umap_y",
            color="kmeans_cluster",
            labels={"umap_x": "UMAP-1", "umap_y": "UMAP-2"},
            opacity=0.7,
            height=550
        )
        st.plotly_chart(fig_umap_km, use_container_width=True)

        # Hierarchical view (sample only)
        if len(hier_sample) > 0:
            st.markdown(f"#### UMAP projection for **Hierarchical Clustering ({hier_n_clusters} clusters)** (sample of {sample_size})")
            hier_sample = hier_sample.join(
                cluster_df[["umap_x", "umap_y"]],
                how="left"
            )

            fig_umap_hier = px.scatter(
                hier_sample,
                x="umap_x",
                y="umap_y",
                color="hier_cluster",
                title=f"Hierarchical Clustering (k={hier_n_clusters})",
                opacity=0.7,
                height=550
            )
            st.plotly_chart(fig_umap_hier, use_container_width=True)

        # DBSCAN view
        st.markdown("#### UMAP projection with **DBSCAN outliers** highlighted")
        inliers = cluster_df[~outlier_mask]
        outliers = cluster_df[outlier_mask]

        fig_umap_db = px.scatter(
            inliers,
            x="umap_x",
            y="umap_y",
            opacity=0.3,
            color_discrete_sequence=["lightgray"],
            height=550,
            labels={"umap_x": "UMAP-1", "umap_y": "UMAP-2"},
        )
        fig_umap_db.add_scatter(
            x=outliers["umap_x"],
            y=outliers["umap_y"],
            mode="markers",
            marker=dict(color="red", size=6),
            name="DBSCAN Outliers"
        )
        st.plotly_chart(fig_umap_db, use_container_width=True)
    else:
        st.warning("UMAP is not available. Please install `umap-learn` to see 2D pattern visualization.")

    st.markdown("---")
    st.markdown("### Pattern Explorer: find similar historical events (same KMeans cluster)")

    col1, col2, col3 = st.columns(3)

    with col1:
        ref_state = st.selectbox(
            "Select reference State",
            sorted(cluster_df["State"].unique())
        )

    with col2:
        ref_city = st.selectbox(
            "Select reference City",
            sorted(cluster_df[cluster_df["State"] == ref_state]["City"].unique())
        )

    available_dates = cluster_df[
        (cluster_df["State"] == ref_state) &
        (cluster_df["City"] == ref_city)
    ]["Date Local"].dt.date.unique()

    if len(available_dates) == 0:
        st.info("No dates found for this state/city in the clustering sample.")
    else:
        with col3:
            ref_date = st.date_input(
                "Select reference Date",
                min_value=min(available_dates),
                max_value=max(available_dates),
                value=min(available_dates)
            )

        ref_rows = cluster_df[
            (cluster_df["State"] == ref_state) &
            (cluster_df["City"] == ref_city) &
            (cluster_df["Date Local"].dt.date == ref_date)
        ]

        if ref_rows.empty:
            st.info("No record exactly matching this state/city/date. Try another date.")
        else:
            ref_row = ref_rows.iloc[0]
            ref_cluster = int(ref_row["kmeans_cluster"])
            st.markdown(f"**Reference event belongs to KMeans cluster:** `{ref_cluster}`")

            similar_events = cluster_df[cluster_df["kmeans_cluster"] == ref_cluster]

            st.markdown(
                f"Found **{len(similar_events)}** similar historical events "
                f"(same KMeans cluster → similar pollution pattern)."
            )

            st.dataframe(
                similar_events[[
                    "Date Local", "State", "City",
                    "NO2 Mean", "O3 Mean", "SO2 Mean", "CO Mean",
                    "NO2 AQI", "O3 AQI", "SO2 AQI", "CO AQI"
                ]].head(20)
            )

            st.write("**Cluster pollution signature (feature means):**")
            st.write(
                similar_events[[
                    "NO2 Mean", "O3 Mean", "SO2 Mean", "CO Mean",
                    "NO2 AQI", "O3 AQI", "SO2 AQI", "CO AQI"
                ]].mean().round(2)
            )


# -----------------------------
# TAB 4: HEALTH ADVISORY
# -----------------------------

with tab_health:
    st.subheader("❤️ Health Advisory – Based on Local AQI")

    def health_category(aqi):
        if aqi <= 50:
            return "Good", "Air quality is clean and safe."
        elif aqi <= 100:
            return "Moderate", "Acceptable air quality. Sensitive individuals should reduce heavy outdoor activity."
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups", "Sensitive individuals should avoid prolonged outdoor exposure."
        elif aqi <= 200:
            return "Unhealthy", "Everyone may begin to experience health effects."
        elif aqi <= 300:
            return "Very Unhealthy", "Health alert: increased risk for everyone."
        else:
            return "Hazardous", "Emergency conditions: serious risk for all individuals."

    col1, col2, col3 = st.columns(3)

    with col1:
        h_state = st.selectbox(
            "State (Health advisory)",
            sorted(filtered["State"].unique())
        )

    with col2:
        h_city = st.selectbox(
            "City (Health advisory)",
            sorted(filtered[filtered["State"] == h_state]["City"].unique())
        )

    with col3:
        h_date = st.date_input(
            "Date (Health advisory)",
            min_value=filtered["Date Local"].min().date(),
            max_value=filtered["Date Local"].max().date()
        )

    h_rows = filtered[
        (filtered["State"] == h_state) &
        (filtered["City"] == h_city) &
        (filtered["Date Local"].dt.date == h_date)
    ]

    if h_rows.empty:
        st.info("No measurement for this state/city/date in current filters.")
    else:
        h_row = h_rows.iloc[0]

        aqis = h_row[["NO2 AQI", "O3 AQI", "SO2 AQI", "CO AQI"]].dropna()
        if aqis.empty:
            st.info("No valid AQI values for this record.")
        else:
            overall_aqi = aqis.max()
            cat, advice = health_category(overall_aqi)

            st.markdown(f"**Overall AQI:** `{overall_aqi:.1f}` → **{cat}**")
            st.markdown(f"🩺 **Advice:** {advice}")

            st.write("Pollutant-specific AQI values:")
            st.write(
                h_row[["NO2 AQI", "O3 AQI", "SO2 AQI", "CO AQI"]]
            )


# -----------------------------
# TAB 5: DATA EXPORT
# -----------------------------

with tab_export:
    st.subheader("📤 Download Filtered Dataset")

    export_df = filtered.copy()
    export_df = export_df.sort_values("Date Local")

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download filtered data as CSV",
        data=csv_bytes,
        file_name="filtered_air_quality.csv",
        mime="text/csv"
    )

    st.markdown(
        "You can use this filtered dataset for further research, modeling, "
        "or reporting in external tools (Excel, Python, R, etc.)."
    )




