import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import math

st.set_page_config(page_title="Stuff+ Dashboard", layout="wide")
st.title("Stuff+ Dashboard")

# -----------------------------
# Remote data URLs
# -----------------------------
PITCHER_HISTORY_URL = "https://huggingface.co/datasets/perld/stuff-plus-data/resolve/main/pitcher_history.parquet"

DF_SCORED_YEAR_URLS = {
    2022: "https://huggingface.co/datasets/perld/stuff-plus-data/resolve/main/df_scored_2022.parquet",
    2023: "https://huggingface.co/datasets/perld/stuff-plus-data/resolve/main/df_scored_2023.parquet",
    2024: "https://huggingface.co/datasets/perld/stuff-plus-data/resolve/main/df_scored_2024.parquet",
    2025: "https://huggingface.co/datasets/perld/stuff-plus-data/resolve/main/df_scored_2025.parquet",
}

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_pitcher_history():
    return pd.read_parquet(PITCHER_HISTORY_URL)

@st.cache_data
def load_df_scored_year(year):
    url = DF_SCORED_YEAR_URLS[int(year)]
    return pd.read_parquet(url)

def add_arm_angle_line(fig, theta_deg, *, is_lefty: bool, xlim=(-25, 25), ylim=(-25, 25), origin_pad=2.0):
    
    if theta_deg is None or not np.isfinite(theta_deg):
        return

    theta = float(theta_deg)
    plot_theta = 180.0 - theta if is_lefty else theta
    plot_theta = max(0.1, min(plot_theta, 179.9))

    rad = math.radians(plot_theta)
    dx = math.cos(rad)
    dy = math.sin(rad)

    x_min, x_max = xlim
    y_min, y_max = ylim

    eps = 1e-12
    dx_safe = dx if abs(dx) > eps else (eps if dx >= 0 else -eps)

    tx = (x_max / dx_safe) if dx_safe > 0 else (x_min / dx_safe)
    ty = (y_max / dy)
    t_end = min(tx, ty)

    t0 = min(origin_pad, 0.9 * t_end)
    t1 = t_end
    if t1 <= t0:
        return

    fig.add_shape(
        type="line",
        x0=dx * t0, y0=dy * t0,
        x1=dx * t1, y1=dy * t1,
        line=dict(dash="dot", color="white", width=2),
    )

def build_usage_splits(player_df, pitch_order, pitch_type_col="pitch_type", min_pitches=25):

    total = len(player_df)
    total_lhh = (player_df["batter_handedness"] == "L").sum()
    total_rhh = (player_df["batter_handedness"] == "R").sum()

    pitch_counts = player_df[pitch_type_col].value_counts()
    allowed_pitches = set(pitch_counts[pitch_counts >= min_pitches].index)

    rows = []

    for pitch in pitch_order:
        if pitch not in allowed_pitches:
            continue

        df_pitch = player_df[player_df[pitch_type_col] == pitch]

        pitch_count = len(df_pitch)
        overall = 100 * pitch_count / total if total > 0 else 0

        lhh_count = (df_pitch["batter_handedness"] == "L").sum()
        rhh_count = (df_pitch["batter_handedness"] == "R").sum()

        vs_lhh = 100 * lhh_count / total_lhh if total_lhh > 0 else 0
        vs_rhh = 100 * rhh_count / total_rhh if total_rhh > 0 else 0

        rows.append({
            "Pitch": pitch,
            "Overall": f"{overall:.1f}%",
            "vs LHH": f"{vs_lhh:.1f}%",
            "vs RHH": f"{vs_rhh:.1f}%"
        })

    return pd.DataFrame(rows)

def build_pitch_finder_table(df, min_pitches=25):
    d = df.copy()

    d["HB"] = pd.to_numeric(d["HB_obs"], errors="coerce")
    d["iVB"] = pd.to_numeric(d["iVB_obs"], errors="coerce")
    d["arm_angle"] = pd.to_numeric(d.get("arm_angle"), errors="coerce")
    d["release_speed"] = pd.to_numeric(d["release_speed"], errors="coerce")
    d["release_spin_rate"] = pd.to_numeric(d["release_spin_rate"], errors="coerce")
    d["release_extension"] = pd.to_numeric(d["release_extension"], errors="coerce")
    d["Stuff+_pt"] = pd.to_numeric(d["Stuff+_pt"], errors="coerce")
    d["spin_axis"] = pd.to_numeric(d.get("spin_axis"), errors="coerce")
    d["ssw_in"] = pd.to_numeric(d.get("ssw_in"), errors="coerce")
    d["spin_efficiency"] = pd.to_numeric(d.get("spin_efficiency"), errors="coerce")

    agg_dict = {
        "Pitches": ("pitch_type", "size"),
        "StuffPlus": ("Stuff+_pt", "mean"),
        "Velo": ("release_speed", "mean"),
        "iVB": ("iVB", "mean"),
        "HB": ("HB", "mean"),
        "Spin": ("release_spin_rate", "mean"),
        "ArmAngle": ("arm_angle", "mean"),
        "Extension": ("release_extension", "mean"),
        "SpinAxis": ("spin_axis", "mean"),
        "SSW": ("ssw_in", "mean"),
        "SpinEff": ("spin_efficiency", "mean"),
    }

    if "is_lefty" in d.columns:
        agg_dict["is_lefty"] = ("is_lefty", "max")

    g = d.groupby(["game_year", "PlayerName", "pitch_type"], as_index=False).agg(**agg_dict)

    totals = (
        g.groupby(["game_year", "PlayerName"], as_index=False)["Pitches"]
         .sum()
         .rename(columns={"Pitches": "TotalPitches"})
    )

    g = g.merge(totals, on=["game_year", "PlayerName"], how="left")
    g["Usage %"] = np.where(g["TotalPitches"] > 0, 100 * g["Pitches"] / g["TotalPitches"], np.nan)

    g = g[g["Pitches"] >= min_pitches].copy()

    if "is_lefty" in g.columns:
        g["Handedness"] = np.where(g["is_lefty"] == 1, "LHP", "RHP")
    else:
        g["Handedness"] = "Unknown"

    g["Stuff+"] = g["StuffPlus"].round(0)
    g["Velo"] = g["Velo"].round(1)
    g["iVB"] = g["iVB"].round(1)
    g["HB"] = g["HB"].round(1)
    g["Spin"] = g["Spin"].round(0)
    g["Arm Angle"] = g["ArmAngle"].round(1)
    g["Extension"] = g["Extension"].round(1)
    g["Spin Axis"] = g["SpinAxis"].round(0)
    g["Usage %"] = g["Usage %"].round(1)
    g["SSW"] = g["SSW"].round(1)
    g["Spin Eff."] = g["SpinEff"].round(3)

    g = g.rename(columns={
        "game_year": "Year",
        "PlayerName": "Pitcher",
        "pitch_type": "Pitch",
    })

    return g[[
        "Year", "Pitcher", "Handedness", "Pitch", "Pitches", "Usage %",
        "Stuff+", "Velo", "iVB", "HB", "Spin", "Spin Axis", "Spin Eff.", "SSW",
        "Arm Angle", "Extension"
    ]]

PITCH_COLORS = {
    # Fastballs
    "FF": "#d85a6f",   # 4-Seam
    "SI": "#f5b335",   # Sinker
    "FA": "#d85a6f",

    # Breaking balls
    "SL": "#e5dc3a",   # Slider
    "ST": "#d7b65c",   # Sweeper
    "CU": "#41c7de",   # Curve
    "KC": "#41c7de",

    # Offspeed
    "CH": "#4cc95a",   # Change
    "FS": "#5faeb0",   # Splitter
    "FO": "#5faeb0",   # Forkball

    # Cutters
    "FC": "#b56a57",
}

PITCH_ORDER = [
    "FF",
    "SI",
    "FC",
    "SL",
    "ST",
    "CU",
    "KC",
    "CH",
    "FS",
    "FO",
]

pitcher_history = load_pitcher_history()

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Filters")

years = sorted(pitcher_history["game_year"].dropna().unique().tolist(), reverse=True)
year = st.sidebar.selectbox("Season", years, index=0)

df_scored = load_df_scored_year(year)

# Min total pitches filter (pitcher-season)
min_total_pitches = st.sidebar.slider("Min total pitches (season)", 0, 5000, 0, step=100)
min_pitches_by_type = 25

# -----------------------------
# Pitcher list + pitcher selector
# -----------------------------
ph = pitcher_history.copy()
ph_year = ph[ph["game_year"] == year].copy()

pitcher_totals = (
    ph_year.groupby("PlayerName", as_index=False)["Pitches"]
    .sum()
    .rename(columns={"Pitches": "n_total"})
)

pitcher_totals = pitcher_totals[
    pd.to_numeric(pitcher_totals["n_total"], errors="coerce") >= min_total_pitches
].copy()

pitcher_list = (
    pitcher_totals["PlayerName"]
    .dropna()
    .sort_values()
    .unique()
    .tolist()
)

previous_pitcher = st.session_state.get("pitcher_name")

if len(pitcher_list) == 0:
    pitcher_name = None
    st.session_state["pitcher_name"] = None
    st.sidebar.info("No pitchers match the current filters.")
else:
    if previous_pitcher in pitcher_list:
        default_index = pitcher_list.index(previous_pitcher)
    else:
        default_index = 0

    pitcher_name = st.sidebar.selectbox(
        "Pitcher",
        pitcher_list,
        index=default_index,
        key="pitcher_name",
    )

if pitcher_name is None:
    st.stop()
    
# -----------------------------
# Tabs
# -----------------------------
tab_profile, tab_lb, tab_finder = st.tabs(["Pitcher Profile", "Stuff+ Leaderboard", "Pitch Finder"])

# -----------------------------
# Filter pitcher-season rows (used on profile + scatter)
# -----------------------------
with tab_profile:
    # Pitch-level data for selected pitcher
    dfp = df_scored[df_scored["PlayerName"] == pitcher_name].copy()

    # Header summary from pitcher_history
    row = pitcher_history[
        (pitcher_history["game_year"] == year) &
        (pitcher_history["PlayerName"] == pitcher_name)
    ].copy()

    if len(row) == 0:
        st.warning("No pitcher summary found for this selection.")
    else:
        n_total = int(row["Pitches"].sum())

        overall_stuff = (
            np.average(row["StuffPlus"], weights=row["Pitches"])
            if row["Pitches"].sum() > 0 else np.nan
        )

        primary_idx = row["Pitches"].idxmax()
        primary_pitch = row.loc[primary_idx, "pitch_type"] if "pitch_type" in row.columns else None
        primary_pitch_usage = row.loc[primary_idx, "Pitches"] / n_total if n_total > 0 else np.nan

        if "is_lefty" in dfp.columns and dfp["is_lefty"].notna().any():
            is_lefty = int(dfp["is_lefty"].iloc[0]) == 1
        else:
            is_lefty = False

        handedness = "LHP" if is_lefty else "RHP"

        c1, c2, c3, c4 = st.columns([2.2, 1.2, 1.2, 1.4])

        with c1:
            st.subheader(f"{pitcher_name} — {year}")

        with c2:
            st.metric("Handedness", handedness)

        with c3:
            st.metric(f"{year} Pitches", f"{n_total}")

        with c4:
            st.metric(
                "Overall Stuff+",
                f"{overall_stuff:.0f}" if pd.notna(overall_stuff) else "NA"
            )

        if pd.notna(primary_pitch) and pd.notna(primary_pitch_usage):
            st.caption(f"Primary pitch: {primary_pitch} ({primary_pitch_usage:.0%})")

    st.divider()
    
    # Determine handedness from pitch-level data
    if "is_lefty" in dfp.columns and dfp["is_lefty"].notna().any():
        raw_lefty = dfp["is_lefty"].iloc[0]
    else:
        raw_lefty = 0

    is_lefty = int(raw_lefty) == 1
    handedness = "LHP" if is_lefty else "RHP"

    dfp["HB"] = pd.to_numeric(dfp["HB_obs"], errors="coerce")
    dfp["iVB"] = pd.to_numeric(dfp["iVB_obs"], errors="coerce")
    dfp["arm_angle"] = pd.to_numeric(dfp.get("arm_angle"), errors="coerce")
    dfp["ssw_in"] = pd.to_numeric(dfp.get("ssw_in"), errors="coerce")
    dfp["spin_efficiency"] = pd.to_numeric(dfp.get("spin_efficiency"), errors="coerce")

    # (2) Arsenal table (grouped)
    ars = (
        dfp.groupby("pitch_type", as_index=False)
           .agg(
               Pitches=("pitch_type", "size"),
               StuffPlus=("Stuff+_pt", "mean"),
               Velo=("release_speed", "mean"),
               HB=("HB", "mean"),
               iVB=("iVB", "mean"),
               ArmAngle=("arm_angle", "mean"),
               Extension=("release_extension", "mean"),
               SSW=("ssw_in", "mean"),
               SpinEff=("spin_efficiency", "mean"),
               Spin=("release_spin_rate", "mean"),
               SpinAxis=("spin_axis", "mean"),
           )
    )

    # Filter by min pitches per pitch type
    ars = ars[ars["Pitches"] >= min_pitches_by_type].copy()

    # Sort by Usage %
    ars = ars.sort_values("Pitches", ascending=False)

    # Display formatting
    ars_display = ars.copy()
    ars_display["StuffPlus"] = ars_display["StuffPlus"].round(0)
    ars_display["Velo"] = ars_display["Velo"].round(1)
    ars_display["iVB"] = ars_display["iVB"].round(1)
    ars_display["HB"] = ars_display["HB"].round(1)
    ars_display["Extension"] = ars_display["Extension"].round(1)
    ars_display["Arm Angle"] = pd.to_numeric(ars_display["ArmAngle"], errors="coerce").round(1)
    ars_display["Spin"] = ars_display["Spin"].round(0)
    ars_display["Spin Axis"] = pd.to_numeric(ars_display["SpinAxis"], errors="coerce").round(0)
    ars_display["SSW"] = pd.to_numeric(ars_display["SSW"], errors="coerce").round(1)
    ars_display["Spin Eff."] = (
        pd.to_numeric(ars_display["SpinEff"], errors="coerce")
        .mul(100).round(0)
        .map(lambda x: f"{x:.0f}%" if pd.notna(x) else ""))
    ars_display = ars_display.rename(columns={"StuffPlus": "Stuff+", "pitch_type": "Pitch"})
    arsenal_pitch_order = ars_display["Pitch"].tolist()
    
    usage_splits = build_usage_splits(
        dfp,
        pitch_order=arsenal_pitch_order,
        pitch_type_col="pitch_type",
        min_pitches=min_pitches_by_type
    )
    
    # Layout: table + movement plot
    left, right = st.columns([1.35, 1.0])

    with left:
        st.subheader("Overview")
        st.dataframe(
            ars_display[[
                "Pitch", "Pitches", "Stuff+", "Velo", "iVB", "HB"
            ]],
            use_container_width=True,
            hide_index=True,
        )
        
        st.subheader("Usage Splits")
        st.dataframe(
            usage_splits,
            use_container_width=True,
            hide_index=True,
        )

    arm_angle = float(dfp["arm_angle"].mean()) if dfp["arm_angle"].notna().any() else None

    with right:
        st.subheader("Movement")
        if len(dfp) == 0:
            st.info("No pitches available.")
        else:
            fig = px.scatter(
                dfp,
                x="HB",
                y="iVB",
                color="pitch_type",
                color_discrete_map=PITCH_COLORS,
                category_orders={"pitch_type": [p for p in PITCH_ORDER if p in dfp["pitch_type"].dropna().unique()]},
                opacity=0.75,
                hover_data={
                    "pitch_type": True,
                    "release_speed": ':.1f',
                    "release_spin_rate": ':.0f',
                    "Stuff+_pt": ':.1f',
                },
                labels={
                    "HB": "1B \u2194 3B",
                    "iVB": "iVB",
                },
            )
            
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )
            
            fig.update_traces(
                marker=dict(
                    size=8,
                    line=dict(width=0.5, color="black")
                )
            )

            fig.update_xaxes(range=[-25, 25], zeroline=False)
            fig.update_yaxes(range=[-25, 25], zeroline=False)

            fig.add_shape(
                type="line",
                x0=0, x1=0,
                y0=-25, y1=25,
                line=dict(dash="dot", color="white", width=2)
            )
            fig.add_shape(
                type="line",
                x0=-25, x1=25,
                y0=0, y1=0,
                line=dict(dash="dot", color="white", width=2)
            )

            # Arm-angle reference line
            if arm_angle is not None:
                add_arm_angle_line(fig, arm_angle, is_lefty=is_lefty, xlim=(-25, 25), ylim=(-25, 25), origin_pad=0.8)
                fig.add_annotation(
                    x=24,
                    y=-23,
                    text=f"Arm Angle: {arm_angle:.0f}°",
                    showarrow=False,
                    xanchor="right",
                    yanchor="bottom",
                    font=dict(color="white", size=12),
                    bgcolor="rgba(0,0,0,0.35)"
                )

            fig.update_layout(
                height=450,
                margin=dict(l=10, r=10, t=30, b=10),
                dragmode=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "staticPlot": True,
                    "displayModeBar": False,
                },
            )

    # -----------------------------
    # Historical tables (below Arsenal/Movement)
    # -----------------------------
    hist = pitcher_history[pitcher_history["PlayerName"] == pitcher_name].copy()
    hist = hist[hist["game_year"].notna() & hist["pitch_type"].notna()].copy()

    hist["HB"] = pd.to_numeric(hist["HB"], errors="coerce")
    hist["iVB"] = pd.to_numeric(hist["iVB"], errors="coerce")
    hist["SSW"] = pd.to_numeric(hist["SSW"], errors="coerce")
    hist["SpinEff"] = pd.to_numeric(hist["SpinEff"], errors="coerce")
    hist["Spin"] = pd.to_numeric(hist["Spin"], errors="coerce")
    hist["SpinAxis"] = pd.to_numeric(hist["SpinAxis"], errors="coerce")
    hist["ArmAngle"] = pd.to_numeric(hist["ArmAngle"], errors="coerce")
    hist["Extension"] = pd.to_numeric(hist["Extension"], errors="coerce")
    hist["StuffPlus"] = pd.to_numeric(hist["StuffPlus"], errors="coerce")
    hist["Pitches"] = pd.to_numeric(hist["Pitches"], errors="coerce")

    if len(hist) == 0:
        st.info("No historical pitch data available for this pitcher.")
    else:
        col1, col2 = st.columns(2)

        # (1) Historical Arsenal
        with col1:
            st.subheader("Arsenal")

            g = hist[["game_year", "pitch_type", "Pitches"]].copy().rename(columns={"pitch_type": "Pitch"})
            totals = (
                g.groupby("game_year", as_index=False)["Pitches"]
                .sum()
                .rename(columns={"Pitches": "Pitches_total"})
            )
            g = g.merge(totals, on="game_year", how="left")
            g["Usage"] = g["Pitches"] / g["Pitches_total"]
            
            usage_wide = g.pivot(index="game_year", columns="Pitch", values="Usage").fillna(0.0)
            usage_wide = usage_wide.sort_index(ascending=False)

            present_pitches = [p for p in PITCH_ORDER if p in usage_wide.columns]
            usage_wide = usage_wide.reindex(columns=present_pitches)

            usage_wide.columns = [f"{c} %" for c in usage_wide.columns]

            usage_wide_display = usage_wide.copy()
            for c in usage_wide_display.columns:
                usage_wide_display[c] = (
                    pd.to_numeric(usage_wide_display[c], errors="coerce")
                    .mul(100)
                    .round(1)
                    .map(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
                )

            usage_wide_display.index = usage_wide_display.index.astype(int)
            usage_wide_display = usage_wide_display.reset_index().rename(columns={"game_year": "Year"})
            column_config = {"Year": st.column_config.NumberColumn(width="small")}

            for c in usage_wide_display.columns:
                if c != "Year":
                    column_config[c] = st.column_config.TextColumn(width="small")
            
            st.dataframe(
                usage_wide_display,
                use_container_width=True,
                hide_index=True,
                column_config=column_config
            )
            
        # (2) Historical Stuff+
        with col2:
            st.subheader("Stuff+")

            s = hist[["game_year", "pitch_type", "Pitches", "StuffPlus"]].copy().rename(columns={"pitch_type": "Pitch"})
            MIN_PITCHES_FOR_YEAR_PITCH = 25
            s.loc[s["Pitches"] < MIN_PITCHES_FOR_YEAR_PITCH, "StuffPlus"] = np.nan

            overall_year = (
                s.groupby("game_year", as_index=False)
                .apply(lambda x: pd.Series({
                    "Stuff+": float(np.average(x["StuffPlus"], weights=x["Pitches"]))
                    if x["StuffPlus"].notna().any() and x["Pitches"].sum() > 0 else np.nan
                }))
                .reset_index(drop=True)
            )

            stuff_wide = s.pivot(index="game_year", columns="Pitch", values="StuffPlus")

            present_pitches = [p for p in PITCH_ORDER if p in stuff_wide.columns]
            stuff_wide = stuff_wide.reindex(columns=present_pitches)

            stuff_wide = stuff_wide.rename(columns=lambda c: f"{c} Stuff+")

            out = overall_year.merge(stuff_wide, left_on="game_year", right_index=True, how="left")
            out = out.sort_values("game_year", ascending=False)

            out["game_year"] = out["game_year"].astype(int)
            out["Stuff+"] = out["Stuff+"].round(0)

            pitch_cols = [f"{p} Stuff+" for p in present_pitches]
            for c in pitch_cols:
                out[c] = pd.to_numeric(out[c], errors="coerce").round(0)

            out = out.rename(columns={"game_year": "Year"})

            column_config = {
                c: st.column_config.NumberColumn(width="small")
                for c in (["Year", "Stuff+"] + pitch_cols)
            }

            stuff_display = out.copy()

            # Turn values into display-friendly strings
            stuff_display["Stuff+"] = (
                pd.to_numeric(stuff_display["Stuff+"], errors="coerce")
                .map(lambda x: f"{x:.0f}" if pd.notna(x) else "")
            )

            for c in pitch_cols:
                stuff_display[c] = (
                    pd.to_numeric(stuff_display[c], errors="coerce")
                    .map(lambda x: f"{x:.0f}" if pd.notna(x) else "")
                )

            column_config = {
                "Year": st.column_config.NumberColumn(width="small"),
                "Stuff+": st.column_config.TextColumn(width="small"),
            }

            for c in pitch_cols:
                column_config[c] = st.column_config.TextColumn(width="small")

            st.dataframe(
                stuff_display[["Year", "Stuff+"] + pitch_cols],
                use_container_width=True,
                hide_index=True,
                column_config=column_config
            )
            
        # -----------------------------
        # Historical Shape / Spin
        # -----------------------------
        hist_shape = hist.rename(columns={
            "pitch_type": "Pitch",
            "ArmAngle": "Arm Angle",
            "SpinAxis": "Spin Axis",
            "SpinEff": "Spin Eff."
        })[[
            "game_year", "Pitch", "Pitches", "Velo", "iVB", "HB",
            "Arm Angle", "Extension", "SSW", "Spin Eff.", "Spin", "Spin Axis"
        ]].copy()

        MIN_PITCHES_FOR_YEAR_PITCH = 25
        hist_shape.loc[
            hist_shape["Pitches"] < MIN_PITCHES_FOR_YEAR_PITCH,
            ["Velo", "iVB", "HB", "Arm Angle", "Extension", "SSW", "Spin Eff.", "Spin", "Spin Axis"]
        ] = np.nan

                # ---------- Historical Shape ----------
        st.subheader("Shape")

        shape_disp = hist_shape[
            ["game_year", "Pitch", "Velo", "iVB", "HB", "Arm Angle", "Extension"]
        ].copy()

        shape_disp["Velo"] = pd.to_numeric(shape_disp["Velo"], errors="coerce").round(1)
        shape_disp["iVB"] = pd.to_numeric(shape_disp["iVB"], errors="coerce").round(1)
        shape_disp["HB"] = pd.to_numeric(shape_disp["HB"], errors="coerce").round(1)
        shape_disp["Arm Angle"] = pd.to_numeric(shape_disp["Arm Angle"], errors="coerce").round(1)
        shape_disp["Extension"] = pd.to_numeric(shape_disp["Extension"], errors="coerce").round(1)

        present_pitches = list(shape_disp["Pitch"].dropna().unique())
        final_pitches = [p for p in PITCH_ORDER if p in present_pitches]

        for pitch in final_pitches:
            pitch_df = shape_disp[shape_disp["Pitch"] == pitch].copy()
            pitch_df = pitch_df.sort_values("game_year", ascending=False)
            pitch_df = pitch_df.rename(columns={"game_year": "Year"})
            pitch_df = pitch_df[["Year", "Velo", "iVB", "HB", "Arm Angle", "Extension"]]
            pitch_df = pitch_df.replace({None: "", "None": "", np.nan: ""})

            with st.expander(f"{pitch}"):
                st.dataframe(
                    pitch_df,
                    use_container_width=True,
                    hide_index=True,
                )

        # ---------- Historical Spin / SSW ----------
        st.subheader("Spin")

        spin_disp = hist_shape[
            ["game_year", "Pitch", "Spin", "Spin Axis", "Spin Eff.", "SSW"]
        ].copy()

        spin_disp["SSW"] = pd.to_numeric(spin_disp["SSW"], errors="coerce").round(1)
        spin_disp["Spin Eff."] = pd.to_numeric(spin_disp["Spin Eff."], errors="coerce").round(2)
        spin_disp["Spin"] = pd.to_numeric(spin_disp["Spin"], errors="coerce").round(0)
        spin_disp["Spin Axis"] = pd.to_numeric(spin_disp["Spin Axis"], errors="coerce").round(0)
        spin_disp["Spin Eff."] = (
            pd.to_numeric(spin_disp["Spin Eff."], errors="coerce")
            .mul(100)
            .round(0)
            .map(lambda x: f"{x:.0f}%" if pd.notna(x) else "")
        )

        present_pitches = list(spin_disp["Pitch"].dropna().unique())
        final_pitches = [p for p in PITCH_ORDER if p in present_pitches]

        for pitch in final_pitches:
            pitch_df = spin_disp[spin_disp["Pitch"] == pitch].copy()
            pitch_df = pitch_df.sort_values("game_year", ascending=False)
            pitch_df = pitch_df.rename(columns={"game_year": "Year"})
            pitch_df = pitch_df[["Year", "Spin", "Spin Axis", "Spin Eff.", "SSW"]]
            pitch_df = pitch_df.replace({None: "", "None": "", np.nan: ""})

            with st.expander(f"{pitch}"):
                st.dataframe(
                    pitch_df,
                    use_container_width=True,
                    hide_index=True,
                )
        
    st.divider()

    # -----------------------------
    # Stuff+ scatter: selectable axes, colored by Stuff+
    # -----------------------------
    st.subheader("Stuff+")

    axis_candidates = {
        "HB": "HB",
        "iVB": "iVB",
        "Velo": "release_speed",
        "Spin": "release_spin_rate",
        "Extension": "release_extension",
        "Release X": "release_pos_x",
        "Release Z": "release_pos_z",
        "Spin Axis X": "spin_axis_x",
        "Spin Axis Y": "spin_axis_y",
        "Spin Efficiency": "spin_efficiency",
        "Δ Velo vs Primary": "primary_delta_release_speed",
        "Δ HB vs Primary": "primary_delta_pfx_x",
        "Δ iVB vs Primary": "primary_delta_pfx_z",
    }

    c1, c2, c3 = st.columns([1.0, 1.0, 1.6])

    with c1:
        x_label = st.selectbox(
            "X axis",
            list(axis_candidates.keys()),
            index=list(axis_candidates.keys()).index("HB")
        )
    with c2:
        y_label = st.selectbox(
            "Y axis",
            list(axis_candidates.keys()),
            index=list(axis_candidates.keys()).index("iVB")
        )
    with c3:
        pitch_filter = st.multiselect(
            "Pitch types",
            sorted(dfp["pitch_type"].dropna().unique().tolist()),
            default=sorted(dfp["pitch_type"].dropna().unique().tolist()),
        )
        
    color_col = "Stuff+_pt"

    plot_df = dfp[dfp["pitch_type"].isin(pitch_filter)].copy()

    if color_col not in plot_df.columns:
        st.info("Stuff+_pt is not available for this plot.")
        st.stop()

    xcol = axis_candidates[x_label]
    ycol = axis_candidates[y_label]

    for col in [xcol, ycol, color_col]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    plot_df = plot_df.dropna(subset=[xcol, ycol, color_col])

    if len(plot_df) == 0:
        st.info("No data after filters for the scatter plot.")
    else:
        x_title = "1B \u2194 3B" if x_label == "HB" else x_label
        y_title = y_label

        fig2 = px.scatter(
            plot_df,
            x=xcol,
            y=ycol,
            color=color_col,
            color_continuous_scale="RdYlBu_r",
            color_continuous_midpoint=100,
            hover_data=["pitch_type"],
            labels={xcol: x_title, ycol: y_title, color_col: "Stuff+", "pitch_type": "Pitch"},
            range_color=[80, 120],
        )

        if x_label == "HB":
            fig2.update_xaxes(range=[-25, 25], zeroline=False)
        else:
            fig2.update_xaxes(zeroline=False)

        if y_label == "iVB":
            fig2.update_yaxes(range=[-25, 25], zeroline=False)
        else:
            fig2.update_yaxes(zeroline=False)

        if x_label == "HB":
            fig2.add_shape(
                type="line",
                x0=0, x1=0,
                y0=-25 if y_label == "iVB" else float(plot_df[ycol].min()),
                y1=25 if y_label == "iVB" else float(plot_df[ycol].max()),
                line=dict(dash="dot", color="white", width=2)
            )

        if y_label == "iVB":
            fig2.add_shape(
                type="line",
                x0=-25 if x_label == "HB" else float(plot_df[xcol].min()),
                x1=25 if x_label == "HB" else float(plot_df[xcol].max()),
                y0=0, y1=0,
                line=dict(dash="dot", color="white", width=2)
            )

        fig2.update_layout(
            height=550,
            margin=dict(l=10, r=10, t=30, b=10),
            dragmode=False
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
            config={
                "staticPlot": True,
                "displayModeBar": False,
            },
        )

# -----------------------------
# LEADERBOARD TAB (with pagination)
# -----------------------------
with tab_lb:
    st.subheader("Stuff+ Leaderboard")

    # Controls
    page_size = st.selectbox("Rows per page", [10, 25, 30, 50, 100, 200], index=2)
    page = st.number_input("Page", min_value=1, value=1, step=1)
    min_pitch_for_pitchcol = 25

    # Build pitcher-year leaderboard from df_scored (FanGraphs style):
    # Overall Stuff+ first, then Stuff+ by pitch as columns
    d = df_scored.copy()
    d = d[d["PlayerName"].notna() & d["pitch_type"].notna()].copy()

    # Enforce "min total pitches (season)" same as sidebar filter
    # (compute from df_scored so it always matches)
    tot = d.groupby("PlayerName", as_index=False).size().rename(columns={"size": "Pitches_total"})
    keep_names = tot.loc[tot["Pitches_total"] >= min_total_pitches, "PlayerName"]
    d = d[d["PlayerName"].isin(keep_names)].copy()

    # Pitch-level aggregates
    g = (
        d.groupby(["PlayerName", "pitch_type"], as_index=False)
         .agg(
             Pitches=("pitch_type", "size"),
             StuffPlus=("Stuff+_pt", "mean"),
         )
         .rename(columns={"pitch_type": "Pitch"})
    )

    # Usage per pitch (within pitcher-season)
    pitcher_totals = (
        g.groupby("PlayerName", as_index=False)["Pitches"]
         .sum()
         .rename(columns={"Pitches": "Pitches_total"})
    )
    g = g.merge(pitcher_totals, on="PlayerName", how="left")
    g["Usage %"] = g["Pitches"] / g["Pitches_total"]

    # Weighted overall Stuff+ per pitcher (weighted by pitch counts)
    overall = (
        g.groupby("PlayerName", as_index=False)
         .apply(lambda x: pd.Series({
             "Stuff+": np.average(x["StuffPlus"], weights=x["Pitches"]) if x["Pitches"].sum() > 0 else np.nan,
             "Pitches": int(x["Pitches"].sum()),
         }))
         .reset_index(drop=True)
    )

    # Wide pivot for Stuff+ by pitch (only show pitch Stuff+ if enough pitches)
    g_pitch = g.copy()
    g_pitch.loc[g_pitch["Pitches"] < min_pitch_for_pitchcol, "StuffPlus"] = np.nan

    wide = g_pitch.pivot(index="PlayerName", columns="Pitch", values="StuffPlus")

    # Rename columns to "Stuff+ XX"
    wide = wide.rename(columns={c: f"Stuff+ {c}" for c in wide.columns})

    # Combine
    lb = overall.merge(wide, left_on="PlayerName", right_index=True, how="left")

    # Round Stuff+ columns to nearest int (including pitch columns)
    stuff_cols = [c for c in lb.columns if c == "Stuff+" or c.startswith("Stuff+ ")]
    for c in stuff_cols:
        lb[c] = pd.to_numeric(lb[c], errors="coerce").round(0)

    # Sort by overall Stuff+ (desc), then Pitches (desc)
    lb = lb.sort_values(["Stuff+", "Pitches"], ascending=[False, False])

    # Order columns: overall Stuff+ up front, then pitch columns
    pitch_cols_sorted = sorted([c for c in lb.columns if c.startswith("Stuff+ ")])
    show_cols = ["PlayerName", "Stuff+", "Pitches"] + pitch_cols_sorted

    # Pagination
    total_rows = len(lb)
    total_pages = max(1, int(np.ceil(total_rows / page_size)))
    if page > total_pages:
        page = total_pages

    start = (page - 1) * page_size
    end = start + page_size

    st.caption(f"Showing rows {start+1}–{min(end, total_rows)} of {total_rows} (Page {page} of {total_pages})")

    st.dataframe(
        lb[show_cols].iloc[start:end].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Pitch Usage")

    # Wide usage table: columns = pitch types only
    wide_u = g.pivot(index="PlayerName", columns="Pitch", values="Usage %")
    wide_u = wide_u.rename(columns=lambda c: f"{c} %")

    # Combine Pitches + usage columns
    usage_lb = overall[["PlayerName", "Pitches"]].merge(wide_u, left_on="PlayerName", right_index=True, how="left")

    # Sort by Pitches decreasing
    usage_lb = usage_lb.sort_values("Pitches", ascending=False)

    # Format percentages like 69.7%
    usage_disp = usage_lb.copy()
    pitch_cols = sorted([c for c in usage_disp.columns if c not in ("PlayerName", "Pitches")])

    for c in pitch_cols:
        usage_disp[c] = (
            pd.to_numeric(usage_disp[c], errors="coerce")
            .mul(100)
            .round(1)
            .map(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
        )

    # Apply pagination (same page/page_size as leaderboard)
    total_rows_u = len(usage_disp)
    total_pages_u = max(1, int(np.ceil(total_rows_u / page_size)))
    if page > total_pages_u:
        page = total_pages_u

    start = (page - 1) * page_size
    end = start + page_size

    # Display
    st.dataframe(
        usage_disp[["PlayerName", "Pitches"] + pitch_cols].iloc[start:end].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    st.caption("Usage values are percentages within each pitcher-season.")
    
st.divider()

def get_numeric_slider_bounds(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return None

    smin = float(s.min())
    smax = float(s.max())

    if np.isclose(smin, smax):
        return None

    is_int_like = np.allclose(s, np.round(s), equal_nan=True)

    if is_int_like:
        lo = int(np.floor(smin))
        hi = int(np.ceil(smax))
        return {
            "min": lo,
            "max": hi,
            "default": (lo, hi),
            "step": 1,
            "format": None,
            "kind": "int",
        }

    span = smax - smin
    if span <= 2:
        step = 0.01
        fmt = "%.2f"
    else:
        step = 0.1
        fmt = "%.1f"

    lo = float(np.floor(smin * 10) / 10)
    hi = float(np.ceil(smax * 10) / 10)

    return {
        "min": lo,
        "max": hi,
        "default": (lo, hi),
        "step": step,
        "format": fmt,
        "kind": "float",
    }

def is_numeric_column(series):
    return pd.api.types.is_numeric_dtype(series)


def is_reasonable_categorical(series, max_unique=40):
    s = series.dropna()
    if len(s) == 0:
        return False
    return s.nunique() <= max_unique

with tab_finder:
    st.subheader("Pitch Finder")

    df_finder_year = df_scored.copy()

    finder_df = build_pitch_finder_table(
        df_finder_year,
        min_pitches=25
    )

    if len(finder_df) == 0:
        st.info("No pitch finder rows available.")
        st.stop()

    # -----------------------------
    # Reset button
    # -----------------------------
    if st.button("Reset Pitch Finder Filters"):
        keys_to_clear = [k for k in st.session_state.keys() if k.startswith("pitch_finder_")]
        for k in keys_to_clear:
            del st.session_state[k]
        st.rerun()

    # -----------------------------
    # Top row: categorical + sort
    # -----------------------------
    top1, top2, top3, top4 = st.columns([1.6, 1.0, 1.2, 0.7])

    with top1:
        all_pitches = [p for p in PITCH_ORDER if p in finder_df["Pitch"].dropna().unique()]
        selected_pitches = st.multiselect(
            "Pitch types",
            all_pitches,
            default=all_pitches,
            key="pitch_finder_pitch_types"
        )

    with top2:
        handedness_options = ["All"]
        if "Handedness" in finder_df.columns:
            handedness_values = finder_df["Handedness"].dropna().unique().tolist()
            handedness_values = [x for x in ["RHP", "LHP", "Unknown"] if x in handedness_values]
            handedness_options += handedness_values

        handedness_filter = st.selectbox(
            "Pitcher handedness",
            handedness_options,
            index=0,
            key="pitch_finder_handedness"
        )

    with top3:
        sortable_cols = [c for c in finder_df.columns if c not in {"Pitcher", "Handedness", "Pitch"}]
        default_sort = "Stuff+" if "Stuff+" in sortable_cols else sortable_cols[0]
        sort_col = st.selectbox(
            "Sort by",
            sortable_cols,
            index=sortable_cols.index(default_sort),
            key="pitch_finder_sort_col"
        )

    with top4:
        sort_desc = st.checkbox("Descending", value=True, key="pitch_finder_sort_desc")

    # -----------------------------
    # Dynamic filters
    # -----------------------------
    excluded_dynamic_filters = {"Pitch", "Handedness"}
    dynamic_cols = [c for c in finder_df.columns if c not in excluded_dynamic_filters]

    numeric_cols = []
    categorical_cols = []

    for col in dynamic_cols:
        series = finder_df[col]
        if is_numeric_column(series):
            if get_numeric_slider_bounds(series) is not None:
                numeric_cols.append(col)
        else:
            if is_reasonable_categorical(series):
                categorical_cols.append(col)

    with st.expander("Numeric filters", expanded=False):
        n_per_row = 4
        for i in range(0, len(numeric_cols), n_per_row):
            row_cols = st.columns(n_per_row)
            for j, col in enumerate(numeric_cols[i:i+n_per_row]):
                bounds = get_numeric_slider_bounds(finder_df[col])
                if bounds is None:
                    continue

                with row_cols[j]:
                    if bounds["kind"] == "int":
                        st.slider(
                            col,
                            min_value=int(bounds["min"]),
                            max_value=int(bounds["max"]),
                            value=(int(bounds["default"][0]), int(bounds["default"][1])),
                            step=int(bounds["step"]),
                            key=f"pitch_finder_num_{col}"
                        )
                    else:
                        st.slider(
                            col,
                            min_value=float(bounds["min"]),
                            max_value=float(bounds["max"]),
                            value=(float(bounds["default"][0]), float(bounds["default"][1])),
                            step=float(bounds["step"]),
                            format=bounds["format"],
                            key=f"pitch_finder_num_{col}"
                        )

    if len(categorical_cols) > 0:
        with st.expander("Additional categorical filters", expanded=False):
            c_per_row = 3
            for i in range(0, len(categorical_cols), c_per_row):
                row_cols = st.columns(c_per_row)
                for j, col in enumerate(categorical_cols[i:i+c_per_row]):
                    options = sorted(finder_df[col].dropna().unique().tolist())
                    with row_cols[j]:
                        st.multiselect(
                            col,
                            options,
                            default=options,
                            key=f"pitch_finder_cat_{col}"
                        )

    # -----------------------------
    # Apply filters
    # -----------------------------
    filtered = finder_df.copy()

    if len(selected_pitches) > 0:
        filtered = filtered[filtered["Pitch"].isin(selected_pitches)].copy()
    else:
        filtered = filtered.iloc[0:0].copy()

    if handedness_filter != "All":
        filtered = filtered[filtered["Handedness"] == handedness_filter].copy()

    for col in numeric_cols:
        key = f"pitch_finder_num_{col}"
        if key in st.session_state:
            lo, hi = st.session_state[key]
            filtered = filtered[filtered[col].between(lo, hi)]

    for col in categorical_cols:
        key = f"pitch_finder_cat_{col}"
        if key in st.session_state:
            selected = st.session_state[key]
            if len(selected) == 0:
                filtered = filtered.iloc[0:0].copy()
            else:
                filtered = filtered[filtered[col].isin(selected)]

    filtered = filtered.sort_values(sort_col, ascending=not sort_desc)

    st.caption(f"{len(filtered)} pitch rows found (minimum 25 pitches)")

    display_df = filtered.copy()
    if "Spin Eff." in display_df.columns:
        display_df["Spin Eff."] = (
            pd.to_numeric(display_df["Spin Eff."], errors="coerce")
            .mul(100)
            .round(0)
            .map(lambda x: f"{x:.0f}%" if pd.notna(x) else "")
        )

    st.dataframe(
        display_df.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )