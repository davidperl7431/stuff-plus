import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math

st.set_page_config(page_title="Perl Stuff+", layout="wide")

# -----------------------------
# Remote data URLs
# -----------------------------
PITCHER_HISTORY_URL = "https://huggingface.co/datasets/perld/stuff-plus-data/resolve/main/pitcher_history.parquet"

DF_SCORED_YEAR_URLS = {
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

def add_arm_angle_line(fig, theta_deg, *, p_throws: str, xlim=(-25, 25), ylim=(-25, 25), origin_pad=2.0):
    
    if theta_deg is None or not np.isfinite(theta_deg):
        return

    theta = float(theta_deg)
    plot_theta = 180.0 - theta if p_throws == "L" else theta
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
        "SSW_X": ("ssw_x", "mean"),
        "SSW_Z": ("ssw_z", "mean"),
        "SpinEff": ("spin_efficiency", "mean"),
    }

    if "p_throws" in d.columns:
        agg_dict["p_throws"] = ("p_throws", "first")

    g = d.groupby(["game_year", "PlayerName", "pitch_type"], as_index=False).agg(**agg_dict)

    totals = (
        g.groupby(["game_year", "PlayerName"], as_index=False)["Pitches"]
         .sum()
         .rename(columns={"Pitches": "TotalPitches"})
    )

    g = g.merge(totals, on=["game_year", "PlayerName"], how="left")
    g["Usage %"] = np.where(g["TotalPitches"] > 0, 100 * g["Pitches"] / g["TotalPitches"], np.nan)

    g = g[g["Pitches"] >= min_pitches].copy()

    if "p_throws" in g.columns:
        g["Handedness"] = np.where(g["p_throws"] == "L", "LHP", "RHP")
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
    g["SSW_X"] = g["SSW_X"].round(1)
    g["SSW_Z"] = g["SSW_Z"].round(1)
    g["Spin Eff."] = g["SpinEff"].round(3)

    g = g.rename(columns={
        "game_year": "Year",
        "PlayerName": "Pitcher",
        "pitch_type": "Pitch",
    })

    return g[[
        "Pitcher", "Handedness", "Pitch", "Pitches", "Usage %",
        "Stuff+", "Velo", "iVB", "HB", "Arm Angle", "Extension",
        "Spin", "Spin Axis", "Spin Eff.", "SSW", "SSW_X", "SSW_Z"
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
# Main-page Pitcher / Season selectors
# -----------------------------
st.title("Perl Stuff+")

all_pitchers = (
    pitcher_history["PlayerName"]
    .dropna()
    .sort_values()
    .unique()
    .tolist()
)

previous_pitcher = st.session_state.get("pitcher_name")

if len(all_pitchers) == 0:
    st.info("No pitchers available.")
    st.stop()

if previous_pitcher in all_pitchers:
    default_pitcher_index = all_pitchers.index(previous_pitcher)
else:
    default_pitcher_index = 0

top1, top2 = st.columns([2.2, 1.0])

with top1:
    pitcher_name = st.selectbox(
        "🔎 Search pitcher",
        all_pitchers,
        index=default_pitcher_index,
        key="pitcher_name",
    )

years = (
    pitcher_history.loc[pitcher_history["PlayerName"] == pitcher_name, "game_year"]
    .dropna()
    .astype(int)
    .sort_values(ascending=False)
    .unique()
    .tolist()
)

if len(years) == 0:
    st.info("No seasons available for this pitcher.")
    st.stop()

with top2:
    year = st.selectbox(
        "Season",
        years,
        index=0,
        key="season_select",
    )

# Reset leaderboard page when season changes
if st.session_state.get("last_year") != year:
    st.session_state["lb_page"] = 1
    st.session_state["last_year"] = year

df_scored = load_df_scored_year(year)
min_pitches_by_type = 25

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

        p_throws = dfp["p_throws"].iloc[0] if "p_throws" in dfp.columns and len(dfp) > 0 else "R"
        handedness = "LHP" if p_throws == "L" else "RHP"

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
    p_throws = dfp["p_throws"].iloc[0] if "p_throws" in dfp.columns and len(dfp) > 0 else "R"
    handedness = "LHP" if p_throws == "L" else "RHP"

    dfp["HB"] = pd.to_numeric(dfp["HB_obs"], errors="coerce")
    dfp["iVB"] = pd.to_numeric(dfp["iVB_obs"], errors="coerce")
    dfp["arm_angle"] = pd.to_numeric(dfp.get("arm_angle"), errors="coerce")
    dfp["ssw_in"] = pd.to_numeric(dfp.get("ssw_in"), errors="coerce")
    dfp["spin_efficiency"] = pd.to_numeric(dfp.get("spin_efficiency"), errors="coerce")

    # (2) Overview table
    # Use pitcher_history for Stuff+ so it matches the historical Stuff+ table exactly.
    # Use dfp-derived shape/velo/spin metrics for the other columns.
    ars_shape = (
        dfp.groupby("pitch_type", as_index=False)
           .agg(
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

    ars_stuff = (
        pitcher_history[
            (pitcher_history["PlayerName"] == pitcher_name) &
            (pitcher_history["game_year"] == year)
        ][["pitch_type", "Pitches", "StuffPlus"]]
        .copy()
    )

    ars = ars_stuff.merge(ars_shape, on="pitch_type", how="left")

    # Filter by min pitches per pitch type
    ars = ars[ars["Pitches"] >= min_pitches_by_type].copy()

    # Sort by pitch count
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
    
    # Overview table
    st.subheader("Overview")

    overview_disp = ars_display[[
        "Pitch", "Pitches", "Stuff+", "Velo", "iVB", "HB"
    ]].copy()

    # Force one decimal place
    for col in ["Velo", "iVB", "HB"]:
        overview_disp[col] = (
            pd.to_numeric(overview_disp[col], errors="coerce")
            .map(lambda x: f"{x:.1f}" if pd.notna(x) else "")
        )

    if "pitch_type" in ars_display.columns:
        overview_merged = ars_display.rename(columns={"pitch_type": "Pitch"})
    else:
        overview_merged = ars_display.copy()

    combined = overview_disp.merge(usage_splits, on="Pitch", how="left")
    combined = combined[[
        "Pitch", "Pitches", "Overall", "vs LHH", "vs RHH",
        "Stuff+", "Velo", "iVB", "HB"
    ]]

    st.dataframe(
        combined,
        use_container_width=True,
        hide_index=True,
    )

    arm_angle = float(dfp["arm_angle"].mean()) if dfp["arm_angle"].notna().any() else None

    st.subheader("Movement")
    if len(dfp) == 0:
        st.info("No pitches available.")
    else:
        fig = go.Figure()

        # Sample up to 100 pitches per type for display (deterministic)
        n_total = min(200, len(dfp))
        dfp_sampled = dfp.sample(n_total, random_state=42)

        # Draw dots first
        for pitch in [p for p in PITCH_ORDER if p in dfp["pitch_type"].dropna().unique()]:
            subset = dfp_sampled[dfp_sampled["pitch_type"] == pitch].copy()
            color = PITCH_COLORS.get(pitch, "white")
            fig.add_trace(go.Scatter(
                x=subset["HB"],
                y=subset["iVB"],
                mode="markers",
                name=pitch,
                marker=dict(
                    color=color,
                    size=8,
                    line=dict(width=0.5, color="black"),
                    opacity=0.8,
                ),
                customdata=subset[["release_speed", "release_spin_rate", "Stuff+_pt"]].values,
                hovertemplate=(
                    f"<b>{pitch}</b><br>"
                    "Velo: %{customdata[0]:.1f}<br>"
                    "Spin: %{customdata[1]:.0f}<br>"
                    "Stuff+: %{customdata[2]:.1f}<extra></extra>"
                ),
            ))

        # Covariance ellipses — drawn as shapes with layer="above" so they render on top
        for pitch in [p for p in PITCH_ORDER if p in dfp["pitch_type"].dropna().unique()]:
            subset = dfp[dfp["pitch_type"] == pitch][["HB", "iVB"]].dropna()
            if len(subset) < 10:
                continue
            cov = np.cov(subset["HB"], subset["iVB"])
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            a = 1.5 * np.sqrt(eigenvalues[0])
            b = 1.5 * np.sqrt(eigenvalues[1])
            hb_mean = subset["HB"].mean()
            ivb_mean = subset["iVB"].mean()
            t = np.linspace(0, 2 * np.pi, 100)
            x_ell = hb_mean + a * np.cos(t) * np.cos(angle) - b * np.sin(t) * np.sin(angle)
            y_ell = ivb_mean + a * np.cos(t) * np.sin(angle) + b * np.sin(t) * np.cos(angle)
            color = PITCH_COLORS.get(pitch, "white")
            r = int(color[1:3], 16)
            g_val = int(color[3:5], 16)
            b_val = int(color[5:7], 16)
            path_pts = [f"M {x_ell[0]:.2f},{y_ell[0]:.2f}"]
            for xi, yi in zip(x_ell[1:], y_ell[1:]):
                path_pts.append(f"L {xi:.2f},{yi:.2f}")
            path_pts.append("Z")
            path_str = " ".join(path_pts)
            fig.add_shape(
                type="path",
                path=path_str,
                line=dict(color=f"rgba({r},{g_val},{b_val},1.0)", width=2, dash="dot"),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            )

        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            legend_title_text="",
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

        for x_val in [-20, -10, 10, 20]:
            fig.add_shape(
                type="line",
                x0=x_val, x1=x_val,
                y0=-25, y1=25,
                line=dict(color="rgba(255,255,255,0.3)", width=1)
            )

        # Arm-angle reference line
        if arm_angle is not None:
            add_arm_angle_line(fig, arm_angle, p_throws=p_throws, xlim=(-25, 25), ylim=(-25, 25), origin_pad=0.8)
            fig.add_annotation(
                x=-24 if p_throws == "L" else 24,
                y=-23,
                xanchor="left" if p_throws == "L" else "right",
                yanchor="bottom",
                text=f"Arm Angle: {arm_angle:.0f}°",
                showarrow=False,
                font=dict(color="white", size=12),
                bgcolor="rgba(0,0,0,0.35)"
            )

        fig.update_layout(
                height=550,
                margin=dict(l=10, r=10, t=30, b=10),
                dragmode=False,
                xaxis_title="Horizontal Break (in)<br>1B ↔ 3B",
                yaxis_title="Induced Vertical Break (in)",
            )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "scrollZoom": False,
                "doubleClick": False,
                "showTips": False,
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
    
    MIN_PITCHES_FOR_YEAR_PITCH = 25

    hist_pitch = hist.copy()
    hist_pitch = hist_pitch[hist_pitch["Pitches"] >= MIN_PITCHES_FOR_YEAR_PITCH].copy()

    if len(hist) == 0:
        st.info("No historical pitch data available for this pitcher.")
    else:
        # -----------------------------
        # Historical Arsenal & Stuff+ (merged)
        # -----------------------------
        st.subheader("Arsenal & Stuff+")

        hist_pitch = hist[hist["Pitches"] >= MIN_PITCHES_FOR_YEAR_PITCH].copy()

        # Usage per year
        totals = (
            hist.groupby("game_year", as_index=False)["Pitches"]
            .sum()
            .rename(columns={"Pitches": "Pitches_total"})
        )
        usage_df = hist_pitch.merge(totals, on="game_year", how="left")
        usage_df["Usage"] = usage_df["Pitches"] / usage_df["Pitches_total"]

        usage_wide = usage_df.pivot(
            index="game_year", columns="pitch_type", values="Usage"
        ).fillna(0.0)

        # Stuff+ per year
        stuff_wide = hist_pitch.pivot(
            index="game_year", columns="pitch_type", values="StuffPlus"
        )

        # Overall Stuff+ per year (usage-weighted)
        overall_year = (
            hist.groupby("game_year", as_index=False)
            .apply(lambda x: pd.Series({
                "Overall": float(np.average(x["StuffPlus"], weights=x["Pitches"]))
                if x["Pitches"].sum() > 0 else np.nan
            }), include_groups=False)
            .reset_index(drop=True)
        )

        present_pitches = [p for p in PITCH_ORDER if p in hist_pitch["pitch_type"].unique()]
        all_years = sorted(hist_pitch["game_year"].unique(), reverse=True)

        # Build rows
        rows = []
        for yr in all_years:
            row_data = {"Year": int(yr)}
            # Overall
            ov = overall_year[overall_year["game_year"] == yr]["Overall"]
            row_data["Overall Stuff+"] = f"{ov.values[0]:.0f}" if len(ov) > 0 and pd.notna(ov.values[0]) else ""
            for pitch in present_pitches:
                u = usage_wide.loc[yr, pitch] if yr in usage_wide.index and pitch in usage_wide.columns else np.nan
                s = stuff_wide.loc[yr, pitch] if yr in stuff_wide.index and pitch in stuff_wide.columns else np.nan
                row_data[f"{pitch} Usage"] = f"{u*100:.1f}%" if pd.notna(u) and u > 0 else ""
                row_data[f"{pitch} Stuff+"] = f"{s:.0f}" if pd.notna(s) else ""
            rows.append(row_data)

        # Build multi-index columns
        arrays = [[""] * 2 + sum([[pitch, pitch] for pitch in present_pitches], []),
                  ["Year", "Overall Stuff+"] + sum([["Usage %", "Stuff+"] for _ in present_pitches], [])]
        
        mi = pd.MultiIndex.from_arrays(arrays)
        
        flat_cols = ["Year", "Overall Stuff+"] + [f"{p} {s}" for p in present_pitches for s in ["Usage", "Stuff+"]]
        
        merged_df = pd.DataFrame(rows)
        merged_df = merged_df[flat_cols].copy()
        merged_df.columns = mi
        
        st.dataframe(
            merged_df,
            use_container_width=True,
            hide_index=True,
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

        MIN_PITCHES_FOR_YEAR_PITCH = 10
        hist_shape.loc[
            hist_shape["Pitches"] < MIN_PITCHES_FOR_YEAR_PITCH,
            ["Velo", "iVB", "HB", "Arm Angle", "Extension", "SSW", "Spin Eff.", "Spin", "Spin Axis"]
        ] = np.nan
        
        # ---------- Historical Pitch Details ----------
        st.subheader("Shape & Spin")

        pitch_disp = hist_shape.copy()

        pitch_disp["Velo"] = pd.to_numeric(pitch_disp["Velo"], errors="coerce").round(1)
        pitch_disp["iVB"] = pd.to_numeric(pitch_disp["iVB"], errors="coerce").round(1)
        pitch_disp["HB"] = pd.to_numeric(pitch_disp["HB"], errors="coerce").round(1)
        pitch_disp["Arm Angle"] = pd.to_numeric(pitch_disp["Arm Angle"], errors="coerce").round(1)
        pitch_disp["Extension"] = pd.to_numeric(pitch_disp["Extension"], errors="coerce").round(1)
        pitch_disp["Spin"] = pd.to_numeric(pitch_disp["Spin"], errors="coerce").round(0)
        pitch_disp["Spin Axis"] = pd.to_numeric(pitch_disp["Spin Axis"], errors="coerce").round(0)
        pitch_disp["SSW"] = pd.to_numeric(pitch_disp["SSW"], errors="coerce").round(1)

        pitch_disp["Spin Eff."] = (
            pd.to_numeric(pitch_disp["Spin Eff."], errors="coerce")
            .mul(100)
            .round(0)
            .map(lambda x: f"{x:.0f}%" if pd.notna(x) else np.nan)
        )

        present_pitches = list(pitch_disp["Pitch"].dropna().unique())
        final_pitches = [p for p in PITCH_ORDER if p in present_pitches]

        for pitch in final_pitches:
            pitch_df = pitch_disp[pitch_disp["Pitch"] == pitch].copy()
            pitch_df = pitch_df.sort_values("game_year", ascending=False)
            pitch_df = pitch_df.rename(columns={"game_year": "Year"})
            pitch_df = pitch_df[[
                "Year", "Velo", "iVB", "HB", "Arm Angle", "Extension",
                "Spin", "Spin Axis", "Spin Eff.", "SSW"
            ]]

            # Skip pitches with no actual data
            if pitch_df.drop(columns=["Year"], errors="ignore").isna().all().all():
                continue

            latest = pitch_df.iloc[0]

            velo = latest["Velo"]
            ivb = latest["iVB"]
            hb = latest["HB"]

            with st.expander(pitch):

                st.dataframe(
                    pitch_df,
                    use_container_width=True,
                    hide_index=True,
                )

# -----------------------------
# LEADERBOARD TAB (with pagination)
# -----------------------------
with tab_lb:
    st.subheader("Stuff+ Leaderboard")

    # Controls
    page_size = st.selectbox("Rows per page", [10, 25, 30, 50, 100, 200], index=2)

    # -----------------------------
    # Build leaderboard from pitcher_history
    # -----------------------------

    lb_controls1, lb_controls2, lb_controls3 = st.columns([1, 1, 1])

    with lb_controls1:
        min_ip = st.number_input(
            "Min IP",
            min_value=0,
            max_value=300,
            value=162,
            step=1,
            help="Minimum innings pitched for the selected season"
        )

    with lb_controls2:
        min_pitch_for_pitchcol = st.number_input(
            "Min pitches (per pitch type)",
            min_value=1,
            max_value=200,
            value=10,
            step=10,
            help="Minimum pitches thrown for a pitch-type column to appear"
        )

    # ph_year = pitcher_history filtered to selected season
    ph_year = pitcher_history[
        pitcher_history["game_year"] == year
    ].copy()

    # --- IP filter ---
    # IP is in pitcher_history directly
    if "IP" in ph_year.columns:
        ph_year = ph_year[
            pd.to_numeric(ph_year["IP"], errors="coerce").fillna(0) >= min_ip
        ].copy()
    else:
        # fallback if pitcher_history predates IP being added
        pitcher_pitch_totals = (
            ph_year.groupby("PlayerName")["Pitches"].sum().reset_index()
        )
        eligible_pitchers = pitcher_pitch_totals[
            pitcher_pitch_totals["Pitches"] >= min_ip * 15
        ]["PlayerName"]
        ph_year = ph_year[ph_year["PlayerName"].isin(eligible_pitchers)].copy()

    # Overall Stuff+ from pitcher_history (arsenal pitches only)
    # Filter to arsenal pitches using same thresholds as profile page
    ph_arsenal = ph_year[ph_year["Pitches"] >= 100].copy()

    # Compute total pitches per pitcher (across arsenal pitches only)
    pitcher_totals = (
        ph_arsenal.groupby("PlayerName")["Pitches"]
        .sum()
        .reset_index()
        .rename(columns={"Pitches": "Pitches_total"})
    )
    ph_arsenal = ph_arsenal.merge(pitcher_totals, on="PlayerName", how="left")
    ph_arsenal["Usage"] = ph_arsenal["Pitches"] / ph_arsenal["Pitches_total"]

    # Weighted overall Stuff+ — usage-weighted to match profile page
    overall = (
        ph_arsenal.groupby("PlayerName", as_index=False)
        .apply(lambda x: pd.Series({
            "Overall": (
                (x["StuffPlus"] * x["Usage"]).sum() / x["Usage"].sum()
                if x["Usage"].sum() > 0 else np.nan
            ),
            "Pitches": int(ph_year[ph_year["PlayerName"] == x.name]["Pitches"].sum())
            if x.name in ph_year["PlayerName"].values else 0,
        }))
        .reset_index(drop=True)
    )

    # Add IP column if available
    if "IP" in ph_year.columns:
        ip_col = (
            ph_year[["PlayerName", "IP"]]
            .drop_duplicates(subset=["PlayerName"])
        )
        overall = overall.merge(ip_col, on="PlayerName", how="left")

    # Wide pivot for per-pitch Stuff+ columns
    # Only show a pitch column if pitcher threw >= min_pitch_for_pitchcol
    g_pitch = ph_year.copy()
    g_pitch.loc[
        g_pitch["Pitches"] < min_pitch_for_pitchcol, "StuffPlus"
    ] = np.nan
    g_pitch = g_pitch.rename(columns={"pitch_type": "Pitch"})

    wide = g_pitch.pivot_table(
        index="PlayerName",
        columns="Pitch",
        values="StuffPlus",
        aggfunc="mean"
    )

    pitch_cols_sorted = (
        [p for p in PITCH_ORDER if p in wide.columns]
        + sorted([c for c in wide.columns if c not in PITCH_ORDER])
    )
    wide = wide.reindex(columns=pitch_cols_sorted)

    # Combine overall + per-pitch columns
    lb = overall.merge(wide, left_on="PlayerName", right_index=True, how="left")

    # Round all Stuff+ columns
    stuff_cols = ["Overall"] + pitch_cols_sorted
    for c in stuff_cols:
        if c in lb.columns:
            lb[c] = pd.to_numeric(lb[c], errors="coerce").round(0)

    # Sort
    lb = lb.sort_values(["Overall", "Pitches"], ascending=[False, False])

    # Columns to show — include IP if available
    show_cols = ["PlayerName", "Overall", "Pitches"]
    if "IP" in lb.columns:
        lb["IP"] = pd.to_numeric(lb["IP"], errors="coerce").round(1)
        show_cols.append("IP")
    show_cols += pitch_cols_sorted

    # --- Pagination ---
    total_rows = len(lb)
    total_pages = max(1, int(np.ceil(total_rows / page_size)))

    # Prev / Next buttons instead of number input
    lb_page_key = "lb_page"
    if lb_page_key not in st.session_state:
        st.session_state[lb_page_key] = 1

    page = st.session_state[lb_page_key]
    if page > total_pages:
        st.session_state[lb_page_key] = 1
        page = 1

    col_prev, col_page_info, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button("← Prev", disabled=(page <= 1)):
            st.session_state[lb_page_key] = page - 1
            st.rerun()
    with col_page_info:
        start = (page - 1) * page_size
        end = min(start + page_size, total_rows)
        st.caption(
            f"Showing {start+1}–{end} of {total_rows} "
            f"(Page {page} of {total_pages})"
        )
    with col_next:
        if st.button("Next →", disabled=(page >= total_pages)):
            st.session_state[lb_page_key] = page + 1
            st.rerun()

    start = (page - 1) * page_size
    end = start + page_size

    display_lb = lb[show_cols].iloc[start:end].reset_index(drop=True).copy()
    display_lb = display_lb.replace({None: np.nan})

    for c in display_lb.columns:
        if c not in ("PlayerName", "IP"):
            display_lb[c] = (
                pd.to_numeric(display_lb[c], errors="coerce")
                .map(lambda x: f"{x:.0f}" if pd.notna(x) else "")
            )

    st.dataframe(
        display_lb,
        use_container_width=True,
        hide_index=True,
    )
    st.divider()

    st.subheader("Pitch Usage")

    # Build usage from pitcher_history (same source as leaderboard)
    ph_usage = ph_year.copy().rename(columns={"pitch_type": "Pitch"})
    pitcher_totals_u = (
        ph_usage.groupby("PlayerName")["Pitches"]
        .sum()
        .reset_index()
        .rename(columns={"Pitches": "Pitches_total"})
    )
    ph_usage = ph_usage.merge(pitcher_totals_u, on="PlayerName", how="left")
    ph_usage["Usage %"] = ph_usage["Pitches"] / ph_usage["Pitches_total"]

    wide_u = ph_usage.pivot_table(
        index="PlayerName",
        columns="Pitch",
        values="Usage %",
        aggfunc="mean"
    )
    wide_u = wide_u.rename(columns=lambda c: f"{c} %")

    pitcher_totals_u2 = (
        ph_usage.groupby("PlayerName")["Pitches"]
        .sum()
        .reset_index()
        .rename(columns={"Pitches": "Pitches_total"})
    )

    usage_lb = pitcher_totals_u2.merge(
        wide_u, left_on="PlayerName", right_index=True, how="left"
    ).rename(columns={"Pitches_total": "Pitches"})

    usage_lb = usage_lb.sort_values("Pitches", ascending=False)

    usage_disp = usage_lb.copy()
    pitch_cols_u = sorted([
        c for c in usage_disp.columns
        if c not in ("PlayerName", "Pitches")
    ])

    for c in pitch_cols_u:
        usage_disp[c] = (
            pd.to_numeric(usage_disp[c], errors="coerce")
            .mul(100)
            .round(1)
            .map(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
        )

    # Pagination — shares lb_page state with leaderboard
    total_rows_u = len(usage_disp)
    start = (page - 1) * page_size
    end = min(start + page_size, total_rows_u)

    st.dataframe(
        usage_disp[["PlayerName", "Pitches"] + pitch_cols_u]
        .iloc[start:end]
        .reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    st.caption("Usage values are percentages within each pitcher-season.")

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