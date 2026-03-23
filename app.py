# app.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

CSV_PATH = Path("data/player_match_data.csv")
EXCLUDE_PLAYER_FULLNAME = "Pepijn van de Merbel"

METRICS: Dict[str, Dict[str, str]] = {
    "air_duels": {"label": "Aerial duels", "y": "Aerial duels"},
    "air_duels_p90": {"label": "Aerial duels per 90", "y": "Aerial duels per 90"},
    "air_duels_won": {"label": "Aerial duels won", "y": "Aerial duels won"},
    "air_duels_won_p90": {"label": "Aerial duels won per 90", "y": "Aerial duels won per 90"},
    "ball_losses_per_touch_pct": {"label": "Ball losses per touch (%)", "y": "Ball losses per touch (%)"},
    "crosses_accurate": {"label": "Accurate crosses", "y": "Accurate crosses"},
    "crosses_accurate_p90": {"label": "Accurate crosses per 90", "y": "Accurate crosses per 90"},
    "crosses_accuracy_pct": {"label": "Cross accuracy (%)", "y": "Cross accuracy (%)"},
    "crosses_total": {"label": "Crosses attempted", "y": "Crosses attempted"},
    "crosses_total_p90": {"label": "Crosses attempted per 90", "y": "Crosses attempted per 90"},
    "defensive_actions_p90": {"label": "Defensive actions per 90", "y": "Defensive actions per 90"},
    "duels_total": {"label": "Total duels", "y": "Total duels"},
    "duels_total_p90": {"label": "Total duels per 90", "y": "Total duels per 90"},
    "duels_won_p90": {"label": "Duels won per 90", "y": "Duels won per 90"},
    "ground_duels": {"label": "Ground duels", "y": "Ground duels"},
    "ground_duels_p90": {"label": "Ground duels per 90", "y": "Ground duels per 90"},
    "ground_duels_won": {"label": "Ground duels won", "y": "Ground duels won"},
    "ground_duels_won_p90": {"label": "Ground duels won per 90", "y": "Ground duels won per 90"},
    "interceptions": {"label": "Interceptions", "y": "Interceptions"},
    "interceptions_p90": {"label": "Interceptions per 90", "y": "Interceptions per 90"},
    "key_passes_p90": {"label": "Key passes per 90", "y": "Key passes per 90"},
    "pass_accuracy_match": {"label": "Pass accuracy (%)", "y": "Pass accuracy (%)"},
    "passes_p90": {"label": "Passes per 90", "y": "Passes per 90"},
    "possession_lost_p90": {"label": "Possessions lost per 90", "y": "Possessions lost per 90"},
    "shots_on_target": {"label": "Shots on target", "y": "Shots on target"},
    "shots_on_target_p90": {"label": "Shots on target per 90", "y": "Shots on target per 90"},
    "shots_total": {"label": "Shots", "y": "Shots"},
    "shots_total_p90": {"label": "Shots per 90", "y": "Shots per 90"},
    "successful_dribbles_p90": {"label": "Successful dribbles per 90", "y": "Successful dribbles per 90"},
    "tackles_won_p90": {"label": "Tackles won per 90", "y": "Tackles won per 90"},
    "touches_p90": {"label": "Touches per 90", "y": "Touches per 90"},
}


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    required = {"match_ts", "event_id", "match_label", "display_name", "player_name", "minutes_played"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    df["match_ts"] = pd.to_numeric(df["match_ts"], errors="coerce").fillna(0).astype(int)
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").fillna(0).astype(int)
    df["minutes_played"] = pd.to_numeric(df["minutes_played"], errors="coerce").fillna(0).astype(float)
    df["match_label"] = df["match_label"].astype(str)
    df["display_name"] = df["display_name"].astype(str)
    df["player_name"] = df["player_name"].astype(str)

    df = df[df["player_name"].str.strip() != EXCLUDE_PLAYER_FULLNAME].copy()
    return df.sort_values(["match_ts", "event_id", "display_name"]).reset_index(drop=True)


def surname_from_fullname(full_name: str) -> str:
    name = (full_name or "").strip()
    if not name:
        return ""
    parts = [p for p in name.split() if p.strip()]
    if len(parts) == 1:
        return parts[0]

    prefixes = {"van", "de", "der", "den", "te", "ter", "ten", "het", "'t", "v/d", "v.d.", "v/d."}
    j = len(parts) - 1
    while j - 1 >= 0:
        prev = parts[j - 1]
        prev_l = prev.lower()
        if prev_l in prefixes or prev.islower():
            j -= 1
            continue
        break
    return " ".join(parts[j:])


def init_state(players_internal: List[str], default_metric_key: str) -> None:
    if "metric_key" not in st.session_state:
        st.session_state["metric_key"] = default_metric_key
    if "min_minutes" not in st.session_state:
        st.session_state["min_minutes"] = 45

    if "smooth" not in st.session_state:
        st.session_state["smooth"] = True
    if "smooth_window" not in st.session_state:
        st.session_state["smooth_window"] = 3

    if "player_selected" not in st.session_state:
        st.session_state["player_selected"] = {p: True for p in players_internal}

    existing = st.session_state["player_selected"]
    for p in players_internal:
        existing.setdefault(p, True)
    for p in list(existing.keys()):
        if p not in players_internal:
            existing.pop(p, None)


def set_all(players_internal: List[str], value: bool) -> None:
    st.session_state["player_selected"] = {p: value for p in players_internal}
    for p in players_internal:
        st.session_state[f"p__{p}"] = value


def get_selected_players(players_internal: List[str]) -> List[str]:
    sel = st.session_state.get("player_selected", {})
    return [p for p in players_internal if sel.get(p, False)]


def build_player_anchors(player_df: pd.DataFrame, metric_col: str, window: int) -> pd.DataFrame:
    """
    Anchor points per player:
      - x=match 1: avg(first 2 matches) (or first 1 if only one match) -> marker ON
      - x=window, 2*window, ...: avg of each full window chunk -> marker ON
      - x=last match if remainder exists: avg(remainder) -> marker OFF
    Output columns: match_label, y, marker_size
    """
    p = player_df.sort_values(["match_ts", "event_id"]).reset_index(drop=True)
    n = len(p)
    if n == 0:
        return pd.DataFrame(columns=["match_label", "y", "marker_size"])

    vals = pd.to_numeric(p[metric_col], errors="coerce").to_numpy()

    def avg_slice(a: np.ndarray) -> float:
        a = a[~np.isnan(a)]
        return float(np.mean(a)) if a.size else np.nan

    anchors: List[Tuple[str, float, int]] = []

    start_avg = avg_slice(vals[:2]) if n >= 2 else avg_slice(vals[:1])
    anchors.append((p.loc[0, "match_label"], start_avg, 8))

    full_chunks = n // window
    for chunk_idx in range(full_chunks):
        start = chunk_idx * window
        end = start + window
        y = avg_slice(vals[start:end])
        x_label = p.loc[end - 1, "match_label"]
        anchors.append((x_label, y, 8))

    rem = n % window
    if rem != 0:
        start = full_chunks * window
        y = avg_slice(vals[start:n])
        x_label = p.loc[n - 1, "match_label"]
        anchors.append((x_label, y, 0))

    out = pd.DataFrame(anchors, columns=["match_label", "y", "marker_size"])
    out = out.drop_duplicates(subset=["match_label"], keep="last").reset_index(drop=True)
    return out


def get_metric_options(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str], str]:
    available_keys = [k for k in METRICS.keys() if k in df.columns]
    if not available_keys:
        raise ValueError("No metric columns found in CSV that match METRICS.")

    pairs = [(METRICS[k]["label"], k) for k in available_keys]
    pairs.sort(key=lambda x: x[0].casefold())

    metric_labels = [p[0] for p in pairs]
    label_to_key = {label: key for label, key in pairs}
    default_key = pairs[0][1]
    return metric_labels, label_to_key, default_key


def main() -> None:
    st.set_page_config(page_title="FC Den Bosch — Player Trends", layout="wide")

    st.markdown(
        """
        <style>
          section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
          section[data-testid="stSidebar"] label { font-size: 0.95rem; }
          section[data-testid="stSidebar"] [data-testid="stCheckbox"] { margin-bottom: -6px; }
          input[type="checkbox"] { accent-color: #2563eb; }
          input[type="range"] { accent-color: #2563eb; }
          section[data-testid="stSidebar"] div[data-baseweb="select"] > div { border-radius: 10px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("FC Den Bosch — 2025/2026 Player trends")

    df = load_data(CSV_PATH)
    if df.empty:
        st.error("CSV loaded but contains no rows (after filtering).")
        st.stop()

    metric_labels, label_to_key, default_metric_key = get_metric_options(df)

    player_meta = (
        df[["display_name", "player_name"]]
        .dropna()
        .drop_duplicates()
        .sort_values("display_name")
        .reset_index(drop=True)
    )
    player_meta["surname"] = player_meta["player_name"].map(surname_from_fullname)

    counts = player_meta["surname"].value_counts()
    dup = set(counts[counts > 1].index.tolist())
    labels: List[str] = []
    seen: Dict[str, int] = {}
    for s in player_meta["surname"].tolist():
        if s in dup:
            seen[s] = seen.get(s, 0) + 1
            labels.append(f"{s} ({seen[s]})")
        else:
            labels.append(s)
    player_meta["surname_label"] = labels

    internal_players = player_meta["display_name"].tolist()
    internal_to_label = dict(zip(player_meta["display_name"], player_meta["surname_label"]))

    init_state(internal_players, default_metric_key=default_metric_key)

    match_order = (
        df[["match_ts", "event_id", "match_label"]]
        .dropna()
        .sort_values(["match_ts", "event_id"])["match_label"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    with st.sidebar:
        st.header("Settings")

        current_key = st.session_state.get("metric_key", default_metric_key)
        if current_key not in df.columns or current_key not in METRICS:
            current_key = default_metric_key
            st.session_state["metric_key"] = current_key

        current_label = METRICS[current_key]["label"]
        chosen_label = st.selectbox(
            "Metric",
            options=metric_labels,
            index=metric_labels.index(current_label) if current_label in metric_labels else 0,
        )
        st.session_state["metric_key"] = label_to_key[chosen_label]

        st.session_state["min_minutes"] = st.slider(
            "Minimum minutes played",
            min_value=0,
            max_value=90,
            value=int(st.session_state["min_minutes"]),
            step=5,
        )

        st.divider()
        st.subheader("Trend display")

        st.session_state["smooth"] = st.checkbox("Smoothed graph", value=bool(st.session_state["smooth"]))
        st.session_state["smooth_window"] = st.slider(
            "Window (matches)",
            min_value=2,
            max_value=16,
            value=max(2, min(16, int(st.session_state["smooth_window"]))),
            step=1,
            disabled=not st.session_state["smooth"],
        )

        st.divider()
        st.subheader("Select players")

        all_now = (
            all(st.session_state["player_selected"].get(p, False) for p in internal_players)
            if internal_players
            else False
        )
        all_toggle = st.checkbox("All players", value=all_now, key="all_players_checkbox")
        if all_toggle != all_now:
            set_all(internal_players, all_toggle)
            st.rerun()

        col_a, col_b = st.columns(2)
        half = (len(internal_players) + 1) // 2

        def _player_cb(container, internal: str) -> None:
            label = internal_to_label.get(internal, internal)
            key = f"p__{internal}"
            if key not in st.session_state:
                st.session_state[key] = bool(st.session_state["player_selected"].get(internal, True))
            val = container.checkbox(label, key=key, value=st.session_state[key])
            st.session_state["player_selected"][internal] = bool(val)

        for p in internal_players[:half]:
            _player_cb(col_a, p)
        for p in internal_players[half:]:
            _player_cb(col_b, p)

    metric_key = st.session_state["metric_key"]
    min_minutes = float(st.session_state["min_minutes"])
    selected_internal = get_selected_players(internal_players)

    selected_labels = [internal_to_label.get(p, p) for p in selected_internal]
    single_player_label = selected_labels[0] if len(selected_labels) == 1 else None

    if not selected_internal:
        st.warning("Select at least 1 player.")
        st.stop()

    if metric_key not in df.columns:
        st.error(f"Metric column missing in CSV: {metric_key}")
        st.stop()

    dff = df[df["display_name"].isin(selected_internal)].copy()
    dff = dff[dff["minutes_played"] >= min_minutes].copy()
    if dff.empty:
        st.warning("No data after filtering (check minutes / players).")
        st.stop()

    dff["player_label"] = dff["display_name"].map(internal_to_label).fillna(dff["display_name"])
    dff["match_label"] = pd.Categorical(dff["match_label"], categories=match_order, ordered=True)
    dff = dff.sort_values(["match_ts", "event_id", "player_label"])

    fig = go.Figure()

    if st.session_state["smooth"]:
        window = int(st.session_state["smooth_window"])
        title_suffix = f" — smoothed (window={window})"

        for player_label, g in dff.groupby("player_label", sort=True):
            anchors = build_player_anchors(g, metric_key, window=window)
            if anchors.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=anchors["match_label"],
                    y=anchors["y"],
                    mode="lines+markers",
                    name=player_label,
                    marker=dict(size=anchors["marker_size"]),
                    connectgaps=False,
                )
            )
    else:
        title_suffix = ""
        for player_label, g in dff.groupby("player_label", sort=True):
            fig.add_trace(
                go.Scatter(
                    x=g["match_label"],
                    y=g[metric_key],
                    mode="lines+markers",
                    name=player_label,
                    marker=dict(size=6),
                    connectgaps=False,
                )
            )

    fig.update_layout(
        height=740,
        title=f"{METRICS[metric_key]['label']}{title_suffix}",
        xaxis_title="Match",
        yaxis_title=METRICS[metric_key]["y"],
        legend_title_text="Player",
        showlegend=True,  # <-- force legend even if only 1 trace

        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=match_order,
        tickangle=-45,
        showgrid=True,
        gridcolor="rgba(37,99,235,0.12)",
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(37,99,235,0.10)")

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
