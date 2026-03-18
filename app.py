# app.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

CSV_PATH = Path("data/player_match_data.csv")
EXCLUDE_PLAYER_FULLNAME = "Pepijn van de Merbel"

METRICS: Dict[str, Dict[str, str]] = {
    "ball_losses_per_touch_pct": {"label": "Balverlies per aanraking (%)", "y": "Balverlies per aanraking (%)"},
    "possession_lost_p90": {"label": "Balverlies per 90", "y": "Balverlies per 90"},
    "touches_p90": {"label": "Aanrakingen per 90", "y": "Aanrakingen per 90"},
    "passes_p90": {"label": "Passes per 90", "y": "Passes per 90"},
    "tackles_won_p90": {"label": "Tackles gewonnen per 90", "y": "Tackles per 90"},
    "duels_won_p90": {"label": "Duels gewonnen per 90", "y": "Duels per 90"},
    "successful_dribbles_p90": {"label": "Succesvolle dribbles per 90", "y": "Dribbles per 90"},
    "defensive_actions_p90": {"label": "Defensive actions per 90", "y": "Defensive actions per 90"},
    "key_passes_p90": {"label": "Key passes per 90", "y": "Key passes per 90"},
    "pass_accuracy_match": {"label": "Passnauwkeurigheid (%)", "y": "Passnauwkeurigheid (%)"},
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

    # Exclude keeper everywhere
    df = df[df["player_name"].str.strip() != EXCLUDE_PLAYER_FULLNAME].copy()

    return df.sort_values(["match_ts", "event_id", "display_name"]).reset_index(drop=True)


def surname_from_fullname(full_name: str) -> str:
    """
    Dutch-friendly surname extraction:
    keeps last token + preceding lowercase tussenvoegsels (van/de/der/den/te/ter/ten/het/'t).
    """
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


def init_state(players_internal: List[str]) -> None:
    if "metric_key" not in st.session_state:
        st.session_state["metric_key"] = "ball_losses_per_touch_pct"
    if "min_minutes" not in st.session_state:
        st.session_state["min_minutes"] = 45

    if "smooth" not in st.session_state:
        st.session_state["smooth"] = False
    if "smooth_window" not in st.session_state:
        st.session_state["smooth_window"] = 5
    if "smooth_mode" not in st.session_state:
        st.session_state["smooth_mode"] = "Chunk average (fewer points)"

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


def get_selected_players(players_internal: List[str]) -> List[str]:
    sel = st.session_state.get("player_selected", {})
    return [p for p in players_internal if sel.get(p, False)]


SmoothMode = Literal["Chunk average (fewer points)", "Rolling mean (same points)"]


def smooth_chunk_average(dff: pd.DataFrame, metric: str, window: int) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Downsample per player_label: average each consecutive 'window' matches -> fewer x points.
    NOTE: pandas passes a Series to agg funcs for a column, not a DataFrame.
    """
    out = dff.copy()
    out["_metric_raw"] = pd.to_numeric(out[metric], errors="coerce")

    out = out.sort_values(["player_label", "match_ts", "event_id"]).reset_index(drop=True)
    out["_match_idx"] = out.groupby("player_label").cumcount()
    out["_chunk"] = (out["_match_idx"] // window).astype(int)

    def chunk_label(series: pd.Series) -> str:
        # series contains match_label values for this chunk
        if series.empty:
            return ""
        first = str(series.iloc[0])
        last = str(series.iloc[-1])
        return last if first == last else f"{first} → {last}"

    agg = (
        out.groupby(["player_label", "_chunk"], as_index=False)
        .agg(
            chunk_end_ts=("match_ts", "max"),
            chunk_end_event=("event_id", "max"),
            chunk_label=("match_label", chunk_label),
            minutes_avg=("minutes_played", "mean"),
            metric_avg=("_metric_raw", "mean"),
        )
        .sort_values(["chunk_end_ts", "chunk_end_event", "player_label"])
        .reset_index(drop=True)
    )

    x_order = agg.sort_values(["chunk_end_ts", "chunk_end_event"])["chunk_label"].drop_duplicates().tolist()
    return agg, "metric_avg", x_order


def smooth_rolling_mean(dff: pd.DataFrame, metric: str, window: int) -> Tuple[pd.DataFrame, str]:
    out = dff.copy()
    out["_metric_raw"] = pd.to_numeric(out[metric], errors="coerce")
    out = out.sort_values(["player_label", "match_ts", "event_id"])
    out["_metric_roll"] = (
        out.groupby("player_label")["_metric_raw"]
        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    )
    return out, "_metric_roll"


def main() -> None:
    st.set_page_config(page_title="FC Den Bosch — Speler Trends", layout="wide")

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

    st.title("FC Den Bosch — Spelertrends per match (CSV)")

    df = load_data(CSV_PATH)
    if df.empty:
        st.error("CSV loaded but contains no rows (na filter).")
        st.stop()

    # Internal list (display_name unique) and UI label (surname)
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

    init_state(internal_players)

    match_order = (
        df[["match_ts", "event_id", "match_label"]]
        .dropna()
        .sort_values(["match_ts", "event_id"])["match_label"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    with st.sidebar:
        st.header("Instellingen")

        metric_labels = [METRICS[k]["label"] for k in METRICS]
        label_to_key = {METRICS[k]["label"]: k for k in METRICS}
        current_key = st.session_state["metric_key"]
        current_label = METRICS.get(current_key, METRICS["ball_losses_per_touch_pct"])["label"]

        chosen_label = st.selectbox(
            "Variabele",
            options=metric_labels,
            index=metric_labels.index(current_label) if current_label in metric_labels else 0,
        )
        st.session_state["metric_key"] = label_to_key[chosen_label]

        st.session_state["min_minutes"] = st.slider(
            "Minimum minuten",
            min_value=0,
            max_value=90,
            value=int(st.session_state["min_minutes"]),
            step=5,
        )

        st.divider()
        st.subheader("Trend weergave")

        st.session_state["smooth"] = st.checkbox("Smoothed graph", value=bool(st.session_state["smooth"]))
        st.session_state["smooth_mode"] = st.radio(
            "Smoothing type",
            options=["Chunk average (fewer points)", "Rolling mean (same points)"],
            index=0 if st.session_state["smooth_mode"].startswith("Chunk") else 1,
            disabled=not st.session_state["smooth"],
        )
        st.session_state["smooth_window"] = st.slider(
            "Window (matches)",
            min_value=3,
            max_value=11,
            value=int(st.session_state["smooth_window"]),
            step=2,
            disabled=not st.session_state["smooth"],
        )

        st.divider()
        st.subheader("Selecteer spelers")

        all_now = all(st.session_state["player_selected"].get(p, False) for p in internal_players) if internal_players else False
        all_toggle = st.checkbox("Alle spelers", value=all_now, key="all_players_checkbox")

        c1, c2 = st.columns(2)
        if c1.button("Selecteer alles", use_container_width=True):
            set_all(internal_players, True)
            st.rerun()
        if c2.button("Wis alles", use_container_width=True):
            set_all(internal_players, False)
            st.rerun()
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

    if not selected_internal:
        st.warning("Selecteer minimaal 1 speler.")
        st.stop()

    if metric_key not in df.columns:
        st.error(f"Metric kolom ontbreekt in CSV: {metric_key}")
        st.stop()

    dff = df[df["display_name"].isin(selected_internal)].copy()
    dff = dff[dff["minutes_played"] >= min_minutes].copy()
    if dff.empty:
        st.warning("Geen data na filtering (check minuten / spelers).")
        st.stop()

    dff["player_label"] = dff["display_name"].map(internal_to_label).fillna(dff["display_name"])
    dff["match_label"] = pd.Categorical(dff["match_label"], categories=match_order, ordered=True)
    dff = dff.sort_values(["match_ts", "event_id", "player_label"])

    x_col = "match_label"
    y_col = metric_key
    x_order = match_order
    title_suffix = ""

    if st.session_state["smooth"]:
        window = int(st.session_state["smooth_window"])
        mode: SmoothMode = st.session_state["smooth_mode"]

        if mode == "Chunk average (fewer points)":
            dff_plot, y_col, x_order = smooth_chunk_average(dff, metric_key, window=window)
            x_col = "chunk_label"
            title_suffix = f" — smoothed (chunk avg {window})"
        else:
            dff_plot, y_col = smooth_rolling_mean(dff, metric_key, window=window)
            title_suffix = f" — smoothed (rolling {window})"
    else:
        dff_plot = dff

    fig = px.line(
        dff_plot,
        x=x_col,
        y=y_col,
        color="player_label",
        markers=True,
        category_orders={x_col: x_order},
        hover_data={"minutes_played": True} if "minutes_played" in dff_plot.columns else None,
    )

    fig.update_layout(
        height=740,
        title=f"{METRICS[metric_key]['label']}{title_suffix}",
        xaxis_title="Match (opponent + thuis/uit) — oud → nieuw" if x_col == "match_label" else "Match chunks (old → new)",
        yaxis_title=METRICS[metric_key]["y"],
        legend_title_text="Speler",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_xaxes(tickangle=-45, showgrid=True, gridcolor="rgba(37,99,235,0.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(37,99,235,0.10)")

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
