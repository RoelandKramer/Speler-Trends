# app.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

CSV_PATH = Path("data/player_match_data.csv")

# CSV columns we expect:
# - match_ts (int), match_label (str), display_name (str), minutes_played (float)
# - per90 metrics and ball_losses_per_touch_pct


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

    # Normalize types
    df["match_ts"] = pd.to_numeric(df.get("match_ts"), errors="coerce").fillna(0).astype(int)
    df["minutes_played"] = pd.to_numeric(df.get("minutes_played"), errors="coerce").fillna(0).astype(float)
    df["match_label"] = df.get("match_label", "").astype(str)
    df["display_name"] = df.get("display_name", "").astype(str)

    # Stable ordering (oldest -> newest)
    df = df.sort_values(["match_ts", "event_id", "display_name"]).reset_index(drop=True)
    return df


def _init_state(all_players: List[str]) -> None:
    if "metric_key" not in st.session_state:
        st.session_state["metric_key"] = "ball_losses_per_touch_pct"
    if "all_toggle" not in st.session_state:
        st.session_state["all_toggle"] = True
    if "selected_players" not in st.session_state:
        st.session_state["selected_players"] = all_players[:]  # default: all selected


def _set_all_players(all_players: List[str]) -> None:
    if st.session_state.get("all_toggle", True):
        st.session_state["selected_players"] = all_players[:]
    else:
        st.session_state["selected_players"] = []


def _sync_all_toggle(all_players: List[str]) -> None:
    sel = st.session_state.get("selected_players", [])
    st.session_state["all_toggle"] = len(all_players) > 0 and len(sel) == len(all_players)


def main() -> None:
    st.set_page_config(page_title="FC Den Bosch — Speler Trends", layout="wide")
    st.title("FC Den Bosch — Spelertrends per match (CSV)")

    df = load_data(CSV_PATH)
    if df.empty:
        st.error("CSV loaded but contains no rows.")
        st.stop()

    # Players (checklist behavior via multiselect + select-all toggle)
    all_players = (
        df["display_name"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    _init_state(all_players)

    # Match order for x-axis: oldest -> newest
    match_order = (
        df[["match_ts", "event_id", "match_label"]]
        .dropna()
        .sort_values(["match_ts", "event_id"])["match_label"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    # Sidebar UI
    with st.sidebar:
        st.header("Instellingen")

        # Metric dropdown
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

        # Minutes slider (default 45)
        min_minutes = st.slider(
            "Minimum minuten (match wordt getoond als speler ≥ deze minuten speelde)",
            min_value=0,
            max_value=90,
            value=45,
            step=5,
        )

        # Select/deselect all
        st.checkbox(
            "Alle spelers",
            key="all_toggle",
            value=st.session_state.get("all_toggle", True),
            on_change=_set_all_players,
            args=(all_players,),
        )

        # Player selection
        st.multiselect(
            "Selecteer spelers",
            options=all_players,
            default=st.session_state.get("selected_players", all_players),
            key="selected_players",
        )
        _sync_all_toggle(all_players)

        if st.button("Herlaad CSV"):
            load_data.clear()
            st.rerun()

        st.caption(f"Rows in CSV: {len(df)}")

    metric_key = st.session_state["metric_key"]
    selected_players = st.session_state.get("selected_players", [])

    if not selected_players:
        st.warning("Selecteer minimaal 1 speler.")
        st.stop()

    if metric_key not in df.columns:
        st.error(f"Metric kolom ontbreekt in CSV: {metric_key}")
        st.stop()

    # Filter by players + minutes
    dff = df[df["display_name"].isin(selected_players)].copy()
    dff = dff[dff["minutes_played"] >= float(min_minutes)].copy()

    if dff.empty:
        st.warning("Geen data na filtering (check minuten slider / spelers).")
        st.stop()

    # Enforce x-axis order
    dff["match_label"] = pd.Categorical(dff["match_label"], categories=match_order, ordered=True)
    dff = dff.sort_values(["match_ts", "event_id", "display_name"])

    # Plot
    fig = px.line(
        dff,
        x="match_label",
        y=metric_key,
        color="display_name",
        markers=True,
        category_orders={"match_label": match_order},
        hover_data={
            "display_name": True,
            "match_label": True,
            "minutes_played": True,
            "passes_p90": "passes_p90" in dff.columns,
            "possession_lost_p90": "possession_lost_p90" in dff.columns,
            "touches_p90": "touches_p90" in dff.columns,
            "pass_accuracy_match": "pass_accuracy_match" in dff.columns,
        },
    )

    fig.update_layout(
        height=720,
        xaxis_title="Match (opponent + thuis/uit) — oud → nieuw",
        yaxis_title=METRICS[metric_key]["y"],
        legend_title_text="Speler",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    fig.update_xaxes(tickangle=-45)

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Data (gefilterd)"):
        show_cols = ["match_date_utc", "match_label", "display_name", "minutes_played", metric_key]
        show_cols = [c for c in show_cols if c in dff.columns]
        st.dataframe(dff[show_cols], use_container_width=True)


if __name__ == "__main__":
    main()
