# app.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

CSV_PATH = Path("data/player_match_data.csv")

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

    required = {"match_ts", "event_id", "match_label", "display_name", "minutes_played"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    df["match_ts"] = pd.to_numeric(df["match_ts"], errors="coerce").fillna(0).astype(int)
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").fillna(0).astype(int)
    df["minutes_played"] = pd.to_numeric(df["minutes_played"], errors="coerce").fillna(0).astype(float)
    df["match_label"] = df["match_label"].astype(str)
    df["display_name"] = df["display_name"].astype(str)

    return df.sort_values(["match_ts", "event_id", "display_name"]).reset_index(drop=True)


def init_state(all_players: List[str]) -> None:
    if "metric_key" not in st.session_state:
        st.session_state["metric_key"] = "ball_losses_per_touch_pct"
    if "selected_players" not in st.session_state:
        st.session_state["selected_players"] = all_players[:]  # default all selected


def main() -> None:
    st.set_page_config(page_title="FC Den Bosch — Speler Trends", layout="wide")
    st.title("FC Den Bosch — Spelertrends per match (CSV)")

    df = load_data(CSV_PATH)
    if df.empty:
        st.error("CSV loaded but contains no rows.")
        st.stop()

    all_players = (
        df["display_name"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    init_state(all_players)

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

        min_minutes = st.slider(
            "Minimum minuten (match wordt getoond als speler ≥ deze minuten speelde)",
            min_value=0,
            max_value=90,
            value=45,
            step=5,
        )

        c1, c2 = st.columns(2)
        if c1.button("Selecteer alles", use_container_width=True):
            st.session_state["selected_players"] = all_players[:]
        if c2.button("Wis alles", use_container_width=True):
            st.session_state["selected_players"] = []

        st.multiselect(
            "Selecteer spelers",
            options=all_players,
            key="selected_players",
        )

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

    dff = df[df["display_name"].isin(selected_players)].copy()
    dff = dff[dff["minutes_played"] >= float(min_minutes)].copy()

    if dff.empty:
        st.warning("Geen data na filtering (check minuten slider / spelers).")
        st.stop()

    dff["match_label"] = pd.Categorical(dff["match_label"], categories=match_order, ordered=True)
    dff = dff.sort_values(["match_ts", "event_id", "display_name"])

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
        cols = ["match_date_utc", "match_label", "display_name", "minutes_played", metric_key]
        cols = [c for c in cols if c in dff.columns]
        st.dataframe(dff[cols], use_container_width=True)


if __name__ == "__main__":
    main()
