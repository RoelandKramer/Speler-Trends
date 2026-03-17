# app.py
"""
Streamlit app: FC Den Bosch player trends per match (>=45 min) using SofaScore API scraping.

How to run locally:
  pip install -r requirements.txt
  streamlit run app.py

Recommended requirements.txt:
  streamlit
  pandas
  numpy
  requests
  plotly
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# =========================
# CONFIG
# =========================

TARGET_TEAM_NAME = "FC Den Bosch"
MIN_MATCH_MINUTES = 45

# Add future matches here (event ids). App will automatically extend.
MATCH_IDS: List[int] = [
    14056658, 14056654, 14056625, 14056580, 14056549, 15392911, 14056516,
    14056489, 14056348, 14056672, 14056661, 14056627, 14056612, 14056577,
    14056544, 14056508, 14056502, 14056467, 14056408, 14751816, 14056450,
    14056447, 14056422, 14056419, 14056402, 14056383, 14056366, 14056357,
    14056321, 14056430, 14056344, 14056692,
]

# =========================
# HTTP + STAT HELPERS
# =========================


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "nl-NL,nl;q=0.9,en;q=0.8",
            "Referer": "https://www.sofascore.com/",
            "Origin": "https://www.sofascore.com",
        }
    )
    return s


def _get_json(path: str, *, retries: int = 4, timeout: float = 20.0) -> Dict[str, Any]:
    bases = ("https://api.sofascore.com", "https://www.sofascore.com")
    last: Optional[Exception] = None

    for base in bases:
        url = base.rstrip("/") + path
        s = _make_session()
        for attempt in range(retries):
            try:
                r = s.get(url, timeout=timeout)
                if r.status_code == 429:
                    time.sleep(1.25 * (attempt + 1) + random.random())
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last = e
                time.sleep(0.6 * (attempt + 1) + random.random() * 0.4)

    raise RuntimeError(f"GET failed {path}: {last}") from last


def _to_num(x: Any) -> float:
    if x is None or isinstance(x, bool):
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        x = x.strip().replace("%", "").replace(",", ".")
        try:
            return float(x)
        except Exception:
            return 0.0
    return 0.0


def _normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _get_stat_by_alias(stats: Dict[str, Any], aliases: Iterable[str], default: float = 0.0) -> float:
    if not stats:
        return default
    norm_map = {_normalize_key(k): v for k, v in stats.items()}
    for alias in aliases:
        val = norm_map.get(_normalize_key(alias))
        if val is not None:
            return _to_num(val)
    return default


def _calc_defensive_actions(stats: Dict[str, Any]) -> float:
    won_tackle = _get_stat_by_alias(stats, ["wonTackle", "tacklesWon", "successfulTackles"])
    interceptions = _get_stat_by_alias(stats, ["interceptionWon", "interceptions", "totalInterceptions"])
    clearances = _get_stat_by_alias(stats, ["totalClearance", "clearances"])
    blocked_shots = _get_stat_by_alias(stats, ["blockedScoringAttempt", "blockedShots"])
    recoveries = _get_stat_by_alias(stats, ["ballRecovery", "recoveries"])
    return won_tackle + interceptions + clearances + blocked_shots + recoveries


# =========================
# MATCH META
# =========================


@dataclass(frozen=True)
class MatchMeta:
    event_id: int
    start_ts: int
    date_utc: datetime
    is_home: bool
    opponent: str
    label_nl: str  # "Jong Ajax. Thuis"


def _get_match_meta(event_id: int, target_team_name: str) -> MatchMeta:
    data = _get_json(f"/api/v1/event/{event_id}")
    event = data.get("event", {}) or {}

    start_ts = int(event.get("startTimestamp") or 0)
    date_utc = datetime.fromtimestamp(start_ts, tz=timezone.utc) if start_ts else datetime(1970, 1, 1, tzinfo=timezone.utc)

    home = (event.get("homeTeam") or {}).get("name", "") or ""
    away = (event.get("awayTeam") or {}).get("name", "") or ""

    is_home = target_team_name.lower() in home.lower()
    opponent = away if is_home else home
    label_nl = f"{opponent}. {'Thuis' if is_home else 'Uit'}"

    return MatchMeta(
        event_id=event_id,
        start_ts=start_ts,
        date_utc=date_utc,
        is_home=is_home,
        opponent=opponent,
        label_nl=label_nl,
    )


# =========================
# PLAYER FETCH
# =========================


def _fetch_match_stats(event_id: int, side: str) -> List[Dict[str, Any]]:
    lineups = _get_json(f"/api/v1/event/{event_id}/lineups")
    incidents = _get_json(f"/api/v1/event/{event_id}/incidents")

    team_data = lineups.get(side) or {}
    players: List[Dict[str, Any]] = team_data.get("players") or []

    starters = [it for it in players if it.get("substitute") is False][:11]
    starter_ids = {
        p.get("player", {}).get("id")
        for p in starters
        if p.get("player", {}).get("id")
    }

    sub_in_ids = set()
    for inc in incidents.get("incidents") or []:
        if str(inc.get("incidentType")).lower() != "substitution":
            continue
        pin = (inc.get("playerIn") or {}).get("id")
        pout = (inc.get("playerOut") or {}).get("id")
        if isinstance(pin, int) and isinstance(pout, int) and pout in starter_ids:
            sub_in_ids.add(pin)

    played_ids = starter_ids | sub_in_ids

    results: List[Dict[str, Any]] = []
    for p_item in players:
        pid = p_item.get("player", {}).get("id")
        if pid in played_ids:
            try:
                stats_payload = _get_json(f"/api/v1/event/{event_id}/player/{pid}/statistics")
                stats = stats_payload.get("statistics", {}) or {}
            except Exception:
                stats = {}

            p_item = dict(p_item)
            p_item["fetched_statistics"] = stats
            results.append(p_item)

    return results


def _get_target_side(event_id: int, target_team_name: str) -> str:
    data = _get_json(f"/api/v1/event/{event_id}")
    event = data.get("event", {}) or {}
    home_name = (event.get("homeTeam") or {}).get("name", "") or ""
    return "home" if target_team_name.lower() in home_name.lower() else "away"


def get_player_match_data(event_id: int, meta: MatchMeta) -> pd.DataFrame:
    side = "home" if meta.is_home else "away"
    items = _fetch_match_stats(event_id, side)

    rows: List[Dict[str, Any]] = []
    for it in items:
        p = it.get("player") or {}
        stats = it.get("fetched_statistics") or {}

        short_name = p.get("shortName") or p.get("name") or "Unknown"
        jersey_num = (
            it.get("jerseyNumber")
            or it.get("shirtNumber")
            or p.get("jerseyNumber")
            or p.get("shirtNumber")
            or "?"
        )

        minutes_played = _get_stat_by_alias(stats, ["minutesPlayed", "minutes"])

        total_pass = _get_stat_by_alias(stats, ["totalPass", "passes"])
        accurate_pass = _get_stat_by_alias(stats, ["accuratePass", "successfulPasses"])

        tackles_won = _get_stat_by_alias(stats, ["wonTackle", "tacklesWon", "successfulTackles"])
        duels_won = _get_stat_by_alias(stats, ["duelWon", "duelsWon", "wonDuels"])
        successful_dribbles = _get_stat_by_alias(stats, ["wonContest", "successfulDribbles", "dribblesSucceeded"])
        key_passes = _get_stat_by_alias(stats, ["keyPass", "keyPasses"])
        possession_lost = _get_stat_by_alias(stats, ["possessionLostCtrl", "possessionLost", "ballPossessionLost"])
        defensive_actions = _calc_defensive_actions(stats)

        # Touches / Aanrakingen (aliases vary; best-effort)
        touches = _get_stat_by_alias(
            stats,
            [
                "touches",
                "totalTouches",
                "ballTouches",
                "touch",
                "totalTouch",
            ],
            default=np.nan,
        )

        pass_accuracy_match = np.nan
        if total_pass > 0:
            pass_accuracy_match = (accurate_pass / total_pass) * 100.0

        ball_losses_per_touch = np.nan
        if touches and not np.isnan(touches) and touches > 0:
            ball_losses_per_touch = (possession_lost / touches) * 100.0

        rows.append(
            {
                "event_id": event_id,
                "match_ts": meta.start_ts,
                "match_date_utc": meta.date_utc,
                "match_label": meta.label_nl,
                "opponent": meta.opponent,
                "home_away": "Thuis" if meta.is_home else "Uit",
                "player_id": p.get("id"),
                "player_name": p.get("name"),
                "display_name": f"{short_name} - {jersey_num}",
                "minutes_played": minutes_played,
                "touches": touches,
                "possession_lost": possession_lost,
                "ball_losses_per_touch": ball_losses_per_touch,
                "passes": total_pass,
                "accurate_pass": accurate_pass,
                "tackles_won": tackles_won,
                "duels_won": duels_won,
                "successful_dribbles": successful_dribbles,
                "defensive_actions": defensive_actions,
                "key_passes": key_passes,
                "pass_accuracy_match": pass_accuracy_match,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Only keep player-match rows with >= 45 minutes
    df = df[df["minutes_played"] >= MIN_MATCH_MINUTES].copy()

    # Add per-90 versions
    per90_cols = [
        "tackles_won",
        "passes",
        "duels_won",
        "successful_dribbles",
        "defensive_actions",
        "key_passes",
        "possession_lost",
    ]
    for col in per90_cols:
        df[f"{col}_p90"] = np.where(
            df["minutes_played"] > 0,
            df[col] / df["minutes_played"] * 90.0,
            np.nan,
        )

    return df


@st.cache_data(show_spinner=False)
def load_all_match_player_data(match_ids: List[int], target_team_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      player_match_df: one row per player per match (only >=45 min)
      matches_df: one row per match with metadata
    """
    metas: List[MatchMeta] = []
    for event_id in match_ids:
        try:
            metas.append(_get_match_meta(event_id, target_team_name))
        except Exception:
            # Keep app usable even if 1 match fails meta
            metas.append(
                MatchMeta(
                    event_id=event_id,
                    start_ts=0,
                    date_utc=datetime(1970, 1, 1, tzinfo=timezone.utc),
                    is_home=False,
                    opponent="Onbekend",
                    label_nl="Onbekend. Uit",
                )
            )

    # Sort by date ascending (oldest left -> newest right)
    metas = sorted(metas, key=lambda m: (m.start_ts, m.event_id))

    dfs: List[pd.DataFrame] = []
    for i, meta in enumerate(metas, start=1):
        try:
            match_df = get_player_match_data(meta.event_id, meta)
            if not match_df.empty:
                dfs.append(match_df)
        except Exception:
            continue

    player_match_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if not player_match_df.empty:
        player_match_df = player_match_df.drop_duplicates(subset=["event_id", "player_id"]).reset_index(drop=True)

    matches_df = pd.DataFrame(
        [
            {
                "event_id": m.event_id,
                "match_ts": m.start_ts,
                "match_date_utc": m.date_utc,
                "match_label": m.label_nl,
                "opponent": m.opponent,
                "home_away": "Thuis" if m.is_home else "Uit",
            }
            for m in metas
        ]
    )

    return player_match_df, matches_df


# =========================
# STREAMLIT UI
# =========================


METRICS: Dict[str, Dict[str, Any]] = {
    "ball_losses_per_touch": {
        "label": "Balverlies per aanraking (%)",
        "format": ".1f",
        "y_title": "Balverlies per aanraking (%)",
    },
    "possession_lost": {
        "label": "Balverlies (totaal per match)",
        "format": ".0f",
        "y_title": "Balverlies",
    },
    "touches": {
        "label": "Aanrakingen (totaal per match)",
        "format": ".0f",
        "y_title": "Aanrakingen",
    },
    "pass_accuracy_match": {
        "label": "Passnauwkeurigheid (%)",
        "format": ".1f",
        "y_title": "Passnauwkeurigheid (%)",
    },
    "passes_p90": {
        "label": "Passes per 90",
        "format": ".1f",
        "y_title": "Passes per 90",
    },
    "tackles_won_p90": {
        "label": "Tackles gewonnen per 90",
        "format": ".2f",
        "y_title": "Tackles per 90",
    },
    "duels_won_p90": {
        "label": "Duels gewonnen per 90",
        "format": ".2f",
        "y_title": "Duels per 90",
    },
    "successful_dribbles_p90": {
        "label": "Succesvolle dribbles per 90",
        "format": ".2f",
        "y_title": "Dribbles per 90",
    },
    "defensive_actions_p90": {
        "label": "Defensive actions per 90",
        "format": ".2f",
        "y_title": "Defensive actions per 90",
    },
    "key_passes_p90": {
        "label": "Key passes per 90",
        "format": ".2f",
        "y_title": "Key passes per 90",
    },
}


def _init_state(players: List[str]) -> None:
    if "selected_metric" not in st.session_state:
        st.session_state["selected_metric"] = "ball_losses_per_touch"

    if "all_players_toggle" not in st.session_state:
        st.session_state["all_players_toggle"] = True

    if "selected_players" not in st.session_state:
        st.session_state["selected_players"] = players[:]  # all selected by default


def _set_all_players(players: List[str]) -> None:
    if st.session_state.get("all_players_toggle", True):
        st.session_state["selected_players"] = players[:]
    else:
        st.session_state["selected_players"] = []


def _sync_all_toggle(players: List[str]) -> None:
    selected = st.session_state.get("selected_players", [])
    st.session_state["all_players_toggle"] = len(selected) == len(players) and len(players) > 0


def main() -> None:
    st.set_page_config(page_title="FC Den Bosch — SofaScore Trends", layout="wide")
    st.title("FC Den Bosch — Trends per speler (≥45 min)")

    with st.spinner("Data laden van SofaScore (1x per sessie)..."):
        df, matches_df = load_all_match_player_data(MATCH_IDS, TARGET_TEAM_NAME)

    if df.empty:
        st.error("Geen data geladen. Check match IDs / SofaScore availability.")
        st.stop()

    # Build canonical match order (x-axis)
    match_order = (
        matches_df.sort_values(["match_ts", "event_id"], ascending=True)["match_label"]
        .dropna()
        .astype(str)
        .tolist()
    )

    # Player list
    players = (
        df[["display_name"]]
        .dropna()
        .astype(str)["display_name"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    _init_state(players)

    # Sidebar
    with st.sidebar:
        st.header("Instellingen")

        metric_keys = list(METRICS.keys())
        metric_labels = [METRICS[k]["label"] for k in metric_keys]
        label_to_key = dict(zip(metric_labels, metric_keys))

        current_metric_key = st.session_state["selected_metric"]
        current_metric_label = METRICS[current_metric_key]["label"]

        chosen_label = st.selectbox(
            "Variabele",
            options=metric_labels,
            index=metric_labels.index(current_metric_label) if current_metric_label in metric_labels else 0,
        )
        st.session_state["selected_metric"] = label_to_key[chosen_label]

        st.checkbox(
            "Alle spelers",
            key="all_players_toggle",
            value=st.session_state.get("all_players_toggle", True),
            on_change=_set_all_players,
            args=(players,),
        )

        chosen_players = st.multiselect(
            "Selecteer spelers",
            options=players,
            default=st.session_state.get("selected_players", players),
            key="selected_players",
        )
        _sync_all_toggle(players)

        st.caption(f"Matches in lijst: {len(MATCH_IDS)}")
        st.caption(f"Rows (player-match, ≥45 min): {len(df)}")

        if st.button("Herlaad data (cache wissen)"):
            load_all_match_player_data.clear()
            st.rerun()

    selected_metric = st.session_state["selected_metric"]
    selected_players = st.session_state.get("selected_players", [])

    if not selected_players:
        st.warning("Selecteer minimaal 1 speler.")
        st.stop()

    # Filter data
    dff = df[df["display_name"].isin(selected_players)].copy()
    if dff.empty:
        st.warning("Geen data voor deze selectie.")
        st.stop()

    # Ensure x-axis categories match chronological order (oldest -> newest)
    dff["match_label"] = dff["match_label"].astype(str)
    dff["match_label"] = pd.Categorical(dff["match_label"], categories=match_order, ordered=True)

    # Plot
    metric_info = METRICS[selected_metric]
    y_title = metric_info["y_title"]

    # Hover: keep it useful
    hover_cols = {
        "display_name": True,
        "match_label": True,
        "minutes_played": True,
        "passes": True,
        "possession_lost": True,
        "touches": True,
        "pass_accuracy_match": True,
        "tackles_won": True,
        "duels_won": True,
        "successful_dribbles": True,
        "defensive_actions": True,
        "key_passes": True,
    }

    fig = px.line(
        dff.sort_values(["match_ts", "event_id"]),
        x="match_label",
        y=selected_metric,
        color="display_name",
        markers=True,
        category_orders={"match_label": match_order},
        hover_data=hover_cols,
    )

    fig.update_layout(
        height=700,
        xaxis_title="Match (opponent + thuis/uit) — oud → nieuw",
        yaxis_title=y_title,
        legend_title_text="Speler",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_xaxes(tickangle=-45)

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Data (gefilterd)"):
        show_cols = [
            "match_date_utc",
            "match_label",
            "display_name",
            "minutes_played",
            selected_metric,
        ]
        extra = [c for c in ["touches", "possession_lost", "passes", "pass_accuracy_match"] if c in dff.columns]
        show_cols = list(dict.fromkeys(show_cols + extra))
        st.dataframe(
            dff.sort_values(["match_ts", "event_id", "display_name"])[show_cols],
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
