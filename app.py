from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import streamlit.components.v1 as components


# -----------------------------------------------------------------------------
# APP CONFIGURATION
# -----------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("pulserank")

st.set_page_config(
    page_title="PulseRank | Adaptive Video Discovery",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_RESULTS = 50
CACHE_TTL_SECONDS = 900
EMBEDDING_DIM = 16
UCB_COEFFICIENT = 0.30
DIVERSITY_COEFFICIENT = 0.50

st.markdown(
    """
    <style>
      html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
      [data-testid="stMainBlockContainer"] { overflow: hidden !important; }
      /* Keep Streamlit's header container so the sidebar toggle remains accessible. */
      [data-testid="stHeader"] {
        height: 2.5rem !important;
        min-height: 2.5rem !important;
        background: transparent !important;
        z-index: 1000 !important;
        overflow: visible !important;
      }
      /* Do not hide the toolbar ancestor: Streamlit may place the sidebar toggle inside it. */
      [data-testid="stToolbar"] {
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        z-index: 1001 !important;
      }
      [data-testid="stSidebarCollapsedControl"],
      [data-testid="stSidebarCollapsedControl"] button,
      [data-testid="stSidebarCollapseButton"],
      [data-testid="stSidebarCollapseButton"] button,
      [data-testid="stHeader"] button[kind="header"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        position: relative !important;
        z-index: 1002 !important;
      }
      [data-testid="stHeader"] button[kind="header"] svg,
      [data-testid="stSidebarCollapsedControl"] svg,
      [data-testid="stSidebarCollapseButton"] svg {
        visibility: visible !important;
        opacity: 1 !important;
      }
      [data-testid="stMainBlockContainer"] { padding: 0 !important; max-height: 100vh; }

      /* Keep the settings sidebar permanently expanded: hide collapse affordances,
         not the sidebar itself. This avoids the previous invisible-toggle state. */
      [data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        transform: none !important;
        min-width: 17rem !important;
        width: 17rem !important;
        max-width: 17rem !important;
        opacity: 1 !important;
        border-right: 1px solid rgba(100,116,139,.18);
        overflow-y: auto !important;
        overflow-x: hidden !important;
      }
      [data-testid="stSidebar"] > div:first-child {
        width: 17rem !important;
        min-width: 17rem !important;
        max-width: 17rem !important;
        height: 100vh !important;
        overflow-y: auto !important;
      }
      [data-testid="stSidebarCollapseButton"],
      [data-testid="stSidebarCollapseButton"] button,
      [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
      }
      /* Keep the custom dashboard within the remaining viewport without
         clipping or overlapping Streamlit's always-open sidebar. */
      [data-testid="stAppViewContainer"] > .main {
        min-width: 0 !important;
        overflow: hidden !important;
      }
      div[data-testid="stAlert"] { border-radius: 12px; }
      iframe[title="streamlit.components.v1.html"] { display:block; width:100%; border:0; }
      .pr-sidebar-note { color: #64748b; font-size: .82rem; line-height: 1.45; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_secret_api_key() -> str:
    """Read an API key from env/.env or Streamlit secrets without exposing it."""
    key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("YOUTUBE_API_KEY", "")).strip()
    except Exception:
        # Streamlit raises when no secrets file is configured; that is normal locally.
        return ""


def _thumbnail_from_snippet(snippet: dict[str, Any]) -> str:
    thumbnails = snippet.get("thumbnails", {})
    for quality in ("maxres", "high", "medium", "default"):
        entry = thumbnails.get(quality, {})
        url = entry.get("url")
        if isinstance(url, str) and url.startswith("https://"):
            return url
    return ""


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_youtube_feed(api_key: str, query: str, max_results: int = MAX_RESULTS) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch real video metadata. Returns records and a safe user-facing error."""
    if not api_key:
        return [], "A YouTube Data API key is required to load live videos."

    try:
        youtube = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
        search_response = (
            youtube.search()
            .list(
                q=query,
                type="video",
                part="id,snippet",
                maxResults=max(1, min(int(max_results), 50)),
                safeSearch="moderate",
                relevanceLanguage="en",
            )
            .execute()
        )

        video_ids = [
            item.get("id", {}).get("videoId")
            for item in search_response.get("items", [])
            if item.get("id", {}).get("videoId")
        ]
        if not video_ids:
            return [], None

        stats_response = (
            youtube.videos()
            .list(part="snippet,statistics", id=",".join(video_ids))
            .execute()
        )

        by_id: dict[str, dict[str, Any]] = {}
        for item in stats_response.get("items", []):
            video_id = item.get("id")
            snippet = item.get("snippet", {})
            if not video_id:
                continue
            try:
                views = int(item.get("statistics", {}).get("viewCount", 0))
            except (TypeError, ValueError):
                views = 0
            by_id[video_id] = {
                "video_id": video_id,
                "title": str(snippet.get("title", "Untitled video")),
                "channel": str(snippet.get("channelTitle", "Unknown channel")),
                "views": max(0, views),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "thumbnail": _thumbnail_from_snippet(snippet),
                "is_demo": False,
            }

        # Preserve search relevance order rather than the order returned by videos.list.
        records = [by_id[video_id] for video_id in video_ids if video_id in by_id]
        return records, None

    except HttpError as exc:
        status = getattr(getattr(exc, "resp", None), "status", None)
        if status == 403:
            message = (
                "YouTube rejected the request (HTTP 403). Check that the YouTube Data API v3 "
                "is enabled, the key is valid, and quota has not been exhausted."
            )
        elif status == 400:
            message = "YouTube rejected the search request (HTTP 400). Check the API key and search query."
        else:
            message = f"YouTube API request failed (HTTP {status or 'unknown'}). Please try again."
        LOGGER.warning("YouTube API error: %s", exc)
        return [], message
    except Exception:
        LOGGER.exception("Unexpected error while fetching YouTube videos")
        return [], "Could not load videos from YouTube. Check your connection and API configuration, then retry."


def make_demo_videos(query: str, count: int = MAX_RESULTS) -> list[dict[str, Any]]:
    """Explicit preview-only data. Demo entries are never presented as live videos."""
    seed = int.from_bytes(hashlib.sha256(query.encode("utf-8")).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    view_counts = (rng.zipf(1.8, count) * 1_000).astype(int)
    records: list[dict[str, Any]] = []
    for idx in range(count):
        video_id = f"demo-{idx + 1}"
        records.append(
            {
                "video_id": video_id,
                "title": f"Preview video {idx + 1}: {query} — discovery example",
                "channel": f"Preview Creator {(idx % 10) + 1}",
                "views": int(view_counts[idx]),
                "url": "",
                "thumbnail": f"https://picsum.photos/seed/pulserank-{seed}-{idx}/640/360",
                "is_demo": True,
            }
        )
    return records


def stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def enrich_videos(records: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Add popularity categories and stable synthetic complex-vector ranking features."""
    if not records:
        return []

    df = pd.DataFrame(records).copy()
    views = df["views"].to_numpy(dtype=float)
    p80 = float(np.percentile(views, 80))
    p50 = float(np.percentile(views, 50))

    def category_for_view_count(view_count: float) -> str:
        if view_count >= p80:
            return "head"
        if view_count >= p50:
            return "mid"
        return "tail"

    df["category"] = df["views"].apply(category_for_view_count)
    df["category_label"] = df["category"].map(
        {"head": "Popular", "mid": "Mid-tail", "tail": "Long-tail"}
    )

    log_views = np.log1p(views)
    view_span = float(log_views.max() - log_views.min()) if len(log_views) else 0.0
    normalized_views = (log_views - log_views.min()) / (view_span + 1e-10)

    user_rng = np.random.default_rng(stable_seed("pulserank-user:" + query.casefold().strip()))
    user_vector = user_rng.normal(0, 0.1, EMBEDDING_DIM) + 1j * user_rng.normal(0, 0.05, EMBEDDING_DIM)
    user_vector = user_vector / (np.linalg.norm(user_vector) + 1e-12)

    enriched: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        video_id = str(row["video_id"])
        rng = np.random.default_rng(stable_seed("pulserank-video:" + video_id))
        real_part = rng.normal(0, 0.1, EMBEDDING_DIM)
        real_part[0] = float(normalized_views[idx])
        imag_part = rng.normal(0, 0.05, EMBEDDING_DIM)
        psi = real_part + 1j * imag_part
        psi = psi / (np.linalg.norm(psi) + 1e-12)

        inner = np.sum(psi * np.conjugate(user_vector))
        phase_angle = float(np.angle(inner) + (np.pi / 4.0))
        score = float(np.real(inner) + UCB_COEFFICIENT * 0.25 * np.cos(phase_angle))

        # Client uses a normalized real-valued signature for lightweight MMR-style diversity.
        signature = np.concatenate([psi.real, psi.imag]).astype(float)
        signature = signature / (np.linalg.norm(signature) + 1e-12)

        video = row.to_dict()
        video["base_score"] = score
        video["features"] = [round(float(x), 7) for x in signature]
        video["views"] = int(video["views"])
        video["title"] = str(video["title"])
        video["channel"] = str(video["channel"])
        video["url"] = str(video.get("url", ""))
        video["thumbnail"] = str(video.get("thumbnail", ""))
        video["is_demo"] = bool(video.get("is_demo", False))
        enriched.append(video)

    return enriched


def build_frontend(records: list[dict[str, Any]], query: str, is_demo: bool) -> str:
    """Build a self-contained front end; scroll tracking and ranking stay in-browser."""
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    # Prevent data from ending the script element if a source title contains that sequence.
    payload = payload.replace("</", "<\\/")
    query_json = json.dumps(query, ensure_ascii=False).replace("</", "<\\/")
    data_mode_json = "true" if is_demo else "false"

    return r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{color-scheme:light;--ink:#182338;--muted:#6f7a8e;--soft:#f5f7fb;--line:#e4e8f0;--white:#fff;--brand:#5946d9;--brand-dark:#4534bc;--brand-soft:#f0edff;--green:#168567;--shadow:0 6px 20px rgba(22,34,58,.045)}
*{box-sizing:border-box}html,body{height:100%;width:100%;margin:0;overflow:hidden!important}body{background:#f6f8fc;color:var(--ink);font:13px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,a{font:inherit}button{cursor:pointer}a{color:inherit}
.topbar{height:54px;min-height:54px;display:flex;align-items:center;justify-content:space-between;padding:0 24px;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line)}.brand{display:flex;align-items:center;gap:10px;font-size:17px;font-weight:800;letter-spacing:-.04em}.brand-mark{width:30px;height:30px;border-radius:9px;background:linear-gradient(145deg,#7362ec,#4434b9);color:#fff;display:grid;place-items:center;box-shadow:0 3px 9px #5946d92b}.brand-mark svg{width:16px;height:16px}.brand-sub{margin-top:0;color:#8992a3;font-size:9px;font-weight:750;letter-spacing:.12em;text-transform:uppercase}.top-actions{display:flex;align-items:center;gap:8px}.source-pill,.live-pill{display:inline-flex;align-items:center;gap:7px;border:1px solid var(--line);border-radius:999px;padding:6px 9px;background:#fff;color:#566176;font-size:10px;font-weight:750;white-space:nowrap}.dot{width:6px;height:6px;border-radius:50%;background:#1ba579;box-shadow:0 0 0 3px #1ba57918}.dot.preview{background:#d99a24;box-shadow:0 0 0 3px #d99a2418}.live-pill{background:#f3f1ff;border-color:#e4dfff;color:#5645c4}
.page{height:calc(100vh - 54px);max-height:calc(100vh - 54px);max-width:1540px;margin:0 auto;padding:13px 22px 10px;display:flex;flex-direction:column;overflow:hidden}.hero{display:flex;flex:0 0 auto;align-items:center;justify-content:space-between;gap:18px;margin:0 0 12px;min-height:82px}.hero-copy{min-width:0}.eyebrow{display:flex;align-items:center;gap:7px;color:var(--brand);font-size:9px;font-weight:850;letter-spacing:.15em;text-transform:uppercase}.eyebrow-line{width:18px;height:1px;background:var(--brand)}h1{margin:5px 0 3px;font-size:clamp(22px,2.3vw,29px);line-height:1.12;letter-spacing:-.055em;font-weight:820}.hero p{margin:0;max-width:740px;color:var(--muted);font-size:11px;line-height:1.45}.hero-state{flex:0 0 225px;min-width:210px;background:#fff;border:1px solid var(--line);border-radius:13px;padding:10px 12px;box-shadow:var(--shadow)}.state-top,.state-foot{display:flex;align-items:center;justify-content:space-between;gap:8px}.state-top{color:#8b94a4;font-size:9px;font-weight:800;letter-spacing:.07em}.mode-name{margin-top:3px;font-size:14px;font-weight:800;letter-spacing:-.025em}.meter{height:4px;margin-top:8px;background:#eceff5;border-radius:999px;overflow:hidden}.meter-fill{height:100%;width:48%;border-radius:inherit;background:linear-gradient(90deg,#a89bff,#5946d9);transition:width .3s ease}.state-foot{margin-top:5px;color:#828da0;font-size:9px}
.workspace{display:grid;grid-template-columns:minmax(0,1fr) 292px;gap:14px;align-items:stretch;flex:1 1 auto;min-height:0;overflow:hidden}.feed-column{display:flex;flex-direction:column;min-width:0;min-height:0;overflow:hidden}.feed-toolbar{flex:0 0 auto;display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 0 7px;padding:0 1px}.feed-title{font-size:14px;font-weight:820;letter-spacing:-.025em}.feed-subtitle{margin-top:1px;color:var(--muted);font-size:10px}.feed-controls{display:flex;align-items:center;gap:7px}.status-chip{padding:5px 8px;border-radius:8px;background:var(--brand-soft);color:#5040bf;font-size:10px;font-weight:800;white-space:nowrap}.feed-list{display:grid;gap:8px;align-content:start;flex:1 1 auto;min-height:0;overflow-y:auto;overflow-x:hidden;overscroll-behavior:contain;scrollbar-gutter:stable;scrollbar-width:thin;scrollbar-color:#c9cfe0 transparent;padding:2px 7px 8px 2px;overflow-anchor:none}.feed-list::-webkit-scrollbar,.right-rail::-webkit-scrollbar{width:6px}.feed-list::-webkit-scrollbar-thumb,.right-rail::-webkit-scrollbar-thumb{background:#cbd1df;border-radius:99px}.feed-list.has-rendered .video-card{animation:none!important}.video-card{min-width:0;min-height:108px;display:grid;grid-template-columns:minmax(138px,18%) minmax(0,1fr);gap:12px;padding:8px;border:1px solid var(--line);border-radius:12px;background:var(--white);box-shadow:0 1px 4px rgba(22,34,58,.025);animation:card-in .28s ease both;transition:border-color .18s,box-shadow .18s,background .18s}.video-card:hover{border-color:#d7d0ff;box-shadow:0 4px 14px rgba(40,32,100,.07)}.video-card.is-adapted{border-color:#c6bef8;background:#fdfcff;box-shadow:0 0 0 1px #5946d911 inset,0 4px 14px #5946d90a}.video-card.is-moving{z-index:1;will-change:transform}@keyframes card-in{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}.thumb-wrap{position:relative;align-self:stretch;min-height:88px;aspect-ratio:16/9;border-radius:8px;overflow:hidden;background:#e9ecf8}.thumb{width:100%;height:100%;object-fit:cover;display:block;transition:transform .25s}.video-card:hover .thumb{transform:scale(1.02)}.thumb-fallback{position:absolute;inset:0;display:grid;place-items:center;color:#6758c6;font-size:22px;font-weight:850;background:linear-gradient(135deg,#f0edff,#e0e8f8)}.thumb-fallback[hidden]{display:none}.card-main{display:flex;flex-direction:column;min-width:0;padding:1px 1px 0}.category-row{display:flex;align-items:center;gap:6px;flex-wrap:wrap}.category{padding:3px 6px;border-radius:5px;font-size:9px;font-weight:850}.category.head{color:#ae5a28;background:#fff0e5}.category.mid{color:#87630b;background:#fff7d8}.category.tail{color:#167357;background:#e4f7ef}.new-badge{font-size:8px;color:#8992a3;font-weight:800;letter-spacing:.08em}.adapted-badge{padding:2px 5px;border-radius:4px;color:#4e3ac0;background:#efebff;font-size:8px;font-weight:850;letter-spacing:.04em}.adapted-badge[hidden]{display:none}.video-title{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;margin:5px 0 4px;color:#202b40;text-decoration:none;font-size:12px;line-height:1.36;font-weight:800;letter-spacing:-.015em}.video-title:hover{color:var(--brand)}.channel{display:flex;align-items:center;gap:6px;color:#778195;font-size:9px;min-width:0}.channel span:last-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.channel-avatar{width:17px;height:17px;border-radius:50%;background:#ece9ff;color:#5b4bd0;display:grid;place-items:center;font-size:8px;font-weight:850;flex-shrink:0}.card-bottom{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-top:auto;padding-top:5px}.video-meta{display:flex;align-items:center;gap:9px;color:#788397;font-size:9px;flex-wrap:wrap}.watch-link{color:var(--brand);text-decoration:none;font-size:9px;font-weight:800;white-space:nowrap}.watch-link:hover{text-decoration:underline}.no-link{font-size:9px;color:#9aa2b0}.footnote{flex:0 0 auto;padding:5px 2px 0;color:#959eae;font-size:9px;line-height:1.35}
.right-rail{display:flex;flex-direction:column;gap:10px;min-width:0;min-height:0;overflow-y:hidden;overflow-x:hidden;overscroll-behavior:contain;padding:0 2px 2px 0}.rail-card{flex:0 0 auto;padding:12px;border:1px solid var(--line);border-radius:13px;background:#fff;box-shadow:0 1px 4px rgba(22,34,58,.025)}.rail-title{font-size:12px;font-weight:850;letter-spacing:-.02em}.rail-description{margin-top:3px;color:var(--muted);font-size:9px;line-height:1.4}.proof-head{display:flex;align-items:center;justify-content:space-between;gap:7px}.proof-status{padding:4px 6px;border-radius:999px;background:#eaf7f1;color:#187958;font-size:8px;font-weight:850;white-space:nowrap}.proof-status.demo{background:#f1edff;color:#5845c6}.proof-status.attention{background:#fff3df;color:#94600e}.count-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin-top:9px}.count-box{padding:7px 6px;border:1px solid #edf0f5;border-radius:8px;background:#f8f9fc}.count-value{font-size:17px;line-height:1.1;font-weight:850;letter-spacing:-.05em}.count-label{margin-top:3px;color:#8992a3;font-size:8px;font-weight:800;letter-spacing:.04em}.mini-divider{height:1px;background:var(--line);margin:10px 0 8px}.mix-heading{display:flex;justify-content:space-between;gap:8px;align-items:center;font-size:10px;font-weight:800}.mix-heading span:last-child{font-size:9px;color:#858fa1;font-weight:700}.signal-list{display:grid;gap:8px;margin-top:9px}.signal-row{display:grid;grid-template-columns:64px minmax(0,1fr) 18px;align-items:center;gap:7px;color:#667286;font-size:9px}.signal-track{height:5px;background:#edf0f5;border-radius:99px;overflow:hidden}.signal-fill{height:100%;border-radius:inherit;transition:width .3s ease}.signal-fill.head{background:#e79a61}.signal-fill.mid{background:#eac95b}.signal-fill.tail{background:#64b69a}.signal-value{text-align:right;color:#475368;font-weight:850}.proof-speed-row{display:flex;align-items:flex-end;justify-content:space-between;gap:8px;margin-top:9px}.proof-kicker{color:#8a93a2;font-size:8px;font-weight:850;letter-spacing:.07em}.proof-speed{margin-top:2px;font-size:23px;line-height:1.05;font-weight:850;letter-spacing:-.055em}.proof-speed small{font-size:10px;color:#7f899b;font-weight:750;letter-spacing:0}.proof-event-stat{text-align:right}.proof-event-stat strong{display:block;margin-top:3px;font-size:16px;letter-spacing:-.04em}.proof-compare{min-height:45px;margin-top:9px;padding:8px 9px;border:1px solid #e9e5ff;border-radius:8px;background:#f8f6ff;color:#554a81;font-size:9px;line-height:1.45;white-space:pre-line;overflow-wrap:anywhere}.proof-button{width:100%;margin-top:8px;border:0;border-radius:8px;padding:8px 9px;background:var(--brand);color:white;font-size:10px;font-weight:850;transition:background .15s,transform .15s}.proof-button:hover{background:var(--brand-dark)}.proof-button:active{transform:translateY(1px)}.proof-button:disabled{opacity:.58;cursor:wait}.proof-disclaimer{margin-top:6px;color:#8a93a3;font-size:8px;line-height:1.4}.event-log-title{margin-top:10px;color:#5a6477;font-size:9px;font-weight:850}.event-log{display:grid;gap:5px;list-style:none;margin:5px 0 0;padding:0}.event-log li{position:relative;padding-left:10px;color:#687386;font-size:8px;line-height:1.4;max-height:2.8em;overflow:hidden;overflow-wrap:anywhere}.event-log li:before{content:"";position:absolute;left:0;top:4px;width:5px;height:5px;border-radius:50%;background:#b9b0f3}.event-log li.event-empty{color:#9aa2b0}.event-log li.event-empty:before{background:#d7dce5}.privacy-line{margin-top:7px;color:#8a93a3;font-size:8px;line-height:1.4}.empty-state{padding:28px;text-align:center;color:var(--muted);background:white;border:1px solid var(--line);border-radius:12px}.feed-updating{animation:soft-pulse 1s ease-in-out infinite}@keyframes soft-pulse{50%{opacity:.7}}.live-label{color:#198566}.preview-label{color:#a86a0a}
@media(max-height:720px){.page{padding-top:8px;padding-bottom:6px}.hero{min-height:67px;margin-bottom:8px}h1{font-size:22px}.hero-state{padding:8px 10px}.rail-card{padding:9px}.workspace{gap:9px}.video-card{min-height:96px;padding:7px;gap:9px}.thumb-wrap{min-height:76px}.proof-speed{font-size:20px}.footnote{display:none}}
@media(max-width:960px){.workspace{grid-template-columns:minmax(0,1fr) 260px;gap:10px}.page{padding-left:14px;padding-right:14px}.video-card{grid-template-columns:minmax(120px,22%) minmax(0,1fr)}}
@media(max-width:720px){.topbar{height:48px;min-height:48px;padding:0 12px}.brand{font-size:15px}.brand-mark{width:27px;height:27px}.brand-sub{display:none}.source-pill{display:none}.page{height:calc(100vh - 48px);max-height:calc(100vh - 48px);padding:9px}.hero{gap:9px;align-items:flex-start;margin-bottom:8px;min-height:0}.eyebrow{font-size:8px}h1{font-size:21px;margin:4px 0}.hero p{font-size:9px}.hero-state{flex-basis:170px;min-width:160px;padding:8px}.mode-name{font-size:12px}.workspace{grid-template-columns:minmax(0,1fr);grid-template-rows:minmax(0,1fr) minmax(125px,34%);gap:8px}.right-rail{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));grid-auto-rows:min-content;overflow-y:auto}.rail-card{padding:8px}.proof-card{grid-column:span 1}.video-card{grid-template-columns:110px minmax(0,1fr);min-height:88px;padding:6px;gap:8px}.thumb-wrap{min-height:62px}.video-title{font-size:10px}.card-bottom{padding-top:4px}.watch-link{font-size:8px}.feed-subtitle{font-size:9px}.footnote{display:none}}
@media(max-width:500px){.hero{display:block}.hero-state{display:flex;flex-direction:column;margin-top:7px;max-width:none;width:100%}.hero p{max-width:none}.workspace{grid-template-rows:minmax(0,1fr) minmax(130px,34%)}.right-rail{grid-template-columns:repeat(2,minmax(0,1fr))}.proof-card{grid-column:span 1}.video-card{grid-template-columns:96px minmax(0,1fr)}.top-actions .live-pill{font-size:8px;padding:5px 7px}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;scroll-behavior:auto!important;transition-duration:.01ms!important}}
</style>
</head>
<body>
<header class="topbar">
  <div class="brand"><div class="brand-mark"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M8 5.8c0-.8.9-1.3 1.6-.9l9.1 5.2a1 1 0 0 1 0 1.8l-9.1 5.2a1.05 1.05 0 0 1-1.6-.9V5.8Z" fill="currentColor"/></svg></div><div>PulseRank<div class="brand-sub">Adaptive discovery</div></div></div>
  <div class="top-actions"><div class="source-pill" id="sourcePill"><span class="dot" id="sourceDot"></span><span id="sourceText">Live source</span></div><div class="live-pill"><span class="dot"></span> LIVE ADAPTATION</div></div>
</header>
<main class="page">
  <section class="hero">
    <div class="hero-copy"><div class="eyebrow"><span class="eyebrow-line"></span> A FEED THAT MOVES WITH YOU</div><h1>Find your next watch.</h1><p id="heroDescription">A clean discovery feed that adjusts its ranking to your browsing pace. All available results are loaded immediately.</p></div>
    <div class="hero-state"><div class="state-top"><span>SCROLL SIGNAL</span><span id="signalState">Listening</span></div><div class="mode-name" id="modeName">Balanced discovery</div><div class="meter"><div class="meter-fill" id="meterFill"></div></div><div class="state-foot"><span id="velocityReadout">0 px/s</span><span id="directionReadout">Scroll to adapt</span></div></div>
  </section>
  <div class="workspace">
    <section class="feed-column">
      <div class="feed-toolbar"><div><div class="feed-title">Recommended videos</div><div class="feed-subtitle" id="feedSubtitle">Results for your topic · all available videos loaded</div></div></div>
      <div class="feed-list" id="feed" aria-live="off"></div>
      <div class="footnote">Experimental ranking prototype · not affiliated with YouTube.</div>
    </section>
    <aside class="right-rail" aria-label="Feed insights">
      <section class="rail-card overview-card">
        <div class="rail-title">Feed overview</div>
        <div class="count-grid"><div class="count-box"><div class="count-value" id="shownCount">0</div><div class="count-label">SHOWN</div></div><div class="count-box"><div class="count-value" id="loadedCount">0</div><div class="count-label">AVAILABLE</div></div><div class="count-box"><div class="count-value" id="viewedCount">0</div><div class="count-label">VIEWED</div></div></div>
        <div class="mini-divider"></div>
        <div class="mix-heading"><span>Upcoming mix</span><span id="mixWindowCount">0 of 10</span></div>
        <div class="signal-list"><div class="signal-row"><span>Popular</span><div class="signal-track"><div class="signal-fill head" id="barHead" style="width:0%"></div></div><span class="signal-value" id="numHead">0</span></div><div class="signal-row"><span>Mid-tail</span><div class="signal-track"><div class="signal-fill mid" id="barMid" style="width:0%"></div></div><span class="signal-value" id="numMid">0</span></div><div class="signal-row"><span>Long-tail</span><div class="signal-track"><div class="signal-fill tail" id="barTail" style="width:0%"></div></div><span class="signal-value" id="numTail">0</span></div></div>
        <span id="mixWindowTitle" hidden>Adaptive window</span><span id="mixDescription" hidden>Actual next-up mix</span>
      </section>
      <section class="rail-card proof-card">
        <div class="proof-head"><div class="rail-title">Adaptation activity</div><span class="proof-status" id="proofStatus">Listening</span></div>
        <div class="proof-speed-row"><div><div class="proof-kicker">SCROLL SPEED</div><div class="proof-speed" id="proofSpeed">0 <small>px/s</small></div></div><div class="proof-event-stat"><div class="proof-kicker">SCROLL EVENTS</div><strong id="scrollEventCount">0</strong></div></div>
        <div class="proof-compare" id="lastMixChange" aria-live="polite">Scroll the feed to see measured speed, mode changes and ranking results here.</div>
        <button class="proof-button" id="runDemo" type="button">▶ Run guided demo</button>
        <div class="proof-disclaimer">Demo cycles ranking modes intentionally. Live mode responds to actual feed scrolling.</div>
        <div class="event-log-title">Recent changes</div><ol class="event-log" id="eventLog"><li class="event-empty">No changes yet. Scroll the feed or run the demo.</li></ol>
        <div class="privacy-line">Scroll measurements stay in this browser.</div>
      </section>
      <span id="modeRowTitle" hidden>Balanced discovery</span><span id="modeVelocity" hidden>Steady pace</span><span id="modeCopy" hidden>A balanced mix</span>
    </aside>
  </div>
</main>
<script>
const VIDEOS = __VIDEO_DATA__;
const SEARCH_QUERY = __QUERY_DATA__;
const IS_DEMO = __IS_DEMO__;
const MODE = {
  slow: {label:'Relaxed pace', title:'Popularity-weighted', chip:'Popular first', copy:'The full feed shifts toward widely viewed videos, including items already viewed.', quotas:{head:0.60,mid:0.30,tail:0.10}, order:['head','mid','tail'], meter:18},
  balanced: {label:'Balanced discovery', title:'Balanced discovery', chip:'Balanced mix', copy:'The full feed balances popularity and discovery across viewed and unviewed items.', quotas:{head:0.30,mid:0.30,tail:0.40}, order:['mid','tail','head'], meter:48},
  fast: {label:'Exploration mode', title:'Long-tail exploration', chip:'More new finds', copy:'The full feed favours less-viewed videos, including items already viewed.', quotas:{head:0.10,mid:0.20,tail:0.70}, order:['tail','mid','head'], meter:84}
};
const CATEGORY = {head:{label:'Popular',className:'head'},mid:{label:'Mid-tail',className:'mid'},tail:{label:'Long-tail',className:'tail'}};
const state = {mode:'balanced', shown:[], viewed:new Set(), rendered:0, smoothedSpeed:0, lastY:0, lastTime:performance.now(), pendingMode:null, pendingSince:0, lastModeChange:0, scrollTimer:null, modeCommitTimer:null, scrollEvents:0, demoRunning:false, direction:'—', isReordering:false, transitionToken:0};
let cardObserver = null;
let activeReorderCleanup = null;
const REDUCED_MOTION = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const REORDER_DURATION = REDUCED_MOTION ? 0 : 460;
const el = id => document.getElementById(id);
const nf = new Intl.NumberFormat(undefined,{notation:'compact',maximumFractionDigits:1});
const rawScores = VIDEOS.map(v => Number(v.base_score) || 0);
const maxScore = rawScores.length ? Math.max(...rawScores) : 0;
const minScore = rawScores.length ? Math.min(...rawScores) : 0;
VIDEOS.forEach(v => { const s=Number(v.base_score)||0; v.quality=(s-minScore)/(maxScore-minScore+1e-10); });
el('feedSubtitle').textContent = `${VIDEOS.length} ${VIDEOS.length === 1 ? 'result' : 'results'} for “${SEARCH_QUERY}” · all available results are shown`;
el('sourceText').textContent = IS_DEMO ? 'Preview data' : 'YouTube API';
el('sourceDot').classList.toggle('preview',IS_DEMO);
if (IS_DEMO) { el('sourcePill').classList.add('preview-label'); el('heroDescription').textContent='Preview mode is using clearly labelled sample content. Connect a YouTube Data API key to load live videos. All preview items are rendered immediately; scroll pace re-ranks the entire feed.'; }
el('loadedCount').textContent = String(VIDEOS.length);
el('shownCount').textContent = '0';
el('viewedCount').textContent = '0';
el('modeName').textContent=MODE.balanced.label;
el('modeRowTitle').textContent=MODE.balanced.title;
el('modeCopy').textContent=MODE.balanced.copy;
el('modeVelocity').textContent='Steady pace';
updateSpeedDisplay(0,'Scroll to adapt');

function cosineSimilarity(a,b){let dot=0,na=0,nb=0;const n=Math.min(a?.length||0,b?.length||0);for(let i=0;i<n;i++){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}return dot/(Math.sqrt(na*nb)+1e-10);}
function allocate(n,weights){
  const keys=Object.keys(weights), total=keys.reduce((s,k)=>s+weights[k],0)||1;
  const raw=keys.map(k=>({key:k,value:n*weights[k]/total}));
  const out={};let used=0;raw.forEach(x=>{out[x.key]=Math.floor(x.value);used+=out[x.key];});
  raw.sort((a,b)=>(b.value-Math.floor(b.value))-(a.value-Math.floor(a.value)));
  for(let i=0;i<n-used;i++)out[raw[i%raw.length].key]++;
  return out;
}
function chooseFromPool(pool,count,chosen){
  const result=[];const working=pool.slice();
  while(result.length<count && working.length){
    let bestIndex=0,bestScore=-Infinity;
    for(let i=0;i<working.length;i++){
      const video=working[i];
      let novelty=0.5;
      if(chosen.length){
        let maxSim=-1;
        for(const prior of chosen){maxSim=Math.max(maxSim,cosineSimilarity(video.features,prior.features));}
        novelty=1-((maxSim+1)/2);
      }
      const adjusted=0.70*video.quality+0.30*novelty;
      if(adjusted>bestScore){bestScore=adjusted;bestIndex=i;}
    }
    const pick=working.splice(bestIndex,1)[0];result.push(pick);chosen.push(pick);
  }
  return result;
}
function rankAllForMode(modeName, sourceVideos=VIDEOS){
  const source = sourceVideos.slice();
  if(!source.length)return [];
  const mode = MODE[modeName] || MODE.balanced;
  const remaining = source.slice();
  const ordered = [];

  // Apply the target category ratio to each 10-video window, not across the
  // entire catalogue. This makes slow/balanced/fast modes observably different
  // in the upcoming recommendation mix while never duplicating an item.
  while(remaining.length){
    const batchSize = Math.min(10, remaining.length);
    const targets = allocate(batchSize, mode.quotas);
    const batchCandidates = [];
    const groups = {head:[],mid:[],tail:[]};

    for(const key of ['head','mid','tail']){
      const pool = remaining.filter(video => video.category === key);
      groups[key] = chooseFromPool(pool, Math.min(targets[key] || 0, pool.length), batchCandidates);
    }

    const batch = [];
    const groupQueues = {
      head:groups.head.slice(),
      mid:groups.mid.slice(),
      tail:groups.tail.slice()
    };
    while(batch.length < batchSize){
      let added = false;
      for(const key of mode.order){
        if(groupQueues[key].length && batch.length < batchSize){
          batch.push(groupQueues[key].shift());
          added = true;
        }
      }
      if(!added)break;
    }

    // If a category has too few remaining videos to meet its quota, fill the
    // shortfall from the remaining unique items using the same quality/novelty
    // helper, rather than returning fewer than the available results.
    if(batch.length < batchSize){
      const batchIds = new Set(batch.map(video => video.video_id));
      const leftovers = remaining.filter(video => !batchIds.has(video.video_id));
      chooseFromPool(leftovers, batchSize - batch.length, batch);
    }

    const batchIds = new Set(batch.map(video => video.video_id));
    if(batchIds.size === 0)break;
    ordered.push(...batch);
    for(let i=remaining.length-1;i>=0;i--){
      if(batchIds.has(remaining[i].video_id))remaining.splice(i,1);
    }
  }
  return ordered;
}
function createCard(v,index){
  const article=document.createElement('article');article.className='video-card';article.dataset.videoId=v.video_id;article.style.animationDelay=`${Math.min(index%10,5)*18}ms`;
  const thumbWrap=document.createElement('div');thumbWrap.className='thumb-wrap';
  const img=document.createElement('img');img.className='thumb';img.loading='lazy';img.alt=`Thumbnail for ${v.title}`;img.referrerPolicy='no-referrer';
  const fallback=document.createElement('div');fallback.className='thumb-fallback';fallback.textContent='PR';fallback.hidden=true;
  if(v.thumbnail){img.src=v.thumbnail;img.onerror=()=>{img.hidden=true;fallback.hidden=false;};}else{img.hidden=true;fallback.hidden=false;}
  thumbWrap.append(img,fallback);
  const main=document.createElement('div');main.className='card-main';
  const categoryRow=document.createElement('div');categoryRow.className='category-row';
  const category=document.createElement('span');category.className=`category ${CATEGORY[v.category]?.className||'mid'}`;category.textContent=CATEGORY[v.category]?.label||'Discovery';
  const fresh=document.createElement('span');fresh.className='new-badge';fresh.textContent=v.is_demo?'PREVIEW ITEM':'YOUTUBE RESULT';const adapted=document.createElement('span');adapted.className='adapted-badge';adapted.textContent='RE-RANKED';adapted.hidden=true;categoryRow.append(category,fresh,adapted);
  const title=document.createElement(v.url?'a':'div');title.className='video-title';title.textContent=v.title;
  if(v.url){title.href=v.url;title.target='_blank';title.rel='noopener noreferrer';title.title='Open video on YouTube';}
  const channelRow=document.createElement('div');channelRow.className='channel';
  const avatar=document.createElement('span');avatar.className='channel-avatar';avatar.textContent=(v.channel||'C').trim().charAt(0).toUpperCase();
  const channel=document.createElement('span');channel.textContent=v.channel||'Unknown channel';channelRow.append(avatar,channel);
  const bottom=document.createElement('div');bottom.className='card-bottom';
  const meta=document.createElement('div');meta.className='video-meta';
  const views=document.createElement('span');views.textContent=`◉ ${nf.format(v.views||0)} views`;
  const detail=document.createElement('span');detail.textContent=v.is_demo?'Sample content':'YouTube video';meta.append(views,detail);
  bottom.append(meta);
  if(v.url){const watch=document.createElement('a');watch.className='watch-link';watch.href=v.url;watch.target='_blank';watch.rel='noopener noreferrer';watch.textContent='Watch on YouTube ↗';bottom.append(watch);}
  else{const noLink=document.createElement('span');noLink.className='no-link';noLink.textContent='Preview only';bottom.append(noLink);}
  main.append(categoryRow,title,channelRow,bottom);article.append(thumbWrap,main);return article;
}
function getAdaptiveWindow(){
  const cards = Array.from(el('feed').querySelectorAll('.video-card'));
  const feedRect = el('feed').getBoundingClientRect();
  const feedBottom = feedRect.top + el('feed').clientHeight;
  const safeLine = feedBottom + Math.max(90, el('feed').clientHeight * 0.25);
  const adaptive = cards.filter(card => {
    const id = card.dataset.videoId;
    return id && !state.viewed.has(id) && card.getBoundingClientRect().top > safeLine;
  });
  let windowCards = adaptive.slice(0, 10);
  if (windowCards.length < 10) {
    const selectedIds = new Set(windowCards.map(card => card.dataset.videoId));
    const fallback = cards.filter(card => {
      const id = card.dataset.videoId;
      return id && !state.viewed.has(id) && !selectedIds.has(id)
        && card.getBoundingClientRect().bottom > feedRect.top + 20;
    });
    windowCards = windowCards.concat(fallback).slice(0, 10);
  }
  const videoById = new Map(VIDEOS.map(video => [video.video_id, video]));
  return windowCards.map(card => videoById.get(card.dataset.videoId)).filter(Boolean);
}
function mixCounts(videos){
  const counts={head:0,mid:0,tail:0};
  videos.forEach(video=>{if(counts[video.category]!==undefined)counts[video.category]++;});
  return counts;
}
function mixText(counts){return `Popular ${counts.head} · Mid-tail ${counts.mid} · Long-tail ${counts.tail}`;}
function compactMix(counts){return `P${counts.head} / M${counts.mid} / L${counts.tail}`;}
function updateCounters(){
  if(state.isReordering) return;
  const total = el('feed').querySelectorAll('.video-card').length;
  const mixVideos = getAdaptiveWindow();
  const counts = mixCounts(mixVideos);
  el('shownCount').textContent = String(total);
  el('loadedCount').textContent = String(VIDEOS.length);
  el('viewedCount').textContent = String(state.viewed.size);
  el('mixWindowCount').textContent = `${mixVideos.length} of 10`;
  const pct = key => mixVideos.length ? Math.round(counts[key] / mixVideos.length * 100) : 0;
  el('numHead').textContent = String(counts.head);
  el('numMid').textContent = String(counts.mid);
  el('numTail').textContent = String(counts.tail);
  el('barHead').style.width = pct('head') + '%';
  el('barMid').style.width = pct('mid') + '%';
  el('barTail').style.width = pct('tail') + '%';
}
function reorderEntireFeed(modeName, onSettled=()=>{}){
  // If another mode change arrives mid-animation, finish the previous FLIP first.
  // This prevents overlapping transforms and keeps ranking transitions deterministic.
  if(activeReorderCleanup) activeReorderCleanup();

  const feed = el('feed');
  const cards = Array.from(feed.querySelectorAll('.video-card'));
  if(cards.length < 2){ window.setTimeout(onSettled, 0); return {moved:0, ids:new Set()}; }
  // Keep the one-time entrance animation from restarting when existing cards move.
  feed.classList.add('has-rendered');

  const beforeOrder = cards.map(card => card.dataset.videoId);
  const beforeIndex = new Map(beforeOrder.map((id, index) => [id, index]));
  const beforeTops = new Map(cards.map(card => [card.dataset.videoId, card.getBoundingClientRect().top]));
  const elementById = new Map(cards.map(card => [card.dataset.videoId, card]));

  // Unlike the previous implementation, ranking uses every result. Viewed,
  // currently visible and near-reader cards are all eligible to change position.
  const rankedVideos = rankAllForMode(modeName, VIDEOS);
  const orderedCards = rankedVideos.map(video => elementById.get(video.video_id)).filter(Boolean);
  const orderedIds = new Set(orderedCards.map(card => card.dataset.videoId));
  // Defensive fallback: never accidentally drop a DOM card if the payload and DOM diverge.
  cards.forEach(card => { if(!orderedIds.has(card.dataset.videoId)) orderedCards.push(card); });

  // Pause visibility observation during motion. Otherwise a card that merely flies
  // through the viewport can be incorrectly counted as viewed before it settles.
  if(cardObserver) cards.forEach(card => cardObserver.unobserve(card));
  state.isReordering = true;

  // Append existing nodes in ranked order. No cards are recreated, so loaded images,
  // links, the viewed-ID history and other DOM state survive the re-rank.
  const fragment = document.createDocumentFragment();
  orderedCards.forEach(card => fragment.appendChild(card));
  feed.appendChild(fragment);

  const afterCards = Array.from(feed.querySelectorAll('.video-card'));
  const afterOrder = afterCards.map(card => card.dataset.videoId);
  const movedIds = new Set();
  afterOrder.forEach((id, index) => {
    if(beforeIndex.has(id) && beforeIndex.get(id) !== index) movedIds.add(id);
  });

  // FLIP: invert each moved card to its previous visual position, then animate
  // to its new ranked position. This animates viewed and near-reader cards too.
  afterCards.forEach(card => {
    const id = card.dataset.videoId;
    const oldTop = beforeTops.get(id);
    const newTop = card.getBoundingClientRect().top;
    const deltaY = typeof oldTop === 'number' ? oldTop - newTop : 0;
    const wasMoved = movedIds.has(id);
    const badge = card.querySelector('.adapted-badge');
    card.classList.toggle('is-adapted', wasMoved);
    if(badge) badge.hidden = !wasMoved;
    card.classList.remove('is-moving');
    card.style.transition = 'none';
    card.style.transform = (wasMoved && Math.abs(deltaY) > 1 && !REDUCED_MOTION)
      ? `translateY(${deltaY}px)` : '';
    if(wasMoved && !REDUCED_MOTION) card.classList.add('is-moving');
  });

  let settled = false;
  let settleTimer = null;
  const finish = () => {
    if(settled) return;
    settled = true;
    if(settleTimer) window.clearTimeout(settleTimer);
    afterCards.forEach(card => {
      card.style.transition = '';
      card.style.transform = '';
      card.classList.remove('is-moving');
    });
    state.isReordering = false;
    if(cardObserver) afterCards.forEach(card => cardObserver.observe(card));
    if(activeReorderCleanup === finish) activeReorderCleanup = null;
    updateCounters();
    onSettled();
  };
  activeReorderCleanup = finish;

  if(REDUCED_MOTION){
    // Keep callback execution asynchronous so callers have received the move count.
    requestAnimationFrame(finish);
  } else {
    // Force the inverted positions to paint before transitions are enabled.
    feed.getBoundingClientRect();
    requestAnimationFrame(() => {
      if(activeReorderCleanup !== finish) return;
      afterCards.forEach(card => {
        card.style.transition = `transform ${REORDER_DURATION}ms cubic-bezier(.2,.75,.25,1), border-color .2s, box-shadow .2s`;
        card.style.transform = '';
      });
    });
    settleTimer = window.setTimeout(finish, REORDER_DURATION + 90);
  }
  return {moved:movedIds.size, ids:movedIds};
}
function addActivity(message, kind='normal'){
  const log=el('eventLog');
  const empty=log.querySelector('.event-empty');
  if(empty)empty.remove();
  const item=document.createElement('li');
  const time=new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});
  item.textContent=`${time} · ${message}`;
  if(kind==='demo')item.style.color='#5b49c4';
  log.prepend(item);
  while(log.children.length>3)log.removeChild(log.lastElementChild);
}
function updateSpeedDisplay(speed, directionLabel){
  const rounded=Math.round(Math.max(0,speed));
  el('velocityReadout').textContent=`${rounded} px/s`;
  el('proofSpeed').innerHTML=`${rounded} <small>px/s</small>`;
  el('directionReadout').textContent=directionLabel || state.direction || '—';
  el('scrollEventCount').textContent=String(state.scrollEvents);
  el('meterFill').style.width=Math.max(8,Math.min(100,rounded/1500*100))+'%';
}
function setMode(modeName, source='Live scroll', reportedSpeed=null){
  if(!MODE[modeName])return false;
  const previousMode=state.mode;
  if(previousMode===modeName){
    updateSpeedDisplay(reportedSpeed ?? state.smoothedSpeed, source==='Guided demo'?'DEMO SIGNAL':state.direction);
    return false;
  }

  const transitionToken = ++state.transitionToken;
  // Commit any in-flight visual transition before measuring the previous mix.
  if(activeReorderCleanup) activeReorderCleanup();
  const beforeCounts = mixCounts(getAdaptiveWindow());
  state.mode = modeName;
  const mode = MODE[modeName];
  el('modeName').textContent = mode.label;
  el('modeRowTitle').textContent = mode.title;
  el('modeCopy').textContent = mode.copy;
  el('modeVelocity').textContent = modeName==='slow'?'Relaxed pace':modeName==='fast'?'Fast pace':'Steady pace';
  updateSpeedDisplay(reportedSpeed ?? state.smoothedSpeed, source==='Guided demo'?'DEMO SIGNAL':state.direction);

  const proofStatus=el('proofStatus');
  proofStatus.textContent = source==='Guided demo' ? 'DEMO RUNNING' : 'Re-ranking';
  proofStatus.classList.toggle('demo',source==='Guided demo');
  proofStatus.classList.add('attention');
  el('signalState').textContent = source==='Guided demo' ? 'Demo running' : 'Re-ranking full feed';
  el('lastMixChange').textContent = `${source} · ${MODE[previousMode].label} → ${mode.label}\nRe-ranking all ${VIDEOS.length} cards, including viewed and near-reader items…`;

  const reorder = reorderEntireFeed(modeName, () => {
    // A newer transition may have superseded this callback while it was animating.
    if(transitionToken !== state.transitionToken) return;
    const afterCounts = mixCounts(getAdaptiveWindow());
    const beforeMix = compactMix(beforeCounts), afterMix = compactMix(afterCounts);
    const changedMix = beforeMix !== afterMix;
    const movementText = `${reorder.moved} of ${VIDEOS.length} cards changed position`;
    const mixTextLine = changedMix
      ? `Next-up mix: ${beforeMix} → ${afterMix}`
      : `Next-up mix after re-ranking: ${afterMix}`;
    el('lastMixChange').textContent = `${source} · ${MODE[previousMode].label} → ${mode.label}\n${movementText}; the entire feed was eligible, including viewed cards.\n${mixTextLine}.`;
    addActivity(`${source}: ${MODE[previousMode].label} → ${mode.label}; ${reorder.moved}/${VIDEOS.length} card positions changed; ${beforeMix} → ${afterMix}`, source==='Guided demo'?'demo':'normal');
    el('signalState').textContent = source==='Guided demo' ? 'Demo running' : 'Mode updated';
    window.setTimeout(() => {
      if(transitionToken === state.transitionToken && !state.demoRunning){
        el('signalState').textContent = 'Listening';
        proofStatus.textContent = 'Listening';
        proofStatus.classList.remove('attention','demo');
      }
    }, 1000);
  });

  // The ranking has already been committed to DOM order. If all IDs happen to
  // retain their positions, still ensure counters are refreshed after the FLIP.
  if(reorder.moved === 0 && !state.isReordering) updateCounters();
  return true;
}
function modeForSpeed(pxPerSecond){
  // Hysteresis gives visibly distinct modes without rapid flicker around thresholds.
  if(state.mode==='slow'){
    if(pxPerSecond>850)return 'fast';
    if(pxPerSecond>340)return 'balanced';
    return 'slow';
  }
  if(state.mode==='fast'){
    if(pxPerSecond<190)return 'slow';
    if(pxPerSecond<560)return 'balanced';
    return 'fast';
  }
  if(pxPerSecond<230)return 'slow';
  if(pxPerSecond>720)return 'fast';
  return 'balanced';
}
function queueModeChange(candidate){
  if(candidate===state.mode){state.pendingMode=null;state.pendingSince=0;if(state.modeCommitTimer)clearTimeout(state.modeCommitTimer);return;}
  if(candidate!==state.pendingMode){
    state.pendingMode=candidate;state.pendingSince=performance.now();
    if(state.modeCommitTimer)clearTimeout(state.modeCommitTimer);
    state.modeCommitTimer=window.setTimeout(()=>{
      if(state.pendingMode===candidate && state.mode!==candidate && !state.demoRunning){
        setMode(candidate,'Live scroll',state.smoothedSpeed);
        state.lastModeChange=performance.now();
        state.pendingMode=null;state.pendingSince=0;
      }
    },260);
  }
}
function onScroll(){
  if(state.demoRunning)return;
  const now=performance.now(),y=el('feed').scrollTop,elapsed=now-state.lastTime,dy=y-state.lastY;
  const instant=elapsed>240?0:Math.min(2800,(Math.abs(dy)/Math.max(12,elapsed))*1000);
  state.smoothedSpeed=state.smoothedSpeed*0.55+instant*0.45;
  state.direction=dy===0?'—':dy>0?'Down ↓':'Up ↑';
  state.lastY=y;state.lastTime=now;state.scrollEvents++;
  updateSpeedDisplay(state.smoothedSpeed,state.direction);
  const candidate=modeForSpeed(Math.max(state.smoothedSpeed,instant*0.72));
  queueModeChange(candidate);
  el('signalState').textContent='Reading pace';
  el('proofStatus').textContent='Measuring';
  el('proofStatus').classList.remove('demo');
  if(state.scrollTimer)clearTimeout(state.scrollTimer);
  state.scrollTimer=setTimeout(()=>{
    el('signalState').textContent='Listening';
    if(!state.demoRunning){el('proofStatus').textContent='Listening';el('proofStatus').classList.remove('attention');}
  },650);
}
async function runGuidedDemo(){
  if(state.demoRunning)return;
  if(state.modeCommitTimer)clearTimeout(state.modeCommitTimer);
  state.pendingMode=null;state.pendingSince=0;
  state.demoRunning=true;
  const button=el('runDemo');button.disabled=true;button.textContent='Demo running…';
  el('proofStatus').textContent='DEMO RUNNING';el('proofStatus').classList.add('demo');
  addActivity('Guided demo started; switching through all three ranking modes.','demo');
  const sequence=[
    {mode:'slow',speed:120},
    {mode:'balanced',speed:480},
    {mode:'fast',speed:1250}
  ];
  for(const step of sequence){
    state.smoothedSpeed=step.speed;
    updateSpeedDisplay(step.speed,'DEMO SIGNAL');
    // Force each demonstration step, including a visible refresh of the starting mode.
    if(state.mode===step.mode){
      const opposite=step.mode==='slow'?'fast':'slow';
      setMode(opposite,'Guided demo',step.speed);
      await new Promise(resolve=>setTimeout(resolve,800));
    }
    setMode(step.mode,'Guided demo',step.speed);
    await new Promise(resolve=>setTimeout(resolve,1350));
  }
  state.demoRunning=false;
  if(state.modeCommitTimer)clearTimeout(state.modeCommitTimer);
  state.pendingMode=null;state.pendingSince=0;
  state.smoothedSpeed=0;state.lastY=el('feed').scrollTop;state.lastTime=performance.now();
  updateSpeedDisplay(0,'Scroll to resume live sensing');
  button.disabled=false;button.textContent='↻ Run guided proof demo again';
  el('proofStatus').textContent='Demo complete';el('proofStatus').classList.remove('attention');el('proofStatus').classList.add('demo');
  el('signalState').textContent='Listening';
  el('directionReadout').textContent='Scroll to resume live sensing';
  addActivity('Guided demo complete. Scroll the feed to see live browser measurements.', 'demo');
}
el('runDemo').addEventListener('click',runGuidedDemo);

let counterFrame = 0;
function scheduleCounterUpdate(){
  if(counterFrame) return;
  counterFrame = requestAnimationFrame(() => {
    counterFrame = 0;
    updateCounters();
  });
}
el('feed').addEventListener('scroll', () => {
  onScroll();
  scheduleCounterUpdate();
}, {passive:true});
// The main page remains fixed; only the feed and insight rail have internal scrollbars.
window.addEventListener('resize', scheduleCounterUpdate, {passive:true});
if(!VIDEOS.length){
  const empty = document.createElement('div');
  empty.className = 'empty-state';
  empty.textContent = 'No videos were returned for this search. Try another topic or refresh the feed.';
  el('feed').appendChild(empty);
} else {
  // Render the complete fetched collection on first paint: no queue, no sentinel, no Load more.
  state.shown = rankAllForMode(state.mode);
  const fragment = document.createDocumentFragment();
  state.shown.forEach((video, index) => fragment.appendChild(createCard(video, index)));
  el('feed').appendChild(fragment);
  // Let the staggered entrance animation finish once; later DOM moves use FLIP only.
  window.setTimeout(() => el('feed').classList.add('has-rendered'), 520);
  updateCounters();
  const initialCounts=mixCounts(getAdaptiveWindow());
  el('lastMixChange').textContent=`Baseline captured · ${VIDEOS.length} videos rendered.\nInitial next-up mix: ${compactMix(initialCounts)}. Scroll to trigger automatic re-ranking, or use the guided demo to show all modes.`;
  cardObserver = new IntersectionObserver(entries => {
    let changed = false;
    entries.forEach(entry => {
      const id = entry.target.dataset.videoId;
      if(entry.isIntersecting && id && !state.viewed.has(id)){
        state.viewed.add(id);
        changed = true;
      }
    });
    if(changed)updateCounters();
  }, {root:el('feed'), rootMargin:'0px 0px -8% 0px', threshold:0.15});
  el('feed').querySelectorAll('.video-card').forEach(card => cardObserver.observe(card));
}
</script>
</body>
</html>'''.replace("__VIDEO_DATA__", payload).replace("__QUERY_DATA__", query_json).replace("__IS_DEMO__", data_mode_json)


# -----------------------------------------------------------------------------
# STREAMLIT SIDEBAR: CONFIGURATION ONLY. NO SCROLL-VELOCITY SLIDER.
# -----------------------------------------------------------------------------
st.sidebar.title("PulseRank")
st.sidebar.caption("Adaptive video discovery")
st.sidebar.markdown("---")
st.sidebar.subheader("Feed settings")

api_key_from_env = get_secret_api_key()
if api_key_from_env:
    api_key = api_key_from_env
    st.sidebar.success("YouTube API key loaded from server configuration.")
else:
    api_key = st.sidebar.text_input(
        "YouTube Data API key",
        type="password",
        help="Used only to request video metadata. For deployment, configure YOUTUBE_API_KEY as a secret/environment variable.",
    ).strip()

query = st.sidebar.text_input(
    "Search topic",
    value=st.session_state.get("pulserank_query", "Tech Reviews and AI"),
    placeholder="e.g. AI engineering, travel, music",
    key="pulserank_query_input",
)
use_demo_data = st.sidebar.checkbox(
    "Enable preview data if live results are unavailable",
    value=False,
    help="Preview entries are synthetic and will be clearly labelled. They are not real YouTube videos.",
)
refresh = st.sidebar.button("↻ Refresh feed", use_container_width=True)
if refresh:
    fetch_youtube_feed.clear()

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="pr-sidebar-note"><b>Adaptive by design</b><br>All retrieved videos render immediately with no queue. Scroll pace is measured in the browser and re-ranks the full feed, including viewed and near-reader cards, using smooth position transitions.</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<div class="pr-sidebar-note" style="margin-top:12px">For deployment, store your API key in server-side secrets and configure quota/rate limits. Do not commit <code>.env</code> to GitHub.</div>',
    unsafe_allow_html=True,
)

if not query.strip():
    st.info("Enter a search topic in the sidebar to load videos.")
    st.stop()

with st.spinner("Loading video feed…"):
    records, fetch_error = fetch_youtube_feed(api_key, query.strip(), MAX_RESULTS) if api_key else ([], "Enter a YouTube Data API key in the sidebar, or configure YOUTUBE_API_KEY in your environment.")

is_demo = False
if not records and use_demo_data:
    records = make_demo_videos(query.strip(), MAX_RESULTS)
    is_demo = True
    if fetch_error:
        st.sidebar.warning("Live results unavailable. Showing explicitly labelled preview data.")
elif not records:
    st.title("PulseRank")
    if api_key and not fetch_error:
        st.subheader("No matching videos found")
        st.info("YouTube returned no videos for this search. Try a broader topic or refresh the feed.")
    else:
        st.subheader("Connect a live video source")
        if fetch_error:
            st.warning(fetch_error)
        st.markdown(
            """
            To load real videos, configure a YouTube Data API key:

            - **Local:** add `YOUTUBE_API_KEY=your_key` to a `.env` file.
            - **Streamlit Community Cloud:** add `YOUTUBE_API_KEY` under the app's Secrets settings.
            - Or enter the key in the sidebar for a local test session.

            Enable **preview data** in the sidebar to inspect the interface without a live API key.
            """
        )
    st.stop()

if fetch_error and not is_demo:
    st.sidebar.error(fetch_error)
if not records:
    st.info("No videos were returned for this query. Try a broader topic or refresh the feed.")
    st.stop()

prepared = enrich_videos(records, query.strip())
if not prepared:
    st.warning("The feed could not be prepared. Please refresh and try again.")
    st.stop()

frontend_html = build_frontend(prepared, query.strip(), is_demo)
components.html(frontend_html, height=760, scrolling=False)
