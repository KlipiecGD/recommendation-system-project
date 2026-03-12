import streamlit as st
import pandas as pd
import ast
import numpy as np
from collections import Counter
import plotly.express as px

from src.config.config import config
from src.logging_utils.logger import logger

THRESHOLD_RATING = config.ui_config.get("insights_threshold_rating", 4.0)


def _safe_to_list(value: object) -> list:
    """
    Safely convert a value to a list.
    Args:
        value (object): The value to convert, which can be a list, a string representation of a list, or a single string.
    Returns:
        list: A list of items extracted from the input value.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                pass
        return [stripped] if stripped else []
    return []


def render_tab_insights(movies_df: pd.DataFrame) -> None:
    """
    Render the "Insights" tab, which analyzes the user's highly-rated movies to extract common genres, directors, actors, and themes.
    Args:
        movies_df (pd.DataFrame): The enriched movies DataFrame containing metadata and user tags.
    """
    st.header("🧠 Your Taste Profile")
    st.caption("What do your favorite movies have in common?")

    if not st.session_state.user_ratings:
        st.info(
            "Rate some movies in the **Rate Movies** tab to generate your taste profile."
        )
        return

    liked_titles = [
        title
        for title, rating in st.session_state.user_ratings.items()
        if rating >= THRESHOLD_RATING
    ]

    if not liked_titles:
        st.warning(
            f"You haven't rated any movies {THRESHOLD_RATING} or higher yet. Rate more movies to generate insights!"
        )
        return

    liked_df = movies_df[movies_df["title"].isin(liked_titles)]

    # Extract Genres
    all_genres = []
    for genres in liked_df["genres_list"].dropna():
        all_genres.extend(_safe_to_list(genres))

    # Extract Cast
    all_cast = []
    for cast in liked_df["cast"].dropna():
        all_cast.extend(_safe_to_list(cast))

    # Extract Directors
    directors = liked_df["director"].dropna().tolist()

    # Extract Tags
    all_tags = []
    for tags_text in liked_df["user_tags_text"].dropna():
        if isinstance(tags_text, str):
            words = [w.lower() for w in tags_text.split() if len(w) > 4]
            all_tags.extend(words)

    # Build the UI
    col1, col2 = st.columns(2, vertical_alignment="top")

    with col1:
        st.subheader("🎭 Top Genres")
        if all_genres:
            genre_counts = (
                pd.Series(Counter(all_genres))
                .sort_values(ascending=False)
                .head(5)
                .reset_index()
            )
            genre_counts.columns = ["Genre", "Count"]
            fig = px.bar(
                genre_counts,
                x="Genre",
                y="Count",
                category_orders={"Genre": genre_counts["Genre"].tolist()},
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.write("Not enough genre data.")

    with col2:
        st.subheader("🏷️ Common Themes (Tags)")
        if all_tags:
            tag_counts = (
                pd.Series(Counter(all_tags)).sort_values(ascending=False).head(125)
            )
            filtered_tags = [
                (tag, count) for tag, count in tag_counts.items() if count > 1
            ]
            if filtered_tags:
                badges = " ".join(
                    f'<span style="background-color:#1f77b4;color:white;padding:4px 10px;'
                    f'border-radius:15px;margin:3px;display:inline-block;font-size:0.85em;">'
                    f"{tag} <b>·{count}</b></span>"
                    for tag, count in filtered_tags
                )
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.write("Rate more movies to find common themes!")
        else:
            st.write("Not enough tag data.")

    st.divider()

    col3, col4 = st.columns(2, vertical_alignment="top")

    with col3:
        st.subheader("🎬 Favorite Directors")
        if directors:
            dir_counts = (
                pd.Series(Counter(directors)).sort_values(ascending=False).head(5)
            )
            for name, count in dir_counts.items():
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:8px 12px;margin:4px 0;background-color:#1e2130;border-radius:8px;">'
                    f'<span style="font-size:0.95em;">🎬 {name}</span>'
                    f'<span style="background-color:#1f77b4;color:white;padding:2px 8px;'
                    f'border-radius:10px;font-size:0.85em;">{count} movies</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

    with col4:
        st.subheader("⭐ Most Watched Actors")
        if all_cast:
            cast_counts = (
                pd.Series(Counter(all_cast)).sort_values(ascending=False).head(5)
            )
            for name, count in cast_counts.items():
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:8px 12px;margin:4px 0;background-color:#1e2130;border-radius:8px;">'
                    f'<span style="font-size:0.95em;">⭐ {name}</span>'
                    f'<span style="background-color:#1f77b4;color:white;padding:2px 8px;'
                    f'border-radius:10px;font-size:0.85em;">{count} movies</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
