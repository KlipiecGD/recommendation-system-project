import streamlit as st
import pandas as pd

from src.ui.helpers import render_poster
from src.config.config import config


def render_tab_similar(
    model, movies_df: pd.DataFrame, all_titles: list[str], title_to_poster: dict
) -> None:
    """
    Render the "Find Similar Movies" tab where users can select a movie and see the most content-similar films.
    Args:
        model: The recommendation model with a similar_movies method.
        movies_df (pd.DataFrame): The DataFrame containing movie information.
        all_titles (list[str]): List of all movie titles for the search dropdown.
        title_to_poster (dict): Mapping of movie titles to their poster URLs for rendering.
    """
    st.header("Find Similar Movies")
    st.caption("Select any movie to find the most content-similar films.")

    col_title, col_n = st.columns([3, 1])

    with col_title:
        query_title = st.selectbox(
            "Select a movie",
            options=[""] + all_titles,
            index=0,
            key="similar_search",
            placeholder="Start typing...",
        )

    with col_n:
        default_n_similar = config.ui_config.get("n_similar", 10)
        n_similar = st.slider(
            "How many", min_value=3, max_value=20, value=default_n_similar, step=1
        )

    if not query_title:
        st.info("Select a movie above to find similar titles.")
        return

    similar = model.similar_movies(query_title, n=n_similar)

    if not similar:
        st.warning(f"Could not find similar movies for '{query_title}'.")
        return

    st.divider()
    q_col1, q_col2 = st.columns([1, 4])

    with q_col1:
        render_poster(title_to_poster.get(query_title), query_title, width=150)

    with q_col2:
        st.subheader(f"Movies similar to **{query_title}**")
        row = movies_df[movies_df["title"] == query_title]
        if not row.empty and "overview" in row.columns:
            overview = row.iloc[0].get("overview")
            if overview and isinstance(overview, str):
                st.write(overview)

    st.divider()

    cols_per_row = 5
    for row_start in range(0, len(similar), cols_per_row):
        row_items = similar[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, rec in zip(cols, row_items):
            with col:
                render_poster(
                    title_to_poster.get(rec["title"]), rec["title"], width=130
                )
                st.caption(f"**{rec['title']}**")
                st.caption(f"Similarity: {int(rec['score'] * 100)}%")
