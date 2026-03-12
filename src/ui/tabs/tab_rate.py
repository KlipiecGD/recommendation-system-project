import streamlit as st
import pandas as pd

from src.ui.helpers import render_stars, render_poster


def render_tab_rate(all_titles: list[str], title_to_poster: dict) -> None:
    """
    Render the "Rate Movies" tab where users can search for movies, rate them, and see their rated movies.
    Args:
        all_titles (list[str]): List of all movie titles for the search dropdown.
        title_to_poster (dict): Mapping of movie titles to their poster URLs for rendering.
    """
    st.header("Rate Movies")
    st.caption("Search for a movie, give it a rating, and build your taste profile.")

    col_search, col_rating, col_btn = st.columns([3, 1, 1])

    with col_search:
        selected_title = st.selectbox(
            "Search movie title",
            options=[""] + all_titles,
            index=0,
            placeholder="Start typing...",
        )

    with col_rating:
        selected_rating = st.selectbox(
            "Rating",
            options=[x / 2 for x in range(1, 11)],
            index=9,  # Default to 5.0 stars
            format_func=lambda x: f"{'⭐' * int(x)}{'½' if x % 1 else ''} ({x})",
        )

    with col_btn:
        st.write("")
        st.write("")
        if st.button("Add Rating", width="stretch"):
            if selected_title:
                st.session_state.user_ratings[selected_title] = selected_rating
                st.success(f"Rated '{selected_title}' {render_stars(selected_rating)}")
            else:
                st.warning("Please select a movie first.")

    if st.session_state.user_ratings:
        st.divider()
        st.subheader(f"Your Ratings ({len(st.session_state.user_ratings)} movies)")

        cols_per_row = 5
        rated_items = list(st.session_state.user_ratings.items())

        for row_start in range(0, len(rated_items), cols_per_row):
            row_items = rated_items[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)

            for col, (title, rating) in zip(cols, row_items):
                with col:
                    render_poster(title_to_poster.get(title), title, width=120)
                    st.caption(f"**{title}**")
                    st.caption(render_stars(rating))
                    if st.button("Remove", key=f"remove_{title}"):
                        del st.session_state.user_ratings[title]
                        st.rerun()
    else:
        st.info(
            "No movies rated yet. Search and rate some movies above to get started."
        )
