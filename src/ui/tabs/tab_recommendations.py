import numpy as np
import streamlit as st
import pandas as pd

from src.ui.tabs.tab_rate import render_poster
from src.config.config import config


def render_tab_recommendations(
    model, movies_df: pd.DataFrame, title_to_poster: dict
) -> None:
    """
    Render the "Your Recommendations" tab where users can see personalized movie recommendations based on their ratings.
    Args:
        model: The recommendation model with a recommend_from_ratings method.
        movies_df (pd.DataFrame): The DataFrame containing movie information.
        title_to_poster (dict): Mapping of movie titles to their poster URLs for rendering.
    """
    st.header("Your Recommendations")
    st.caption("Based on films you liked in the **Rate Movies** tab.")

    if not st.session_state.user_ratings:
        st.info(
            "Rate at least one movie in the **Rate Movies** tab to get recommendations."
        )
        return

    n_recs_default = config.ui_config.get("n_recommendations", 10)
    n_recs = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=n_recs_default,
        step=1,
    )

    if st.button("Get Recommendations", type="primary", width="stretch"):
        with st.spinner("Finding movies you'll love..."):
            recs = model.recommend_from_ratings(
                user_ratings=st.session_state.user_ratings,
                n=n_recs,
            )

        if not recs:
            st.warning("Could not generate recommendations. Try rating more movies.")
            return

        metric = getattr(model, "metric", "cosine")

        if metric == "cosine":
            # Cosine scores are in [-1, 1] - clamp to [0, 1] for the progress bar;
            # negative values (very dissimilar) are just shown as 0%.
            for r in recs:
                r["display_score"] = float(np.clip(r["score"], 0.0, 1.0))
        else:
            # Euclidean scores are negative squared distances (≤ 0).
            # Map the full range to [0, 1] via min-max normalisation.
            raw_scores = np.array([r["score"] for r in recs], dtype=np.float64)
            s_min, s_max = raw_scores.min(), raw_scores.max()
            score_range = s_max - s_min if s_max != s_min else 1.0
            for r in recs:
                r["display_score"] = float((r["score"] - s_min) / score_range)

        # Score explanation expander
        with st.expander("ℹ️ How are match scores calculated?"):
            if metric == "cosine":
                st.markdown(
                    """
                    **Metric: Cosine Similarity**

                    Your taste profile is built by averaging the feature vectors of your rated movies,
                    weighted by your ratings. Each candidate movie is then scored by the **cosine similarity**
                    between its feature vector and your profile vector.

                    - Score range: **−1 to 1** (only values > 0 are shown as a non-zero match %).
                    - **1.0** means the movie is perfectly aligned with your profile.
                    - **0.0** means no relationship.
                    - **Negative** values indicate the movie is opposite to your taste profile.

                    The percentage bar shows the raw cosine score clipped to [0 %, 100 %] - no rescaling applied.

                    > ⚠️ **Why are some scores low?**
                    > Low ratings (e.g. 1–2 ★) **subtract** from your taste profile — they actively push it
                    > *away* from movies with similar features. This is intentional: if you disliked an
                    > action-heavy movie, action movies in general will score lower. However, if most of your
                    > ratings are low, your profile may become dominated by *negative* preferences, making it
                    > harder to find strong positive matches. Try rating a few movies you genuinely enjoyed
                    > to anchor your profile in a positive direction.

                    > 📊 **The more diverse your ratings, the lower the scores may appear.**
                    > If you have rated movies across many different genres and styles, your taste profile
                    > becomes a broad average — and no single movie will align with it perfectly. This is
                    > completely normal. **Match scores are additional context, not the most important signal** —
                    > focus on the movie titles and descriptions rather than chasing high percentages.
                    """
                )
            else:
                st.markdown(
                    """
                    **Metric: Euclidean Distance**

                    Your taste profile is built by averaging the feature vectors of your rated movies,
                    weighted by your ratings. Each candidate movie is scored as the **negative squared
                    Euclidean distance** from that movie's feature vector to your profile - so higher (less
                    negative) means closer.

                    - Raw scores are always **≤ 0**; the closest movie has the highest (least negative) score.
                    - For display, scores are mapped to **0 – 100 %** using min-max normalisation over
                      the returned result set.
                    - **100 %** = the closest movie to your profile in this result set.
                    - **0 %** = the furthest movie in this result set.

                    > ⚠️ **Why are some scores low?**
                    > Low ratings pull your taste profile toward disliked content, increasing the distance
                    > to movies you might actually enjoy. If recommendations seem off, try adding more
                    > highly-rated movies to better define your positive preferences.

                    > 📊 **The more diverse your ratings, the lower the scores may appear.**
                    > Rating movies across many different genres spreads your taste profile in all directions,
                    > increasing the average distance to any individual movie. This is completely normal.
                    > **Match scores are additional context, not the most important signal** — the
                    > recommendations themselves are what matter most.
                    """
                )

        # Top 3 featured cards
        st.subheader("🏆 Top Picks For You")
        top3_cols = st.columns(3)

        for i, (col, rec) in enumerate(zip(top3_cols, recs[:3])):
            with col:
                render_poster(
                    title_to_poster.get(rec["title"]), rec["title"], width=200
                )
                st.markdown(f"**#{i + 1} {rec['title']}**")

                row = movies_df[movies_df["title"] == rec["title"]]
                if not row.empty and "overview" in row.columns:
                    overview = row.iloc[0].get("overview")
                    if overview and isinstance(overview, str):
                        st.caption(
                            overview[:200] + "..." if len(overview) > 200 else overview
                        )

                score_pct = int(rec["display_score"] * 100)
                st.progress(rec["display_score"], text=f"Match: {score_pct}%")

        # Remaining recommendations
        if len(recs) > 3:
            st.divider()
            st.subheader("More Recommendations")
            cols_per_row = 5
            remaining = recs[3:]

            for row_start in range(0, len(remaining), cols_per_row):
                row_items = remaining[row_start : row_start + cols_per_row]
                cols = st.columns(cols_per_row)

                for col, rec in zip(cols, row_items):
                    with col:
                        render_poster(
                            title_to_poster.get(rec["title"]), rec["title"], width=120
                        )
                        st.caption(f"**{rec['title']}**")
                        st.caption(f"Match: {int(rec['display_score'] * 100)}%")
