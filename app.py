import streamlit as st
import pandas as pd
from pathlib import Path

from src.models.base_model import BaseModel
from src.models.content_based.cb4_genome_tags import CB4GenomeTags
from src.ui.tabs.tab_rate import render_tab_rate
from src.ui.tabs.tab_recommendations import render_tab_recommendations
from src.ui.tabs.tab_similar import render_tab_similar
from src.ui.tabs.tab_insights import render_tab_insights

from src.config.config import config
from src.logging_utils.logger import logger

# Config
PROCESSED_DIR = config.ui_config.get("dataset_path", "data/processed/full")
ARTIFACTS_DIR = config.ui_config.get("artifact_path", "artifacts/full/cb4")
MODEL_NAME = config.ui_config.get("model_name", "CB4_GenomeTags")

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)


# Loading functions with caching
@st.cache_resource
def load_model() -> BaseModel:
    """
    Load the content-based recommendation model from artifacts.
    Currently loads the CB4GenomeTags model, which uses TMDB genome features.
    Returns:
        BaseModel: The loaded recommendation model.
    """
    logger.info(f"Loading {MODEL_NAME} model from artifacts...")
    return CB4GenomeTags.load(Path(ARTIFACTS_DIR), MODEL_NAME)


@st.cache_resource
def load_movies() -> pd.DataFrame:
    """
    Load the enriched movies DataFrame from processed data.
    Returns:
        pd.DataFrame: The enriched movies DataFrame.
    """
    df = pd.read_parquet(Path(PROCESSED_DIR) / "movies_enriched.parquet")
    return df.dropna(subset=["movie_idx"]).copy()


# Load model and data
model = load_model()
movies_df = load_movies()

# Title lists for search
all_titles = sorted(movies_df["title"].dropna().tolist())
title_to_poster = dict(
    zip(movies_df["title"], movies_df.get("poster_url", pd.Series()))
)

# Session state for user ratings
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}  # {title: rating}

# Tabs
st.title("🎬 Movie Recommender")

tab1, tab2, tab3, tab4 = st.tabs(
    ["⭐ Rate Movies", "🎯 Recommendations", "🔍 Similar Movies", "🧠 Taste Profile"]
)

with tab1:
    render_tab_rate(all_titles, title_to_poster)

with tab2:
    render_tab_recommendations(model, movies_df, title_to_poster)

with tab3:
    render_tab_similar(model, movies_df, all_titles, title_to_poster)

with tab4:
    render_tab_insights(movies_df)
