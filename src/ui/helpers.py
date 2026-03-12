import streamlit as st


def render_stars(rating: float) -> str:
    """
    Convert a numeric rating (e.g., 4.5) into a string of star emojis.
    Args:
        rating (float): The numeric rating, typically between 0 and 5.
    Returns:
        str: A string representation of the rating using stars (e.g., "⭐⭐⭐⭐½").
    """
    full = int(rating)
    half = 1 if rating % 1 else 0
    empty = 5 - full - half
    return "⭐" * full + ("½" if half else "") + "☆" * empty


def render_poster(url: str | None, title: str, width: int = 150) -> None:
    """
    Render a movie poster image or a placeholder if the URL is invalid.
    Args:
        url (str | None): The URL of the poster image.
        title (str): The title of the movie.
        width (int): The width of the poster image.
    """
    if url and isinstance(url, str) and url.startswith("http"):
        st.image(url, width=width)
    else:
        st.markdown(
            f"<div style='width:{width}px;height:{int(width * 1.5)}px;"
            f"background:#2a2a2a;display:flex;align-items:center;"
            f"justify-content:center;border-radius:8px;color:#888;font-size:12px;"
            f"text-align:center;padding:8px'>{title}</div>",
            unsafe_allow_html=True,
        )
