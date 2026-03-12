import pandas as pd


def _col_widths(headers: list[str], min_width: int = 8) -> list[int]:
    """
    Compute column widths as the maximum of the header length and min_width.
    Args:
        headers (list[str]): Column header names.
        min_width (int): Minimum column width in characters.
    Returns:
        list[int]: Width for each column.
    """
    return [max(len(h), min_width) for h in headers]


def markdown_sep(col_widths: list[int]) -> str:
    """
    Build a Markdown table separator row from column widths.
    Args:
        col_widths (list[int]): Width of each column.
    Returns:
        str: Separator row string, e.g. '| -------- | -------- |'.
    """
    return "| " + " | ".join("-" * w for w in col_widths) + " |"


def markdown_row(values: list, col_widths: list[int]) -> str:
    """
    Build a single Markdown table row, left-padding each cell to its column width.
    Args:
        values (list): Cell values for this row (will be stringified).
        col_widths (list[int]): Width of each column.
    Returns:
        str: Formatted row string, e.g. '| cb1      | 62411    |'.
    """
    cells = [str(v).ljust(w) for v, w in zip(values, col_widths)]
    return "| " + " | ".join(cells) + " |"


def render_markdown_table(
    display: pd.DataFrame,
    min_width: int = 8,
) -> str:
    """
    Render a DataFrame as a Markdown table string.
    The DataFrame is expected to already be formatted (numeric columns
    pre-rounded / stringified as desired by the caller).
    Args:
        display (pd.DataFrame): Fully prepared DataFrame — index is reset,
            columns are in the desired display order, values are strings or
            ints ready to render.
        min_width (int): Minimum column width in characters.
    Returns:
        str: Multi-line Markdown table string.
    """
    headers = list(display.columns)
    widths = _col_widths(headers, min_width=min_width)

    lines = [
        markdown_row(headers, widths),
        markdown_sep(widths),
        *[markdown_row(list(row), widths) for _, row in display.iterrows()],
    ]
    return "\n".join(lines)