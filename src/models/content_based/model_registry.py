from src.models.content_based import (
    CB1Genome,
    CB2GenresYear,
    CB3GenomeGenres,
    CB4GenomeTags,
    CB5OverviewTFIDF,
    CB6OverviewSBERT,
    CB7TMDBStructural,
    CB8Full,
    CB9GenomeTMDB,
)

MODEL_REGISTRY_STRINGS = {
    "cb1": ("cb1", "CB1_Genome"),
    "cb2": ("cb2", "CB2_GenresYear"),
    "cb3": ("cb3", "CB3_GenomeGenres"),
    "cb4": ("cb4", "CB4_GenomeTags"),
    "cb5": ("cb5", "CB5_OverviewTFIDF"),
    "cb6": ("cb6", "CB6_OverviewSBERT"),
    "cb7": ("cb7", "CB7_TMDBStructural"),
    "cb8": ("cb8", "CB8_Full"),
    "cb9": ("cb9", "CB9_GenomeTMDB"),
}

MODEL_REGISTRY = {
    "cb1": (CB1Genome, True),
    "cb2": (CB2GenresYear, False),
    "cb3": (CB3GenomeGenres, True),
    "cb4": (CB4GenomeTags, True),
    "cb5": (CB5OverviewTFIDF, False),
    "cb6": (CB6OverviewSBERT, False),
    "cb7": (CB7TMDBStructural, False),
    "cb8": (CB8Full, True),
    "cb9": (CB9GenomeTMDB, True),
}
