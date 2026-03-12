from surprise import SVD, SVDpp, NMF

ALGORITHM_REGISTRY = {"svd": SVD, "svdpp": SVDpp, "nmf": NMF}

DEFAULT_PARAMS = {
    "svd": {"n_factors": 100, "n_epochs": 20, "lr_all": 0.005, "reg_all": 0.02},
    "svdpp": {"n_factors": 20, "n_epochs": 20, "lr_all": 0.007, "reg_all": 0.02},
    "nmf": {"n_factors": 15, "n_epochs": 50},
}
