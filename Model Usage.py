# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
from joblib import dump, load

model = load("HousingPricePredicter.joblib")
# -

features = np.array(
    [
        [
            -0.43942006,
            3.12628155,
            -1.12165014,
            -0.27288841,
            -1.42262747,
            -0.24141041,
            -1.31238772,
            2.61111401,
            -1.0016859,
            -0.5778192,
            -0.97491834,
            0.41164221,
            -0.86091034,
        ]
    ]
)

model.predict(features)
