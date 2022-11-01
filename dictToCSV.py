import pandas as pd
from Symmetry import Dict
from projectionHistogram import projectionHistogram

projectionHistogram()

df = pd.DataFrame(Dict)
df.to_csv("features.csv",)