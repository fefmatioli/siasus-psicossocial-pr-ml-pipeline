import pandas as pd
from scripts.io_utils import load_siasus_tratado

df = load_siasus_tratado()

print("=== EDA DEPOIS DO TRATAMENTO ===")
print("Dimensão:", df.shape)
print("\nTipos de dados:")
print(df.dtypes)

print("\nValores nulos por coluna:")
print(df.isna().sum().sort_values(ascending=False).head(10))

print("\nResumo estatístico:")
print(df.describe(include="all"))