import pandas as pd

df = pd.read_csv("siasus_consolidado.csv", sep=";", low_memory=False)

print(df.head())
print(df.info())
print(df.describe(include="all"))