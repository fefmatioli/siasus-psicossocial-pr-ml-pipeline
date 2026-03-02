import pandas as pd
import os

pasta_csv = "siasus_csv"

arquivos = [
    os.path.join(pasta_csv, f)
    for f in os.listdir(pasta_csv)
    if f.endswith(".csv")
]

dfs = []

for arq in arquivos:
    print(f"Lendo {arq}")
    df = pd.read_csv(arq, sep=";", low_memory=False)
    df["arquivo_origem"] = os.path.basename(arq)  # rastreabilidade
    dfs.append(df)

dados = pd.concat(dfs, ignore_index=True)

print("Dataset final:", dados.shape)

dados.to_csv("siasus_consolidado.csv", index=False, sep=";")