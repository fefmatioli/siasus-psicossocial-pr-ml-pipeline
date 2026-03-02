import pandas as pd

# Ler base já consolidada
df = pd.read_csv("siasus_consolidado.csv", sep=";", low_memory=False)

print("Antes da limpeza:", df.shape)

# Remover colunas totalmente vazias
df = df.dropna(axis=1, how="all")

# Padronizar nomes das colunas
df.columns = df.columns.str.lower().str.strip()

# Remover registros sem município (se existir a coluna)
if "munic_res" in df.columns:
    df = df[df["munic_res"].notna()]

print("Depois da limpeza:", df.shape)

# Salvar base tratada
df.to_csv("siasus_tratado.csv", index=False, sep=";")

print("Base tratada salva.")
