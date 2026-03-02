import os
import pandas as pd
from dbfread import DBF

pasta_dbf = "siasus_dbf"
pasta_csv = "siasus_csv"

os.makedirs(pasta_csv, exist_ok=True)

arquivos = [f for f in os.listdir(pasta_dbf) if f.lower().endswith(".dbf")]

for arq in arquivos:
    caminho_dbf = os.path.join(pasta_dbf, arq)
    print(f"Lendo {arq}...")

    tabela = DBF(caminho_dbf, encoding="latin1")
    df = pd.DataFrame(iter(tabela))

    nome_csv = arq.replace(".dbf", ".csv")
    caminho_csv = os.path.join(pasta_csv, nome_csv)

    df.to_csv(caminho_csv, index=False, sep=";")
    print(f"Salvo {nome_csv}")

print("Conversão concluída.")