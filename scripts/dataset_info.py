from pathlib import Path
import pandas as pd
from scripts.io_utils import load_siasus_tratado

df = load_siasus_tratado()

def mostrar_info_dataset():
    print("\n===== INFORMAÇÕES DO DATASET =====\n")
    print(f"Número de registros: {df.shape[0]:,}")
    print(f"Número de colunas: {df.shape[1]}\n")

    print("Colunas disponíveis:")
    for col in df.columns:
        print(f"- {col}")

    print("\nContexto do dataset:")
    print(
        "Base administrativa do SIASUS Psicossocial, contendo registros de "
        "procedimentos ambulatoriais relacionados à atenção em saúde mental."
    )

    print("\nObjetivos de uso:")
    print(
        "- Analisar padrões de atendimento psicossocial\n"
        "- Comparar municípios da região Oeste do Paraná\n"
        "- Apoiar decisões de gestão pública\n"
        "- Explorar técnicas de Ciência de Dados e Machine Learning"
    )
