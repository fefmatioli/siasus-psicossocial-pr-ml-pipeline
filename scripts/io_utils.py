import pandas as pd
from scripts.config import SIASUS_TRATADO_CSV

def load_siasus_tratado() -> pd.DataFrame:
    """
    Carrega o dataset tratado do SIASUS (CSV).
    Centraliza caminho, separador e encoding para evitar inconsistências.
    """
    return pd.read_csv(
        SIASUS_TRATADO_CSV,
        sep=";",
        encoding="latin1",
        low_memory=False
    )