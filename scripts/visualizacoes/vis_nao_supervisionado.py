import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scripts.io_utils import load_siasus_tratado

def dashboard_nao_supervisionado():
    df = load_siasus_tratado()

    # Agregação por município
    df_mun = (
        df.groupby("ufmun")
        .agg(
            total_atendimentos=("pa_qtdapr", "sum"),
            media_idade=("idadepac", "mean"),
            total_procedimentos=("pa_class_s", "nunique")
        )
        .reset_index()
    )

    X = df_mun[["total_atendimentos", "media_idade", "total_procedimentos"]]
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_mun["cluster"] = kmeans.fit_predict(X_scaled)

    # ===============================
    # DASHBOARD
    # ===============================
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Aprendizagem Não Supervisionada — Clusterização de Municípios",
        fontsize=18
    )

    # 1. Municípios por cluster
    df_mun["cluster"].value_counts().plot(
        kind="bar",
        ax=axs[0, 0]
    )
    axs[0, 0].set_title("Quantidade de municípios por cluster")
    axs[0, 0].set_xlabel("Cluster")
    axs[0, 0].set_ylabel("Municípios")

    axs[0, 0].text(
        0.02, -0.35,
        "A clusterização agrupa municípios com perfis semelhantes de atendimento,\n"
        "evidenciando polos regionais e municípios de menor demanda.",
        transform=axs[0, 0].transAxes
    )

    # 2. Média de atendimentos por cluster
    df_mun.groupby("cluster")["total_atendimentos"].mean().plot(
        kind="bar",
        ax=axs[0, 1]
    )
    axs[0, 1].set_title("Média de atendimentos por cluster")
    axs[0, 1].set_ylabel("Atendimentos")

    # 3. Média de idade por cluster
    df_mun.groupby("cluster")["media_idade"].mean().plot(
        kind="bar",
        ax=axs[1, 0]
    )
    axs[1, 0].set_title("Média de idade dos pacientes por cluster")
    axs[1, 0].set_ylabel("Idade média")

    # 4. Texto final
    axs[1, 1].axis("off")
    axs[1, 1].text(
        0, 0.6,
        "Síntese da Aprendizagem Não Supervisionada:\n\n"
        "- Identificação de perfis territoriais distintos.\n"
        "- Existência de municípios polos com alta concentração de atendimentos.\n"
        "- Presença de outlier como fenômeno real do sistema de saúde.\n\n"
        "A clusterização mostrou-se adequada para análise exploratória\n"
        "e apoio à compreensão da organização regional dos serviços.",
        fontsize=11
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    dashboard_nao_supervisionado()