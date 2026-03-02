import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def dashboard_reforco():
    # Tabela Q (mesma lógica do modelo)
    Q = np.array([
        [7.4, 3.1, 2.9],   # Alta demanda
        [2.4, 2.7, 7.3],   # Outlier
        [2.4, 7.8, 2.8]    # Baixa demanda
    ])

    estados = ["Alta demanda", "Outlier", "Baixa demanda"]
    acoes = ["Manter recursos", "Aumentar recursos", "Redistribuir recursos"]

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Aprendizagem por Reforço — Simulação de Apoio à Decisão",
        fontsize=18
    )

    # 1. Heatmap da Tabela Q
    sns.heatmap(
        Q,
        annot=True,
        fmt=".2f",
        xticklabels=acoes,
        yticklabels=estados,
        ax=axs[0],
        cmap="Blues"
    )
    axs[0].set_title("Tabela Q — Estado x Ação")

    # 2. Política ótima
    politica = Q.argmax(axis=1)
    axs[1].bar(estados, politica)
    axs[1].set_title("Ação ótima aprendida por estado")
    axs[1].set_ylabel("Índice da ação")

    axs[1].text(
        0.02, -0.35,
        "O agente aprende ações distintas conforme o perfil do município,\n"
        "reforçando o uso da aprendizagem por reforço como modelo normativo\n"
        "de apoio à decisão, e não como ferramenta preditiva.",
        transform=axs[1].transAxes
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


if __name__ == "__main__":
    dashboard_reforco()