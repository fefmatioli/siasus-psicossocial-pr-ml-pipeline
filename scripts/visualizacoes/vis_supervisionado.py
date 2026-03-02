import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from scripts.io_utils import load_siasus_tratado

def dashboard_supervisionado():
    # ===============================
    # Leitura dos dados
    # ===============================
    df = load_siasus_tratado()
    # ===============================
    # Seleção de variáveis (as mesmas do modelo)
    # ===============================
    cols = [
        "ufmun",
        "tpups",
        "tippre",
        "gestao",
        "condic",
        "pa_qtdapr",
        "idadepac",
        "sexopac",
        "pa_class_s"
    ]
    df = df[cols]

    # ===============================
    # Codificação
    # ===============================
    for c in ["condic", "sexopac"]:
        df[c] = LabelEncoder().fit_transform(df[c])

    X = df.drop(columns=["pa_class_s"])
    y = df["pa_class_s"]

    # ===============================
    # Treino rápido do modelo
    # ===============================
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X, y)

    # ===============================
    # Layout dos gráficos
    # ===============================
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Aprendizagem Supervisionada — SIASUS Psicossocial",
        fontsize=18
    )
    
    fig.subplots_adjust(hspace=0.45)

    # -------------------------------
    # 1. Distribuição da variável alvo
    # -------------------------------
    y.value_counts().plot(
        kind="bar",
        ax=axs[0, 0]
    )
    axs[0, 0].set_title("Distribuição da variável alvo (pa_class_s)")
    axs[0, 0].set_xlabel("Classe do procedimento")
    axs[0, 0].set_ylabel("Frequência")

    axs[0, 0].text(
        0.02, -0.45,
        "Observa-se forte desbalanceamento entre as classes, característica comum\n"
        "em bases administrativas do SUS, o que limita aplicações preditivas clássicas.",
        transform=axs[0, 0].transAxes
    )

    # -------------------------------
    # 2. Importância das variáveis
    # -------------------------------
    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    importances.plot(
        kind="bar",
        ax=axs[0, 1]
    )
    axs[0, 1].set_title("Importância das variáveis (Árvore de Decisão)")
    axs[0, 1].set_ylabel("Importância")

    axs[0, 1].text(
        0.02, -0.45,
        "Poucas variáveis concentram a maior parte da decisão do modelo,\n"
        "indicando padrões administrativos simples aprendidos pelo algoritmo.",
        transform=axs[0, 1].transAxes
    )

    # -------------------------------
    # 3. Distribuição etária
    # -------------------------------
    df["idadepac"].hist(bins=30, ax=axs[1, 0])
    axs[1, 0].set_title("Distribuição etária dos atendimentos")
    axs[1, 0].set_xlabel("Idade")
    axs[1, 0].set_ylabel("Frequência")

    axs[1, 0].text(
        0.02, -0.45,
        "A distribuição etária ajuda a compreender o perfil geral da demanda\n"
        "psicossocial, complementando a análise supervisionada.",
        transform=axs[1, 0].transAxes
    )

    # -------------------------------
    # 4. Texto final interpretativo
    # -------------------------------
    axs[1, 1].axis("off")
    axs[1, 1].text(
        0, 0.5,
        "Síntese da Aprendizagem Supervisionada:\n\n"
        "- O dataset apresenta forte desbalanceamento da variável alvo.\n"
        "- O modelo supervisionado aprende padrões administrativos simples.\n"
        "- O maior valor está na análise crítica das limitações do uso preditivo.\n\n"
        "Este resultado reforça que, para o SIASUS Psicossocial,\n"
        "a aprendizagem supervisionada deve ser utilizada com cautela,\n"
        "priorizando interpretação e não performance.",
        fontsize=11
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.0)
    plt.show()
