import numpy as np
import random

# Estados baseados nos clusters encontrados
# 0 = alta demanda
# 1 = outlier (demanda atípica)
# 2 = baixa demanda
estados = [0, 1, 2]

# Ações possíveis
# 0 = manter recursos
# 1 = aumentar recursos
# 2 = redistribuir recursos
acoes = [0, 1, 2]

# Tabela Q inicializada com zeros
Q = np.zeros((len(estados), len(acoes)))

# Parâmetros do aprendizado
alpha = 0.1      # taxa de aprendizado
gamma = 0.9      # fator de desconto
epsilon = 0.2    # exploração

# Função de recompensa
def recompensa(estado, acao):
    if estado == 0 and acao == 0:
        return 1      # alta demanda, manter recursos
    if estado == 2 and acao == 1:
        return 1      # baixa demanda, aumentar recursos
    if estado == 1 and acao == 2:
        return 1      # outlier, redistribuir
    return -1         # ação inadequada

# Treinamento do agente
episodios = 500

for _ in range(episodios):
    estado = random.choice(estados)

    if random.uniform(0, 1) < epsilon:
        acao = random.choice(acoes)
    else:
        acao = np.argmax(Q[estado])

    recompensa_recebida = recompensa(estado, acao)

    Q[estado, acao] = Q[estado, acao] + alpha * (
        recompensa_recebida + gamma * np.max(Q[estado]) - Q[estado, acao]
    )

print("Tabela Q aprendida:")
print(Q)
