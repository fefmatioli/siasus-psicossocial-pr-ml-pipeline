import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from scripts.io_utils import load_siasus_tratado


df = load_siasus_tratado()

print(df.shape)
print(df.columns)

# Seleção das colunas para o modelo supervisionado
colunas_modelo = [
    'ufmun',
    'tpups',
    'tippre',
    'gestao',
    'condic',
    'pa_qtdapr',
    'idadepac',
    'sexopac',
    'pa_class_s'
]

df_modelo = df[colunas_modelo].copy()

print(df_modelo.head())
print(df_modelo.isnull().sum())

# Separação entre variáveis explicativas (X) e variável alvo (y)
X = df_modelo.drop(columns=['pa_class_s'])
y = df_modelo['pa_class_s']

print(X.shape)
print(y.value_counts().head())

# Codificação das variáveis categóricas
colunas_categoricas = ['condic', 'sexopac']

encoder = LabelEncoder()

for col in colunas_categoricas:
    X[col] = encoder.fit_transform(X[col])

print(X.head())

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_treino.shape)
print(X_teste.shape)

modelo = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

modelo.fit(X_treino, y_treino)

y_pred = modelo.predict(X_teste)

print(classification_report(y_teste, y_pred))