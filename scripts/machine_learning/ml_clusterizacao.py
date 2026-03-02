import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = load_siasus_tratado()

print(df.shape)

# Agregação dos dados por município
df_municipio = (
    df.groupby('ufmun')
      .agg(
          total_atendimentos=('pa_qtdapr', 'sum'),
          media_idade=('idadepac', 'mean'),
          total_procedimentos=('pa_proc_id', 'nunique')
      )
      .reset_index()
)

print(df_municipio.head())
print(df_municipio.shape)

# Seleção das variáveis para clusterização
X_cluster = df_municipio[
    ['total_atendimentos', 'media_idade', 'total_procedimentos']
]

print(X_cluster.head())

scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

print(X_cluster_scaled[:5])

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_cluster_scaled)

df_municipio['cluster'] = clusters

print(df_municipio.head())
print(df_municipio['cluster'].value_counts())