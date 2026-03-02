# Projeto Final — Tópicos em Ciência de Dados
Análise do SIASUS Psicossocial na Região Oeste do Paraná

## Descrição Geral

Este projeto tem como objetivo aplicar técnicas de Ciência de Dados sobre bases administrativas do Sistema de Informações Ambulatoriais do SUS (SIASUS), com foco nos procedimentos psicossociais realizados nos municípios da região Oeste do Paraná.

O projeto contempla todas as etapas do pipeline de dados, incluindo ingestão, tratamento, análise exploratória, modelagem com diferentes paradigmas de Machine Learning e visualização de resultados. O detalhamento metodológico completo encontra-se descrito em relatório técnico entregue separadamente.

## Fonte dos Dados

Os dados utilizados são provenientes de fontes oficiais do Ministério da Saúde, disponibilizados por meio do DATASUS, no formato DBF. Após os processos de conversão, unificação e tratamento, o dataset final ultrapassa 1 milhão de registros.

Devido ao grande volume de dados, os arquivos CSV e DBF não estão incluídos neste repositório. Todos os scripts necessários para reproduzir o processo de obtenção, conversão, unificação e tratamento dos dados estão disponíveis no projeto.

## Estrutura do Projeto

data/
raw/
siasus_dbf/
siasus_csv/
processed/
siasus_consolidado.csv
siasus_tratado.csv

scripts/
tratamento_dados/
converter_dbf_para_csv.py
unificar_siasus.py
limpeza_transformacao.py
eda_inicial.py
eda_depois.py

machine_learning/
ml_supervisionado.py
ml_clusterizacao.py
ml_reforco.py

visualizacoes/
vis_supervisionado.py
vis_nao_supervisionado.py
vis_reforco.py

dataset_info.py
main.py
requirements.txt
README.md

## Requisitos

Python 3.9 ou superior.

Instalação das dependências:
pip install -r requirements.txt

## Como Executar o Projeto

Execução principal:
python main.py

O menu interativo permite visualizar informações do dataset, executar modelos de Machine Learning e exibir dashboards.

## Tratamento de Dados

Para reproduzir o pipeline completo a partir dos dados brutos:

python converter_dbf_para_csv.py
python unificar_siasus.py
python limpeza_transformacao.py
python eda_inicial.py
python eda_depois.py

## Modelagem de Machine Learning

O projeto utiliza:
- Aprendizagem supervisionada com árvore de decisão
- Aprendizagem não supervisionada com K-Means
- Aprendizagem por reforço com Q-Learning (simulação conceitual)

## Visualizações

Os resultados são apresentados por meio de dashboards estáticos desenvolvidos com Matplotlib e Seaborn, priorizando interpretação e análise crítica.

## Observações Finais

Este repositório contém exclusivamente o código-fonte e a documentação operacional do projeto. O relatório técnico é entregue separadamente.
