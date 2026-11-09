<p align="center">

  <img src="arquivos/images/header_ilum.png" width="100%">
</p>


<p align="center">
  <img src= Arquivos/images/brasao.png width="500">
</p>

# **Casa Noctis**

<h1 style="text-align: center;">"Onde a luz se extingue, nosso juramento permanece."</h1>

<br><br>




<p align="center">
  <img src="https://img.shields.io/badge/Linguagem-Python%203.10+-blue">
  <img src="https://img.shields.io/badge/Área-Supercondutividade-green">
  <img src="https://img.shields.io/badge/Foco-Aprendizado%20de%20Máquina-blueviolet">
  <img src="https://img.shields.io/badge/Tarefa-Regressão-orange">
  <img src="https://img.shields.io/badge/Otimização-Optuna-red">
  <img src="https://img.shields.io/badge/Computação%20de%20Alto%20Desempenho-SLURM-purple">
</p>


<h1 style="text-align: center;">Tiamat — Predição da temperatura crítica de supercondutores</h1>


<p align="center">
  <img src= Arquivos/images/grafico_modelos.png width="500">
</p>

Projeto de Machine Learning voltado à **predição da temperatura crítica ($T_c$)** de materiais supercondutores a partir de descritores físico-químicos derivados de suas composições químicas. O estudo compara algoritmos de regressão, investiga a seleção de atributos e discute a importância preditiva de propriedades elementares, sem confundi-la com causalidade física.

Projeto desenvolvido pelo aluno **Matheus Luiz Mendes de Souza** como trabalho final da disciplina de **Aprendizado de máquina** do segundo semestre do **Bacharelado em Ciência e Tecnologia** ministrada pelo Prof. Dr. **Daniel R. Cassar** na **Ilum Escola de Ciência (CNPEM)**.

## Objetivo

Investigar em que medida os atributos de composição química permitem estimar a temperatura crítica dos supercondutores e como a escolha do algoritmo e a seleção de atributos afetam o desempenho preditivo.

O trabalho busca, em particular:

- Comparar quatro métodos baseados em árvores com uma regressão linear regularizada.

- Estabelecer referências de desempenho utilizando previsões constantes (média e mediana).

- Evitar o compartilhamento de fórmulas químicas textualmente idênticas entre treinamento, validação e teste.

- Investigar os atributos utilizados pelos modelos por meio de **Permutation Importance**.

- Discutir as limitações científicas de prever $T_c$ apenas a partir da composição.

## Conjunto de dados

Foi utilizado o **Superconductivity Data**, disponibilizado no [UCI Machine Learning Repository](https://doi.org/10.24432/C53P47) e descrito por Hamidieh (2018).

| Característica | Descrição |
| --- | --- |
| Observações | 21.263 |
| Atributos preditivos | 81 |
| Variável-alvo | Temperatura crítica, $T_c$, em kelvin (K) |
| train.csv | Descritores físico-químicos e critical_temp |
| unique_m.csv | Composições químicas correspondentes às observações |

Os descritores são derivados de propriedades elementares, incluindo **massa atômica, primeira energia de ionização, raio atômico, densidade, afinidade eletrônica, calor de fusão, condutividade térmica e valência**. São utilizadas estatísticas como médias, amplitudes, entropias e desvios-padrão, inclusive versões ponderadas pela composição.

**Importante:** esses valores descrevem propriedades dos elementos e estatísticas de suas composições. Não equivalem necessariamente a medidas diretas das propriedades efetivas do material supercondutor.

## Metodologia

1. **Exploração dos dados:** distribuição da temperatura crítica, valores ausentes, registros repetidos, correlações e escalas dos atributos.

2. **Separação treino–teste:** GroupShuffleSplit, com test_size=0.20 aplicado aos grupos definidos pela string da fórmula química (material), reservando as observações desses grupos para teste.

3. **Validação cruzada:** GroupKFold com **5 folds**, respeitando os grupos no conjunto de treinamento.

4. **Baselines:** DummyRegressor com estratégias mean e median.

5. **Modelagem:** pipelines para Ridge Regression, Random Forest, Extra Trees, HistGradientBoosting e XGBoost. A Ridge utiliza StandardScaler.

6. **Seleção de atributos:** SelectKBest(f_regression) dentro dos pipelines, comparando a preservação dos 81 atributos com subconjuntos de 10 a 80 atributos.

7. **Otimização:** Optuna, com amostrador TPE e minimização do **MAE médio dos cinco folds**.

8. **Avaliação final:** ajuste dos modelos selecionados em todo o conjunto de treinamento, seguido de avaliação no conjunto de teste independente.

9. **Interpretabilidade:** importância por permutação dos atributos nos modelos Extra Trees e XGBoost.

Semente aleatória principal: **SEED = 210906**. O objetivo da otimização é o MAE; **RMSE** e **MAAPE** são métricas complementares de avaliação.

### Execução no HPC

A otimização dos modelos baseados em árvores foi conduzida com scripts Python submetidos via **SLURM** no HPC institucional. O notebook concentra a documentação, a análise e a reconstrução dos modelos a partir dos resultados exportados.

O planejamento previa até **700 trials por algoritmo baseado em árvores**, com tarefas inicialmente configuradas para **4 CPUs e 8 GB de RAM**. A busca do **XGBoost foi interrompida antes de atingir o orçamento planejado**. A otimização da Ridge foi realizada no notebook, com 100 trials.

Os históricos dos trials foram preservados em arquivos CSV, e os melhores parâmetros dos modelos foram registrados para a reconstrução dos pipelines. Esses arquivos são necessários para reproduzir as etapas posteriores do notebook sem repetir a otimização.

## Resultados


### Comparação pelo MAE

Valores apresentados em kelvin (K); **menor é melhor**. A coluna de validação cruzada dos modelos otimizados corresponde ao melhor MAE médio identificado na busca. Para os baselines, corresponde à média dos cinco folds.

| Modelo | MAE — validação cruzada (K) | MAE — teste (K) |
| --- | ---: | ---: |
| **Extra Trees** | **5,167** | **5,411** |
| XGBoost | 5,251 | 5,508 |
| HistGradientBoosting | 5,419 | 5,661 |
| Random Forest | 5,470 | 5,685 |
| Ridge Regression | 13,462 | 13,272 |
| Baseline — Mediana | 27,477 | 27,629 |
| Baseline — Média | 29,261 | 29,656 |

O **Extra Trees** obteve o menor MAE no conjunto de teste, além de **RMSE de 9,881 K** e **MAAPE de 0,271 rad**. O XGBoost obteve um resultado próximo, mas a diferença de MAE entre os dois não demonstra, isoladamente, superioridade estatisticamente significativa.

Os quatro modelos baseados em árvores superaram a regressão Ridge em MAE. Todos os cinco algoritmos apresentaram erros menores que os baselines, indicando que os descritores químicos fornecem informação preditiva útil no conjunto de dados avaliado.

### Seleção e importância dos atributos


<p align="center">
  <img src= Arquivos/images/selecao_atributos.png width="500">
</p>


As configurações de menor MAE dos cinco algoritmos **preservaram todos os 81 atributos**. As distribuições observadas também apresentaram menores medianas de MAE sem seleção, mas essa comparação é **exploratória**: os números de trials e os hiperparâmetros explorados diferiram entre condições.

A análise de *Permutation Importance* destacou descritores relacionados à **condutividade térmica, ao raio atômico, à valência e à heterogeneidade das propriedades elementares**. O atributo range_ThermalConductivity apresentou aumentos médios de MAE após permutação de **3,737 K** no Extra Trees e **9,232 K** no XGBoost.

Esses valores representam **dependência preditiva dos modelos**, e não evidência de que a condutividade térmica determine causalmente a temperatura crítica.

## Estrutura de arquivos utilizada

O notebook espera encontrar os dados e os resultados da otimização em caminhos relativos ao diretório a partir do qual é executado. A estrutura abaixo apresenta os **caminhos utilizados no código**; ela não é uma confirmação de que todos os arquivos estejam incluídos neste repositório.

```text

Tiamat/
├── README.md
├── LICENSE
└── Arquivos/
    ├── tiamat.ipynb
    ├── data/
    │   ├── train.csv
    │   └── unique_m.csv
    ├── images/
    │   ├── brasao.png
    │   ├── grafico_modelos.png
    │   ├── header_ilum.png
    │   └── selecao_atributos.png
    ├── results/
    │   └── optuna/
    └── tiamat_hpc/
        ├── README.md
        ├── requirements.txt
        ├── data/
        │   ├── train.csv
        │   ├── unique_m.csv
        │   ├── divisoes.npz
        │   └── divisoes_info.json
        ├── scripts/
        │   ├── otimizar.py
        │   ├── preparar_divisoes.py
        │   ├── preparar.sh
        │   └── logs/
        ├── hpc/
        │   ├── job.sh
        │   └── logs/
        ├── logs/
        └── results/
            ├── best_params/
            │   ├── random_forest.json
            │   ├── extra_trees.json
            │   └── hist_gradient_boosting.json
            └── trials/
                ├── random_forest.csv
                ├── random_forest_parcial.csv
                ├── extra_trees.csv
                ├── extra_trees_parcial.csv
                ├── hist_gradient_boosting.csv
                ├── hist_gradient_boosting_parcial.csv
                ├── ridge.csv
                └── xgboost_parcial.csv

```

O notebook também cria results/optuna/ durante a preparação da otimização local. Os scripts Python e o arquivo de submissão SLURM utilizados no HPC são componentes adicionais da execução, mas seus nomes e caminhos não estão especificados nesta estrutura.

## Como executar

1. Prepare um ambiente Python com as bibliotecas utilizadas no notebook. Por exemplo:

   ```bash
   python -m pip install numpy pandas matplotlib scikit-learn xgboost optuna jupyterlab
   ```

2. Abra o notebook a partir da raiz do projeto e inicie o jupyter no diretório:

   
   ```bash
   cd Arquivos
   
   jupyter lab
   ```

3. Execute as etapas iniciais de exploração, preparação e definição dos modelos. Para reproduzir as seções de análise da otimização e de avaliação final **sem refazer a busca**, disponibilize também os CSVs e JSONs gerados pelo HPC nos caminhos esperados.

**Nota de reprodutibilidade:** a instalação acima informa as dependências principais, mas não fixa versões. A reprodução exata também depende das versões das bibliotecas, da disponibilidade dos arquivos de resultados e das configurações de execução. Para uma nova execução integral da otimização no HPC, são necessários os scripts e a configuração SLURM utilizados no experimento.

## Limitações

- Os 81 descritores não incluem diretamente estrutura cristalina, propriedades eletrônicas, condições de processamento ou condições experimentais que podem influenciar $T_c$.

- O agrupamento por material separa fórmulas **textualmente idênticas**, mas não verifica automaticamente representações diferentes de composições equivalentes.

- Os algoritmos foram otimizados em espaços de hiperparâmetros distintos, e o XGBoost não concluiu todo o orçamento planejado.

- A análise da seleção de atributos não constitui um experimento controlado do efeito isolado dessa etapa.

- A importância por permutação pode ser influenciada pela correlação entre atributos e não revela relações causais.

- O desempenho no teste não garante a mesma precisão para famílias de supercondutores pouco representadas nos dados.


## Docente responsável

> Este trabalho só foi possível graças ao apoio do professor da matéria de Aprendizado de Máquina.

<table>
  
  <td align="center">
          <a href="#" title="Prof. Daniel R. Cassar">
        <img src="https://avatars.githubusercontent.com/u/9871905?v=4" width="100px;" alt="Foto do Daniel do Github"/><br>
          <a href="https://github.com/drcassar"><b>Prof. Dr. Daniel R. Cassar<b></a>
      </a>

  </tr>
</table>

---

## Autor
> O material foi produzido por Matheus Luiz Mendes de Souza, aluno do segundo semestre do curso de Bacharelado em Ciência e Tecnologia da Ilum - Escola de Ciência. 
<table>
  <tr>

   <td align="center">
      <a href="#" title="Matheus Souza">
        <img src="https://avatars.githubusercontent.com/u/284382226?v=4" width="100px;" alt="Foto do Matheus do Github"/><br>
          <a href="https://github.com/matheuslmds"><b>Matheus Souza<b></a>
      </a>
    </td>
  </tr>
</table>

--- 

## Referências principais

- **Hamidieh, K. (2018).** *A data-driven statistical model for predicting the critical temperature of a superconductor*. Computational Materials Science, 154, 346–354. [DOI](https://doi.org/10.1016/j.commatsci.2018.07.052).
- **Hamidieh, K. (2018).** *Superconductivty Data* [conjunto de dados]. UCI Machine Learning Repository. [DOI](https://doi.org/10.24432/C53P47).
- **Kim, S.; Kim, H. (2016).** *A new metric of absolute percentage error for intermittent demand forecasts*. International Journal of Forecasting, 32(3), 669–679. [DOI](https://doi.org/10.1016/j.ijforecast.2015.12.003).

Outras referências e a descrição do uso de ferramentas de inteligência artificial estão documentadas no notebook.
