# Tiamat — execução HPC (modelo inicial)

## Preparação
1. Coloque os arquivos originais `train.csv` e `unique_m.csv` em `data/`.
2. **Revise/canonicalize fórmulas equivalentes** na coluna `material` antes de tratar a separação agrupada como definitiva; o script apenas remove espaços nas extremidades.
3. Crie ambiente Python (por exemplo `python -m venv .venv`), instale dependências (`pip install -r requirements.txt`) e registre suas versões finais.
4. Execute **uma vez** `python scripts/preparar_divisoes.py`. O notebook da Ridge precisa **ler `data/divisoes.npz`**, não gerar divisões próprias. Folds no NPZ são posições relativas ao conjunto de treinamento.
5. Crie `hpc/logs/` antes de submeter (`mkdir -p hpc/logs`).
6. Ajuste os recursos (`--partition`, `--account`, memória, tempo, CPUs) e a ativação do Python em `hpc/job.sh` às normas do seu cluster.
7. Faça um teste curto: `python scripts/otimizar.py --modelo random_forest --trials 2`.
8. Envie o array: `sbatch hpc/job.sh`. O job executa 4 tarefas independentes.

## Resultados
- `results/trials/<modelo>.csv`: histórico da otimização.
- `results/best_params/<modelo>.json`: melhor configuração, MAEs por fold, versões e job ID.
- `hpc/logs/`: saída e erros SLURM.

## Observações
- A seleção dos atributos (`SelectKBest`) é ajustada dentro de cada fold; nunca usa o conjunto de teste.
- `n_jobs=1` por modelo; os quatro jobs podem executar em paralelo, sujeitos aos limites da instituição.
- Os intervalos dos hiperparâmetros são **provisórios**. Justifique ou refine antes da execução final.
- Estudos Optuna estão em memória: interrupção do job perde trials não exportados. Para jobs longos, estude persistência/checkpoint ou divida execuções com cuidado.
- Se o notebook já treinou Ridge com divisões diferentes, **refaça sua avaliação usando `divisoes.npz`** antes de comparar os algoritmos.
- Não execute ou escolha modelos com base no conjunto de teste antes da avaliação final.
- Nome dos autores e instituição devem ser preenchidos no README definitivo.
