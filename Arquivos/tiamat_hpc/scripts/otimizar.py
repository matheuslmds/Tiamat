import argparse
import json
import os
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import sklearn
import xgboost
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

RAIZ = Path(__file__).resolve().parents[1]
SEED = 210906
N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
MODELOS = ('random_forest', 'extra_trees', 'hist_gradient_boosting', 'xgboost')

def construir(trial, nome):
    usar = trial.suggest_categorical('usar_selecao', [True, False])
    k = trial.suggest_int('k', 10, 80) if usar else 'all'
    selecao = SelectKBest(score_func=f_regression, k=k)
    if nome in ('random_forest', 'extra_trees'):
        kwargs = dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 400, step=50),
            max_depth=trial.suggest_categorical('max_depth', [None, 5, 10, 15, 20, 25, 30]),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            max_features=trial.suggest_float('max_features', .3, 1.0),
            random_state=SEED, n_jobs=N_CPUS,
        )
        if nome == 'random_forest':
            kwargs['max_samples'] = trial.suggest_float('max_samples', .6, 1.0)
            modelo = RandomForestRegressor(**kwargs)
        else:
            modelo = ExtraTreesRegressor(**kwargs)
    elif nome == 'hist_gradient_boosting':
        modelo = HistGradientBoostingRegressor(
            learning_rate=trial.suggest_float('learning_rate', .01, .3, log=True),
            max_iter=trial.suggest_int('max_iter', 100, 500, step=50),
            max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 15, 63),
            max_depth=trial.suggest_categorical('max_depth', [None, 4, 6, 8, 12]),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 10, 60),
            l2_regularization=trial.suggest_float('l2_regularization', 1e-5, 10.0, log=True),
            early_stopping=False, random_state=SEED,
        )
    elif nome == 'xgboost':
        modelo = XGBRegressor(
            n_estimators=trial.suggest_int('n_estimators', 100, 600, step=50),
            learning_rate=trial.suggest_float('learning_rate', .01, .3, log=True),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            min_child_weight=trial.suggest_float('min_child_weight', 1., 15., log=True),
            subsample=trial.suggest_float('subsample', .6, 1.),
            colsample_bytree=trial.suggest_float('colsample_bytree', .6, 1.),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-5, 10., log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 100., log=True),
            objective='reg:squarederror', tree_method='hist',
            random_state=SEED, n_jobs=N_CPUS,
        )
    else:
        raise ValueError(nome)
    return Pipeline([('selecao', selecao), ('modelo', modelo)])

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--modelo', choices=MODELOS, required=True)
    p.add_argument('--trials', type=int, default=100)
    args = p.parse_args()
    if args.trials < 1:
        p.error('trials deve ser positivo')

    df = pd.read_csv(RAIZ / 'data/train.csv')
    X = df.drop(columns='critical_temp')
    y = df['critical_temp']
    with np.load(RAIZ / 'data/divisoes.npz') as divisao:
        idx_treino = divisao['treino']
        idx_teste = divisao['teste']
        folds = [(divisao[f'fold_{i}_treino'], divisao[f'fold_{i}_validacao']) for i in range(5)]
    if set(idx_treino) & set(idx_teste) or len(idx_treino) + len(idx_teste) != len(df):
        raise ValueError('Particao treino/teste inconsistente')
    Xtr, ytr = X.iloc[idx_treino], y.iloc[idx_treino]
    for tr, val in folds:
        if (set(tr) & set(val)) or max(np.r_[tr, val]) >= len(Xtr):
            raise ValueError('Fold inconsistente')

    def objetivo(trial):
        modelo = construir(trial, args.modelo)
        erros = []
        from sklearn.base import clone
        for tr, val in folds:
            estimador = clone(modelo)
            estimador.fit(Xtr.iloc[tr], ytr.iloc[tr])
            pred = estimador.predict(Xtr.iloc[val])
            erros.append(float(mean_absolute_error(ytr.iloc[val], pred)))
        trial.set_user_attr('mae_por_fold', erros)
        return float(np.mean(erros))



    def salvar_parcial(study, trial):
        if len(study.trials) % 10 != 0:
            return

        pasta = RAIZ / "results/trials"
        pasta.mkdir(parents=True, exist_ok=True)

        destino = pasta / f"{args.modelo}_parcial.csv"
        temporario = pasta / f"{args.modelo}_parcial.tmp"

        study.trials_dataframe().to_csv(
            temporario,
            index=False
        )

        temporario.replace(destino)

        print(
            f"Checkpoint: {len(study.trials)} trials registrados",
            flush=True
        )
                
    estudo = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    # Armazenamento em memoria; apenas os trials concluidos ficam exportados ao final.
    estudo.optimize(objetivo, n_trials=args.trials, n_jobs=1, callbacks=[salvar_parcial])
    (RAIZ / 'results/trials').mkdir(parents=True, exist_ok=True)
    (RAIZ / 'results/best_params').mkdir(parents=True, exist_ok=True)
    estudo.trials_dataframe().to_csv(RAIZ / f'results/trials/{args.modelo}.csv', index=False)
    resumo = dict(modelo=args.modelo, n_trials=args.trials, melhor_mae_cv=estudo.best_value,
                  melhores_parametros=estudo.best_params,
                  mae_por_fold=estudo.best_trial.user_attrs['mae_por_fold'],
                  seed=SEED, n_folds=5, sklearn=sklearn.__version__,
                  optuna=optuna.__version__, xgboost=xgboost.__version__,
                  python=platform.python_version(), job_id=os.getenv('SLURM_JOB_ID'))
    (RAIZ / f'results/best_params/{args.modelo}.json').write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding='utf-8')
    print('CONCLUIDO', json.dumps(resumo, ensure_ascii=False))

if __name__ == '__main__':
    main()
