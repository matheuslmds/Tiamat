"""Execute UMA VEZ; o notebook deve reutilizar estes mesmos indices."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

RAIZ = Path(__file__).resolve().parents[1]
SEED = 210906

def main():
    df = pd.read_csv(RAIZ / 'data/train.csv')
    formulas = pd.read_csv(RAIZ / 'data/unique_m.csv')
    if len(df) != len(formulas):
        raise ValueError('Arquivos com tamanhos distintos')
    if not np.allclose(df['critical_temp'], formulas['critical_temp'], rtol=0, atol=1e-9):
        raise ValueError('Temperaturas nao correspondem; verificar alinhamento de linhas')
    grupos = formulas['material'].astype('string').str.strip()
    if grupos.isna().any() or (grupos == '').any():
        raise ValueError('Formulas quimicas ausentes')
    if df.drop(columns='critical_temp').shape[1] != 81:
        raise ValueError('Numero inesperado de preditores')
    # IMPORTANTE: strings equivalentes de formulas precisam ser normalizadas previamente.
    separador = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    treino, teste = next(separador.split(df, groups=grupos))
    cv = GroupKFold(n_splits=5)
    indices_folds = list(cv.split(df.iloc[treino], groups=grupos.iloc[treino]))
    saida = RAIZ / 'data/divisoes.npz'
    dados = {'treino': treino, 'teste': teste}
    for i, (tr, val) in enumerate(indices_folds):
        dados[f'fold_{i}_treino'] = tr
        dados[f'fold_{i}_validacao'] = val
    np.savez_compressed(saida, **dados)
    resumo = {'seed': SEED, 'n_registros': len(df), 'n_treino': len(treino),
              'n_teste': len(teste), 'n_folds': 5,
              'agrupamento': 'string de material com espacos removidos; revisar equivalencias quimicas'}
    (RAIZ / 'data/divisoes_info.json').write_text(json.dumps(resumo, indent=2), encoding='utf-8')
    print('Salvo:', saida, '|', resumo)

if __name__ == '__main__':
    main()
