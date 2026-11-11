# Sistema recomanador de pel·lícules

Pràctica de factorització de matrius amb biaixos sobre MovieLens Small, implementada amb NumPy i pandas. L'entrenament SGD es troba a `src/`; el notebook només carrega, crida i mostra els resultats.

## Contingut

- `src/data.py`: càrrega, estadístiques i partició cronològica.
- `src/model.py`: model biased MF, entrenament i predicció.
- `src/recommendation.py`: recomanacions i historial.
- `src/evaluation.py`: RMSE, MAE i NDCG@k.
- `notebooks/notebook.ipynb`: demostració de les funcionalitats.
- `spec.md`, `architecture.md`, `tasks.md`, `reglesIA.md`: especificació i documentació del projecte.
- `respostes.md`: justificació dels conceptes de l'enunciat.

## Execució

Des de l'arrel del repositori:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
jupyter notebook notebooks/notebook.ipynb
```

Executa les cel·les del notebook de dalt a baix. El model utilitza inicialització aleatòria sense seed fixa, per tant els resultats numèrics poden variar.

## Casos no predicibles

Les puntuacions d'usuaris o pel·lícules absents del train retornen `None`; aquestes parelles s'ometen en les mètriques. Les recomanacions es limiten a pel·lícules conegudes pel model. Consulta `spec.md` i `respostes.md` per a les decisions i limitacions.

## Verificació

Les proves cobreixen la partició, l'equivalència de les actualitzacions SGD, els casos cold-start, recomanacions, historial i mètriques.
