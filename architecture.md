# Arquitectura

> Document redactat retrospectivament a partir de la implementació existent; descriu l'organització actual, no una decisió prèvia al desenvolupament.

## Estructura

```text
P2/
├── data/                 # CSV MovieLens i particions generades pel notebook
├── notebooks/
│   └── notebook.ipynb    # demostració basada en les funcions de src
├── src/
│   ├── data.py           # càrrega, estadístiques i partició temporal
│   ├── model.py          # model biased MF, SGD i predicció
│   ├── recommendation.py # top-N i historial
│   └── evaluation.py     # RMSE, MAE i NDCG@k
├── tests/
│   └── test_recommender.py
├── spec.md
├── tasks.md
├── reglesIA.md
├── respostes.md
└── README.md
```

## Responsabilitats i interfícies

- `src.data.load_data(data_dir)` llegeix i retorna `ratings, movies`.
- `src.data.dataset_statistics(ratings, movies)` calcula comptatges i densitat.
- `src.data.temporal_train_test_split(ratings)` retorna els DataFrames de train i test.
- `src.model.BiasedMatrixFactorization` rep `k`, `lr`, `lambda_` i `epochs`. `fit(train)` aprèn els paràmetres i retorna la pèrdua per època; `predict_rating(user_id, movie_id)` retorna una puntuació o `None`.
- `src.recommendation.recommend_movies(...)` retorna un DataFrame amb pel·lícula, títol, gènere i puntuació predita; `get_user_history(...)` retorna les valoracions conegudes.
- `src.evaluation.evaluate_regression(...)` calcula RMSE/MAE, `ndcg_at_k(...)` calcula una mètrica de rànquing, i `evaluate_model(...)` les combina.
- `src/__init__.py` reexporta les interfícies per simplificar les crides del notebook.

## Flux de dades

Els CSV es carreguen una vegada; ratings es divideixen per usuari i timestamp. El model calcula els índexs, `mu`, biaixos i factors exclusivament des del train. Predicció, recomanació i avaluació reben el model entrenat. Els candidats es limiten a pel·lícules presents al train, ja que no hi ha vectors apresos per a ítems desconeguts.

## Decisions i límits

- La partició és cronològica per usuari, no aleatòria, per simular predicció de puntuacions posteriors. La regla actual usa `int(0.8 * nombre_de_valoracions)` i garanteix almenys una fila a train.
- La inicialització i la permutació d'èpoques fan servir el generador aleatori global de NumPy, com el codi original; per això els resultats poden variar entre execucions.
- El cold-start es tracta ometent prediccions no disponibles, sense fallback a la mitjana global.
- No s'ha incorporat validació ni ajust d'hiperparàmetres. La corba és de train i no és suficient per detectar sobreajustament.
- S'han conservat la fórmula i les actualitzacions SGD del codi original; el refactor encapsula aquesta lògica en mòduls.
