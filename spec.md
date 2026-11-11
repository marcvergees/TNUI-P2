# Especificació del sistema

> Document de referència redactat retrospectivament a partir del codi i de l'enunciat. No representa una especificació escrita abans de la implementació.

## Objectiu

Construir amb Python un sistema de recomanació de pel·lícules basat en factorització de matrius amb biaixos, entrenat des de zero amb MovieLens Small.

## Dades

- `data/ratings.csv`: `userId`, `movieId`, `rating`, `timestamp`.
- `data/movies.csv`: `movieId`, `title`, `genres`.
- Les valoracions van de 0,5 a 5,0.
- Els identificadors originals es mapegen a índexs densos interns; el mapa i tots els paràmetres apresos provenen únicament del train.

## Model i entrenament

La predicció és `mu + b_u + b_i + dot(P[u], Q[i])`, amb mitjana global, biaixos d'usuari i pel·lícula, i factors latents.

L'entrenament minimitza l'error quadràtic amb regularització L2 mitjançant SGD. Per defecte:

| Paràmetre | Valor |
|---|---:|
| `k` | 20 |
| `lr` | 0.005 |
| `lambda_` | 0.02 |
| `epochs` | 20 |

Es retorna i es mostra el MSE de train de cada època. L'ordre aleatori de les mostres es torna a generar a cada època.

## Funcionalitats

1. Llegir els dos CSV de MovieLens.
2. Calcular comptatges i densitat de la matriu usuari-pel·lícula.
3. Dividir per usuari en ordre cronològic: el 80% inicial a train i el tram més recent a test; cada usuari conserva com a mínim una fila a train.
4. Entrenar el model i predir puntuacions dins `[0.5, 5.0]`.
5. Recomanar les N pel·lícules conegudes pel model que l'usuari no ha puntuat a train, amb títol, gèneres i puntuació.
6. Mostrar l'historial de train amb títol i gèneres, ordenat per puntuació descendent.
7. Calcular RMSE, MAE i NDCG@10 al test.

## Casos extrems i avaluació

- Una predicció per a un usuari o una pel·lícula no presents al train retorna `None`.
- Les files de test no predictibles s'exclouen de RMSE i MAE; el denominador és el nombre de prediccions vàlides. Si no n'hi ha cap, les mètriques de regressió són `NaN`.
- NDCG es calcula per als usuaris amb almenys una predicció vàlida, i després se'n fa la mitjana. La rellevància gradual és `2**rating - 1`.
- Les recomanacions d'un usuari desconegut retornen una taula buida; si no hi ha candidats, la taula també és buida.
- El sistema no fa prediccions per a usuaris nous sense historial al train (cold-start).

## Criteris d'acceptació

- Els CSV s'han de carregar i els comptatges/densitat han de ser calculables.
- El MSE de train ha de tendir a disminuir durant les èpoques.
- RMSE de test inferior a 0,95 i MAE al voltant de 0,67 (resultats dependents de la inicialització aleatòria).
- Els usuaris coneguts reben recomanacions amb metadades i puntuacions dins del rang.
- El notebook demostratiu s'executa de dalt a baix i mostra càrrega, estadístiques, partició, entrenament, corba, mètriques, exemples, recomanacions i historial.

## Restriccions tècniques

Python amb NumPy, pandas i matplotlib; Jupyter per al notebook. La factorització s'implementa al projecte i no s'utilitzen llibreries de recomanació.
