# Respostes i justificació

> Respostes preparades a partir del codi i de l'enunciat. Cal revisar-les i assegurar que se'n pot explicar cada punt personalment.

## 1. Especificació i implementació

Els criteris principals són: carregar les dades, separar cronològicament per usuari, entrenar MF amb biaixos, mantenir les prediccions dins `[0.5, 5.0]`, recomanar només pel·lícules no puntuades que el model coneix, i avaluar amb RMSE, MAE i NDCG@10. La implementació està separada per responsabilitat: dades, model, recomanació i avaluació. Aquesta separació facilita provar cada part sense duplicar l'algorisme al notebook.

Una partició aleatòria podria barrejar valoracions antigues i futures del mateix usuari. La partició temporal representa millor l'ús real si es vol predir què puntuarà més endavant. La regla actual ordena per timestamp i deixa com a mínim una valoració a train; els casos petits poden no tenir gaire informació de test.

## 2. Comprensió del model

La predicció és:

```text
r_hat_ui = mu + b_u + b_i + dot(p_u, q_i)
```

`mu` és la mitjana de train. `b_u` modela si un usuari tendeix a puntuar per sobre o per sota de la mitjana; `b_i` captura la tendència general de la pel·lícula. El producte escalar dels vectors latents modela la compatibilitat específica usuari-pel·lícula.

La regularització L2 penalitza paràmetres grans i redueix l'ajust excessiu a les valoracions observades. En cada observació, l'error és `rating - prediction`. SGD actualitza biaixos i vectors en la direcció que redueix l'error, amb un terme de regularització. El codi copia el vector `P[u]` abans d'actualitzar-lo i el fa servir també per actualitzar `Q[i]`; així ambdues actualitzacions es basen en el mateix vector previ, no en un `P[u]` ja modificat.

Es barregen els índexs de les observacions a cada època per evitar que l'ordre fix de les files condicioni les actualitzacions. Els valors per defecte són 20 factors, `lr=0.005`, `lambda_=0.02` i 20 èpoques.

## 3. Entrenament

El notebook mostra MSE de train per època, però no hi ha conjunt de validació ni early stopping. Per tant, una pèrdua decreixent demostra ajust al train, no necessàriament millora de generalització. Un conjunt de validació temporal permetria escollir el nombre d'èpoques i els hiperparàmetres sense consultar el test. No s'ha utilitzat per no canviar el procediment original.

Una taxa d'aprenentatge massa alta pot fer divergir l'optimització; una massa baixa pot fer-la lenta. Més factors augmenten la capacitat i el risc d'overfitting. Més regularització restringeix els paràmetres; massa regularització pot infraajustar. Més èpoques poden millorar l'ajust de train, però no garanteixen millor test.

## 4. Avaluació

- **RMSE** és l'arrel de la mitjana dels errors al quadrat i penalitza més els errors grans.
- **MAE** és la mitjana del valor absolut de l'error i és més fàcil d'interpretar en unitats de puntuació.
- **NDCG@10** mesura la qualitat de l'ordre dels ítems rellevants en les primeres deu posicions. La rellevància és `2**rating - 1`; després es compara el DCG obtingut amb l'ideal.

Una predicció numèrica bona no implica necessàriament un rànquing bo: RMSE/MAE mesuren proximitat de puntuacions, mentre NDCG valora l'ordre relatiu. NDCG es calcula per usuari amb almenys una predicció vàlida i després se'n fa la mitjana; usuaris amb poques valoracions de test tenen una estimació més variable.

Els usuaris i pel·lícules desconeguts al train no tenen paràmetres apresos. `predict_rating` retorna `None`; l'avaluació els omet, i RMSE/MAE divideixen pel nombre de prediccions vàlides. Això evita una predicció fictícia, però les mètriques descriuen només la població amb cobertura i poden ser optimistes respecte de tot el test. Al dataset actual, els usuaris del test també són presents a train perquè la partició es fa dins de cada usuari; hi ha ítems de test no presents al train.

## 5. Casos extrems i robustesa

- **Usuari nou o pel·lícula desconeguda:** `predict_rating` retorna `None`; no es fa fallback a `mu`. Les mètriques ometen la parella no predicible.
- **Usuari amb poques valoracions:** es manté almenys una fila a train. Amb una sola fila, el model individual pot tenir poca evidència; la regularització limita els paràmetres, però no elimina la incertesa.
- **Usuari sense candidats:** recomanació retorna una taula buida.
- **Predicció fora de rang:** es retalla a `[0.5, 5.0]`.
- **Zero files predictibles en avaluació:** RMSE/MAE retornen `NaN`, no hi ha divisió per zero. NDCG global també és `NaN` quan cap usuari té predicció.

No s'exclouen explícitament usuaris amb menys de `k` test items: NDCG calcula sobre els que hi ha. Excloure'ls reduiria la cobertura i canviaria la població de l'avaluació.

## 6. Comportament i limitacions

Per a cada usuari conegut, es puntuen les pel·lícules presents al train que no apareixen al seu historial de train; s'ordenen de puntuació predita més alta a més baixa. Els factors latents fan que usuaris amb preferències semblants puguin tenir rànquings semblants. És un recomanador basat només en identificadors i historial de valoracions: no generalitza a usuaris/ítems nous i no utilitza contingut de gènere en l'entrenament.

La inicialització i l'ordre de mostres són aleatoris i no s'ha fixat una seed per preservar el comportament del codi anterior. Per això les mètriques poden variar lleugerament entre execucions. La corba de train i les mètriques de test s'han d'interpretar juntes i no com una garantia universal del rendiment.
