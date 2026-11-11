# Tasques

> Estat completat després d'inspeccionar i provar la implementació existent. Aquest registre és retrospectiu: no documenta una seqüència històrica de prompts o d'ús d'IA.

## Fase 1 — Especificació i disseny

- [x] Documentar requisits, comportament i criteris d'acceptació a `spec.md`.
- [x] Documentar mòduls, interfícies, flux i decisions a `architecture.md`.

## Fase 2 — Dades

- [x] Encapsular la càrrega de ratings i pel·lícules a `src/data.py`.
- [x] Encapsular estadístiques de densitat i partició cronològica per usuari.
- [x] Verificar la conservació temporal i que cada usuari tingui train.

## Fase 3 — Model

- [x] Encapsular factors, biaixos, mitjana i entrenament SGD a `src/model.py`.
- [x] Mantenir la predicció dins l'interval de valoracions i tractar cold-start.
- [x] Comparar actualitzacions SGD i pèrdua amb la implementació anterior en una prova determinista.

## Fase 4 — Recomanació i avaluació

- [x] Encapsular recomanacions i historial a `src/recommendation.py`.
- [x] Encapsular RMSE, MAE i NDCG@k a `src/evaluation.py`.
- [x] Dividir RMSE/MAE només per prediccions vàlides i tractar el cas de zero prediccions.
- [x] Provar mètriques, cold-start i absència de candidats.

## Fase 5 — Demostració i lliurament

- [x] Canviar el notebook per cridar la llibreria de `src/`.
- [x] Mostrar les etapes i exemples demanats a l'enunciat.
- [x] Executar proves i el notebook de dalt a baix.
- [x] Preparar README, respostes i paquet comprimit del projecte.
