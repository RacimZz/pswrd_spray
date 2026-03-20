# Détection de Password Spraying

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Isolation%20Forest-orange)
![Cybersecurity](https://img.shields.io/badge/Domain-Cybersecurity-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Ce projet explore la détection d’attaques de password spraying à l’aide d’une approche de machine learning comportementale.

Le travail se concentre sur la détection d’anomalies avec **Isolation Forest** et sur des expériences menées sur des jeux de données publics tels que **CIC-IDS-2017** et **UNSW-NB15**. L’idée principale est de comparer une détection basée sur les flux avec une agrégation comportementale sur des fenêtres temporelles afin de mieux capturer des schémas d’attaque lents et répétitifs.

## Aperçu

Les attaques de password spraying sont souvent difficiles à détecter lorsque chaque événement est analysé indépendamment.  
Ce projet étudie si l’agrégation de l’activité dans le temps permet d’obtenir un signal plus efficace pour identifier les comportements suspects.

## Fonctionnalités principales

- Extraction de caractéristiques comportementales à partir d’événements d’authentification ou réseau.
- Détection d’anomalies avec Isolation Forest.
- Expériences sur des jeux de données publics en cybersécurité.
- Comparaison entre une analyse en flux brut et une agrégation temporelle.
- Rapport inclus avec la méthodologie et les résultats.

## Jeux de données

Le projet utilise ou adapte des expériences sur les jeux de données publics suivants :

- CIC-IDS-2017
- UNSW-NB15
- LANL RBA (travail d’adaptation)

## Structure du dépôt

```text
.
├── data/
├── reports/
├── scripts/
└── README.md
```

## Résultats

Les expériences montrent qu’une approche fondée sur l’agrégation comportementale peut améliorer significativement la détection de certaines attaques discrètes ou répétitives par rapport à une analyse en flux brut.

Une discussion détaillée de la méthodologie, des choix d’implémentation et des résultats est disponible dans le rapport joint.

## Prise en main

```bash
git clone https://github.com/RacimZz/pswrd_spray.git
cd pswrd_spray
pip install -r requirements.txt
```

Ensuite, exécutez les scripts depuis le sous-répertoire approprié du projet selon la configuration de votre jeu de données. Pour les expériences basées sur CIC, le projet a été exécuté depuis le répertoire `scripts` avec des chemins de données configurés en conséquence.

## Auteur
Racim ZENATI - Etudiant à l'ENSIMAG
