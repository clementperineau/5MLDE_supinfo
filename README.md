# 5MLDE_supinfo

Pour ce projet nous devons utiliser le Dataset Titanic. Le but est de réaliser une automatisation de tous les processus à savoir le clean des données et leurs validations, le preprocessing, l'entrainement et enfin pouvoir réaliser une prédiction via un site web déployé pour tous.

## Workflow

Nous avons utilisé Prefect pour l'orchestration qui nous permet de créer nos taches et nos flux ainsi que des automatisation pour relancer nos taches s'il y a eu un soucis afin d'éviter un maximun d'intervention humaine. Cela nous permet aussi de monitorer via l'interface web le bon déroulement de nos modèles ainsi que le temps d'execution.

Nous avons utilisé MLflow qui permet d'industrialiser nos modèles, on retrouve directement via le panel web par exemple nos grahpiques provenant de seaborn ou matplotlib, des infos sur la donnée, des métriques ainsi que pouvoir l'exposer via une api afin de récupérer des données pour notre front.


## Validation des données

Utilisation de Great Expectation. 
Qualification des données pour cadrer les données entrantes dans le model.

# Environnement

Pour lancer le projet, veuillez vous référencer au dépot suivant:  https://github.com/artefactory/supinfo_course_setup

# Lancer l'application:

Une fois ce dépot cloné dans le docker `mlops_notebooks`, rendez-vous dans le terminal.

![image](https://user-images.githubusercontent.com/49674310/228940382-90f6055f-c615-4f6f-9237-a731ccefba67.png)
![image](https://user-images.githubusercontent.com/49674310/228940583-f23bbcd2-44ba-4062-a907-087e69a50050.png)

Une fois dans le terminal exécuter:
```bash
bash run.sh
```

Vous pouvez vous rendre sur `localhost:4200` pour Prefect et `localhost:5001` pour MLflow
