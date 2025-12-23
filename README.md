# LearningDynamics_Paper

Pour l'architecture du projet, j'avais pensé a faire un notebook dans le dossier results la on mettra tous nos scripts pour exec le code (dans src/),
dites moi ce que vous en pensez.

Comme le modèle je le trouvais pas trop complexe et l'environnement PDG aussi j'ai tout fait avec numpy et j'ai pas utiliser PettingZoo, je sais pas si ca va poser problème pour les résulats ou avec le PGG.

Pour comprendre ce que j'ai fait, 
    dans BMmodel.py : 
    - c'est juste l'application des 2 formules du paper 
    - j'ai fait un fonctions fix01 pour bien rester entre [0,1] quand on calcul les proba

    dans agents.py :
    - c'est assez clair je trouve juste regardez les fct et on comprend, juste à la ligne 32 c'est le choix de l'action en prenant en compte le bruit

    dans PDGenv.py:
    - classique encore une fois juste dans step() on calcul bien le mean reward de tous les voisins.

    dans loop.py ca c'est chatgpt qui a fait la loop