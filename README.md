TP3 - Thomas Cenci, Richard Nguyen et Théodore Simon

Utilisation de evaluate_lr.sh
1)Le script evaluate_lr.sh utilise les données "02Heart.hdf5" qui ne sont pas sur dépot git car trop volumineux. Pour pouvoir utiliser le script, il faut donc ajouter manuellement les données dans le répertoire /prog
2)Le script utilise un environnement virtuel python avec les dépendances du requirements.txt
  -Se mettre dans le dossier /prog
  -Ecrire:
     virtualenv env
     source env/bin/activate
     pip install -r requirements.txt
  -Installer la version torch et cuda adapter à votre machine. Torch a été retiré des requirements car la version est dépendant de la version de cuda installée sur la machine.
