# TP3 du cours IFT780 d'Antoine Théberge - Thomas Cenci, Richard Nguyen et Théodore Simon

Les données "02Heart.hdf5" pour l'entrainement dans le projet ne sont pas sur le dépot GitHub car trop volumineux. **Veuillez les ajouter manuellement dans le dossier /prog.**

## Exemple d'entrainement utilisant train.py  
!python train.py acdc_train --model=FullNet --dgr=4 --dnl=4 --load_checkpoint --save_checkpoint --dataset='./02Heart.hdf5' --num-epochs=10 --batch_size=2

## Utilisation de evaluate_lr.sh
Le script utilise un environnement virtuel python avec les dépendances du requirements.txt  
1. Se mettre dans le dossier /prog  
2. Installer l'environnement virtuel:  
* virtualenv env  
* source env/bin/activate  
* pip install -r requirements.txt  
3. Installer les versions de torch et cuda adaptées à votre machine. Torch a été retiré du requirements.txt car la version étant dépendante de la version de cuda installée et du gpu de la machine, cela ne fonctionnait pas sur toutes les machines.  
4. Exemple d'exécution: ./evaluate_lr.sh FullNet 10 0.00075 0.00125 0.00005

## Références

[1] ImageNet Classification with Deep Convolutional Neural Networks

[2] Improving Nuclei/Gland Instance Segmentation in Histopathology Images by Full Resolution Neural Network and Spatial Constrained Loss

[3] [1608.06993] Densely Connected Convolutional Networks

[4] [1505.04597] U-Net: Convolutional Networks for Biomedical Image Segmentation
