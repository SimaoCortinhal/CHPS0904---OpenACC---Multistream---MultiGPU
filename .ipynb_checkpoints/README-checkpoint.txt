Commande salloc fonctionnelle pour tous les projets (multigpu, multistreams et openacc

salloc --nodes=1 --time=4:00:00 --account=r250059 --constraint="armgpu" --mem=6G --gpus-per-node=4 --ntasks=4 --cpus-per-task=4


Récapitulatif des commandes :
1.	Sur le nœud de calcul exemple=romeo-a057 :
spack load cuda@12.6
spack laod openmpi
spack load nccl (spack install nccl cuda_arch=90 si pas fait)

Ajouter : spack load nvhpc pour openACC

jupyter lab --no-browser --port=7777
Configurer le tunneling SSH (si nécessaire) pour accéder à JupyterLab depuis votre machine locale :
Par exemple, via deux sauts :
•	Depuis le nœud de login (création d’un tunnel entre le nœud de login et le nœud de calcul) :
•	ssh -N -L 4567:localhost:7777 username@romeo-a057
Depuis votre machine locale (rediriger localement vers le nœud de login) :
sudo ssh -i /etc/ssh/romeo -N -L 9999:localhost:4567 username@romeo1.univ-reims.fr
Ensuite, connectez-vous à JupyterLab via l’URL :
http://localhost:9999
