# TP6 — Accélérer un ResNet

## Exercice 0 — Mise en place

### 1. Vérification de l'environnement

La sortie du script `check_env.py` est la suivante :

```text
Machine        : arcadia-slurm-node-3
Python         : 3.10.21
torch          : 2.14.0+cu130 (CUDA 13.0)
torchvision    : 0.28.0
torchao        : 0.18.0
CPU alloués    : 4  (x86_64)
Drapeaux CPU   : avx512f, avx512_vnni, avx512_bf16
gcc            : /usr/bin/gcc
GPU            : NVIDIA H100 NVL
Mémoire GPU    : 95321 Mio au total, 94795 Mio libres
bf16 supporté  : True
Données raw    : /mnt/ssd/imagenet/2012 -> (12894, 500) images (train, val)
Données small  : /tmp/mboukraa/tp6/imagenet10_256 -> (12894, 500) images (train, val)
Poids ResNet-50 pré-entraînés : OK
```

Le GPU utilisé est un NVIDIA H100 NVL disposant de 95 321 Mio de mémoire totale. Quatre CPU sont alloués au job. Le GPU prend en charge le format BF16.

L'environnement logiciel est constitué de Python 3.10.21, PyTorch 2.14.0+cu130, Torchvision 0.28.0 et TorchAO 0.18.0.

Les deux jeux de données sont accessibles et contiennent chacun 12 894 images d'entraînement et 500 images de validation. Enfin, les poids pré-entraînés du ResNet-50 ont été chargés avec succès.

### 2. Conséquence du nombre de CPU sur le chargement des images

Nous disposons de seulement 4 CPU pour alimenter un septième de H100. Le chargement et le prétraitement des images peuvent donc devenir un goulot d'étranglement.

En effet, ces opérations nécessitent du temps CPU, notamment pour lire les fichiers JPEG, décoder les images, effectuer les redimensionnements, les recadrages et les augmentations de données.

Les quatre CPU risquent de ne pas préparer les lots d'images suffisamment rapidement pour alimenter le GPU en continu. Le GPU pourrait alors rester momentanément inactif en attendant les données, ce qui réduirait son taux d'utilisation et les performances globales de l'entraînement.

L'utilisation de plusieurs workers dans le DataLoader, du chargement asynchrone et de la mémoire épinglée (pin_memory) peut aider à limiter ce problème.
```


# Exercice 1 — Benchmark de référence

## 1. Benchmark synthétique et données réelles

Les benchmarks ont été réalisés sur un GPU NVIDIA H100 NVL, avec ResNet-50, en précision FP32 et un batch de 64.

| Mesure | Données synthétiques | Images réelles |
|---|---:|---:|
| Débit moyen | 710,5 images/s | 696,2 images/s |
| Débit minimal | 709,0 images/s | 694,2 images/s |
| Débit maximal | 710,8 images/s | 696,2 images/s |
| Pic mémoire GPU | 5,95 Go | 5,95 Go |
| Durée d'une époque | — | 19,4 s |

Pour les images réelles, le DataLoader utilise 3 workers et `pin_memory=True`.

La différence entre les deux débits est d'environ 2 %. Le chargement et le prétraitement des images réelles ont donc un impact relativement limité dans cette configuration.

## 2. Estimation de la mémoire GPU

La mémoire utilisée pendant l'entraînement comprend plusieurs éléments :
- Les paramètres du modèle.
- Les gradients associés aux paramètres.
- Les deux états de l'optimiseur AdamW (premier et deuxième moments).
- Les activations intermédiaires conservées pour la rétropropagation.
- Les données d'entrée et les buffers temporaires.

ResNet-50 possède environ 25,6 millions de paramètres. En FP32, les paramètres occupent environ 102 Mo. Les gradients occupent également environ 102 Mo, et les deux états d'AdamW environ 205 Mo supplémentaires.

Ces éléments représentent environ 410 Mo, sans compter les activations et les allocations temporaires. Le pic mémoire réellement mesuré est de 5,95 Go pour un batch de 64.

## 3. Exécution asynchrone CUDA

La mesure du benchmark asynchrone donne :
- Temps de retour de l'appel CPU : 12,1 ms.
- Temps jusqu'à la fin réelle du calcul GPU : 87,7 ms.

CUDA exécute les opérations de manière asynchrone : le CPU peut continuer son exécution sans attendre la fin du calcul GPU. Une mesure sans synchronisation sous-estime donc le temps réel d'exécution. Il faut utiliser `torch.cuda.synchronize()` pour obtenir une mesure correcte.

## 4. Recherche du batch maximal

Le batch maximal obtenu en FP32 est de 1128 images sur le GPU NVIDIA H100 NVL.

Pendant la recherche, plusieurs allocations mémoire échouent lorsque la taille du batch devient trop importante. Ces erreurs Out Of Memory sont attendues dans cette expérience : elles permettent de déterminer la limite de mémoire disponible.

## 5. Comparaison et conclusion

Le débit passe de 710,5 images/s avec des données synthétiques à 696,2 images/s avec les images JPEG réelles, soit une diminution d'environ 2 %.

Cette faible différence indique que, dans cette configuration avec trois workers et `pin_memory=True`, le chargement des données ne constitue pas un goulot d'étranglement majeur.

Le benchmark fournit ainsi une référence de performance pour les optimisations qui seront étudiées dans les exercices suivants.



# TP6 — Exercice 2 : Chargement des données

## 1. Tableau des débits

Les benchmarks ont été réalisés sur un GPU NVIDIA H100 NVL avec ResNet50, en précision FP32, et une taille de batch de 64.

| Configuration | Workers | pin_memory | Débit (img/s) |
|---|---:|:---:|---:|
| ref-synth | - | - | 710.5 |
| ref-raw | 3 | True | 696.2 |
| raw-w0 | 0 | True | 219.9 |
| raw-w1 | 1 | True | 331.2 |
| raw-w2 | 2 | True | 718.7 |
| raw-w3 | 3 | True | 720.6 |
| raw-w3-nopin | 3 | False | 647.7 |
| small-w0 | 0 | True | 331.0 |
| small-w3 | 3 | True | 710.8 |

Le meilleur débit mesuré est obtenu avec la configuration `raw-w3`, avec 720.6 images/s. La référence synthétique atteint 710.5 images/s.

## 2. Influence du nombre de workers

On observe une forte amélioration du débit lorsque le nombre de workers augmente :

- 0 worker : 219.9 images/s.
- 1 worker : 331.2 images/s.
- 2 workers : 718.7 images/s.
- 3 workers : 720.6 images/s.

Le gain le plus important est observé entre 1 et 2 workers. Le passage de 2 à 3 workers apporte seulement une amélioration marginale.

Les workers permettent de paralléliser le chargement et le prétraitement des images pendant que le GPU effectue les calculs du réseau de neurones.

Avec 4 CPU disponibles, l'utilisation de 2 workers suffit pratiquement à atteindre le débit maximal. Le troisième worker n'apporte qu'un gain très limité.

**Conclusion :** deux workers suffisent pratiquement à saturer le GPU dans cette configuration.

## 3. Influence de pin_memory

On compare les deux configurations suivantes :

| Configuration | pin_memory | Débit (img/s) |
|---|:---:|---:|
| raw-w3 | True | 720.6 |
| raw-w3-nopin | False | 647.7 |

La désactivation de `pin_memory` entraîne une diminution du débit d'environ 10.1 %.

En effet, `pin_memory=True` utilise une mémoire hôte verrouillée (page-locked memory), ce qui facilite les transferts de données entre la mémoire CPU et la mémoire GPU.

Cette optimisation permet donc d'améliorer les performances du transfert des données et de mieux alimenter le GPU.

## 4. Influence de la taille des images

On compare les résultats des configurations raw et small avec zéro worker :

| Configuration | Débit (img/s) |
|---|---:|
| raw-w0 | 219.9 |
| small-w0 | 331.0 |

La configuration small atteint un débit supérieur d'environ 50.5 %.

Cela s'explique par le fait que les images de plus petite taille nécessitent moins de données à lire, décoder et transférer en mémoire.

Le redimensionnement réduit donc le coût du prétraitement des images. Cependant, les images redimensionnées ne sont plus strictement identiques pixel par pixel aux images originales.

Avec trois workers, les débits deviennent très proches :

- raw-w3 : 720.6 images/s.
- small-w3 : 710.8 images/s.

Le parallélisme permet ainsi de masquer une grande partie du coût du chargement et du prétraitement.

## 5. Estimation du temps de décodage d'une image

On utilise la formule suivante :

T_decode = 1 / D_0 - 1 / D_synth

Avec :
- D_synth = 710.5 images/s : débit de référence synthétique.
- D_0 : débit mesuré sans workers.

### Cas des images raw

D_0 = 219.9 images/s.

T_decode = 1 / 219.9 - 1 / 710.5

T_decode ≈ 0.00314 s = 3.14 ms.

Le débit théorique d'un worker est donc :

D_worker = 1 / T_decode

D_worker ≈ 318.5 images/s.

Pour trois workers :

D_3_workers ≈ 3 × 318.5 = 955.5 images/s.

### Cas des images small

D_0 = 331.0 images/s.

T_decode = 1 / 331.0 - 1 / 710.5

T_decode ≈ 0.00161 s = 1.61 ms.

Le débit théorique d'un worker est :

D_worker = 1 / T_decode

D_worker ≈ 619.7 images/s.

Pour trois workers :

D_3_workers ≈ 3 × 619.7 = 1859.1 images/s.

### Comparaison

| Mesure | Raw | Small |
|---|---:|---:|
| Temps de décodage estimé | 3.14 ms | 1.61 ms |
| Débit théorique d'un worker | 318.5 img/s | 619.7 img/s |
| Débit théorique de trois workers | 955.5 img/s | 1859.1 img/s |

La version small nécessite environ deux fois moins de temps de décodage que la version raw.

Dans les deux cas, la capacité théorique de trois workers dépasse le débit du GPU. Cela explique pourquoi le chargement des données ne constitue plus un goulot d'étranglement important avec plusieurs workers.

Ces valeurs restent des estimations basées sur le modèle simplifié de l'énoncé.

## 6. Analyse du profiling avec Perfetto

Le profiling permet d'observer les principales opérations exécutées pendant l'entraînement de ResNet50.

Les résultats obtenus montrent notamment :

| Opération | Part du temps CUDA |
|---|---:|
| Convolution backward | 43.46 % |
| ProfilerStep | 23.89 % |
| Calcul des gradients (wgrad) | 19.68 % |
| Calcul des gradients (dgrad) | 17.48 % |
| Convolution forward | 16.33 % |

Ces pourcentages correspondent aux mesures du profiler. Certaines opérations étant imbriquées, leurs proportions ne doivent pas être additionnées.

Les convolutions occupent une place importante dans le calcul du réseau, notamment lors de la propagation avant et du calcul des gradients.

Les débits mesurés avec trois workers sont très proches de la référence synthétique :

- ref-synth : 710.5 images/s.
- raw-w3 : 720.6 images/s.
- small-w3 : 710.8 images/s.

Cela indique que le chargement des données est suffisamment rapide pour alimenter efficacement le GPU.

Le système est donc principalement limité par les calculs du modèle (compute-bound), plutôt que par le chargement des images.

Les traces Perfetto permettent de vérifier cette interprétation en observant l'activité du GPU et les éventuelles périodes d'inactivité.

**Captures à ajouter au rapport :**

- Trace Perfetto de raw-w3.
- Trace Perfetto de small-w3.

## 7. Fraction du débit synthétique atteinte

On calcule la fraction du débit synthétique atteinte par la meilleure configuration :

Fraction = (D_raw-w3 / D_synth) × 100

Fraction = (720.6 / 710.5) × 100

Fraction ≈ 101.42 %

La configuration raw-w3 atteint donc environ 101.4 % du débit synthétique de référence.

Cette valeur légèrement supérieure à 100 % peut s'expliquer par les variations expérimentales entre les benchmarks. Elle ne signifie pas que le chargement des données rend le GPU intrinsèquement plus rapide que dans le test synthétique.

## Conclusion de l'exercice 2

Les expérimentations montrent que le nombre de workers et l'utilisation de pin_memory ont un impact important sur les performances du chargement des données.

Le passage de zéro à deux workers permet de passer de 219.9 à 718.7 images/s. L'ajout d'un troisième worker apporte ensuite un gain marginal.

L'utilisation de pin_memory améliore également le débit, avec une différence d'environ 10.1 % par rapport à la configuration sans cette option.

Enfin, la meilleure configuration atteint 720.6 images/s, soit environ 101.4 % du débit synthétique de référence.

**En conclusion, deux workers avec pin_memory=True suffisent pratiquement à exploiter toute la capacité de calcul du GPU NVIDIA H100 NVL dans cette expérience.**






## Exercice 5 — Quantification INT8 sur CPU

### 1. Résultats de calib200

Les mesures ont été réalisées sur 500 images de validation, avec 4 threads CPU.

| Variante | Taille (Mo) | Latence (ms) | Premier appel (s) | Top-1 |
|---|---:|---:|---:|---:|
| FP32 eager | 94,3 | 34,66 | 0,0 | 99,4 % |
| FP32 compilé | 94,3 | 21,28 | 24,8 | 99,4 % |
| BF16 | 47,2 | 11,83 | 4,7 | 99,4 % |
| INT8 (calibration 200) | 23,9 | 7,69 | 4,2 | 99,4 % |

La compilation FP32 améliore déjà les performances par rapport à l'exécution eager. BF16 et INT8 réduisent davantage la latence et la taille du modèle. INT8 obtient ici les meilleures performances.

### 2. Rapport de taille FP32 / INT8

Le rapport de taille est :

94,3 / 23,9 ≈ 3,95.

Il est donc proche de 4, mais pas exactement égal à 4. En théorie, les poids FP32 occupent 4 octets par valeur, contre 1 octet en INT8. Cependant, le modèle quantifié contient également des informations supplémentaires, notamment les échelles de quantification, les points zéro, les biais et d'autres tenseurs nécessaires à l'exécution.

La fonction `weights_mb` distingue deux cas. Pour un `GraphModule`, elle récupère les tenseurs réellement référencés par les nœuds `get_attr` du graphe. Pour un module classique, elle utilise les tenseurs du `state_dict`.

On ne mesure donc pas simplement le `state_dict` du modèle converti, car celui-ci peut encore contenir des paramètres FP32 d'origine qui ne sont plus utilisés par le calcul quantifié. Compter ces paramètres donnerait une estimation trompeuse de la mémoire réellement mobilisée par le graphe.

### 3. Comparaison des latences et intérêt des instructions CPU

Les drapeaux CPU relevés à l'exercice 0 indiquent la présence de `avx512f`, `avx512_vnni` et `avx512_bf16`.

La présence de VNNI est particulièrement intéressante pour INT8 : cette extension accélère les opérations de multiplication et d'accumulation sur des entiers, utilisées dans les calculs quantifiés. Cela peut contribuer au gain de performances observé.

Le processeur prend également en charge BF16 grâce à `avx512_bf16`, ce qui est cohérent avec les bonnes performances de cette variante (11,83 ms). En revanche, aucun flag AMX n'a été relevé. On ne peut donc pas attribuer les gains mesurés à AMX.

### 4. Précision et rôle de la calibration

Sur les 500 images de validation, les trois variantes FP32, BF16 et INT8 obtiennent un top-1 de 99,4 %.

Pour INT8 avec 200 images de calibration :
- `accord_fp32` = 1,0000, soit 100 %.
- `ecart_logits` = 0,0923.

Le top-1 ne suffit pas à juger la quantification. Il mesure uniquement la proportion de bonnes prédictions, tandis que `accord_fp32` mesure la proportion de prédictions identiques à celles du modèle FP32. `ecart_logits` mesure la différence numérique moyenne entre les logits des deux modèles. Deux modèles peuvent donc avoir le même top-1 tout en produisant des scores différents.

Comparaison des deux calibrations :

| Calibration | Latence (ms) | Accord FP32 | Écart logits |
|---|---:|---:|---:|
| 200 images | 7,69 | 100 % | 0,0923 |
| 10 images | 7,65 | 100 % | 0,0929 |

Les résultats sont presque identiques. La calibration consiste à faire passer des données représentatives dans le modèle préparé afin que les observateurs estiment les plages de valeurs des activations. Ces estimations permettent ensuite de déterminer les échelles et les points zéro utilisés pour la quantification.

Dans cette expérience, 10 images suffisent à obtenir des résultats très proches de ceux obtenus avec 200 images. Cela dépend toutefois de la représentativité des données de calibration et ne peut pas être généralisé à tous les jeux de données.

### 5. Influence du nombre de threads

Comparaison entre 1 et 4 threads :

| Variante | 1 thread (ms) | 4 threads (ms) | Accélération |
|---|---:|---:|---:|
| FP32 compilé | 72,60 | 21,28 | 3,41× |
| INT8 | 21,34 | 7,69 | 2,78× |

Le passage de 1 à 4 threads améliore nettement les performances, mais l'accélération n'est pas proportionnelle au nombre de threads : elle reste inférieure à 4×.

Cela s'explique par les coûts de synchronisation, les accès mémoire, les limites de parallélisme et les ressources partagées du processeur.

Pour un déploiement, il faut donc mesurer les performances avec différents nombres de threads et choisir le meilleur compromis selon la machine et la charge. Utiliser davantage de threads n'est pas systématiquement plus efficace, notamment lorsque plusieurs applications partagent le CPU.

### 6. Pourquoi la quantification dynamique est-elle peu efficace sur ResNet ?

ResNet est principalement constitué de couches convolutives (`Conv2d`), qui réalisent l'essentiel du calcul du réseau.

La quantification dynamique classique cible surtout les couches linéaires (`Linear`) : les poids sont quantifiés à l'avance, tandis que les activations sont quantifiées dynamiquement au moment de l'inférence. Elle est donc particulièrement adaptée aux modèles dominés par les couches linéaires, comme certains réseaux de traitement du langage.

Dans ResNet, la majorité du coût de calcul se trouve dans les convolutions. Une quantification dynamique limitée principalement aux couches linéaires ne réduit donc qu'une petite partie du coût total. La quantification statique, qui prend en charge les convolutions et leurs activations, est ainsi beaucoup plus intéressante pour ce type de réseau.

**Conclusion :** la quantification statique INT8 permet ici de réduire la taille du modèle d'environ 75 % et d'obtenir une latence environ 2,77 fois plus faible que le FP32 compilé, tout en conservant les mêmes prédictions sur les 500 images évaluées.






## Exercice 3 — Optimisation des performances GPU

### 1. Présentation des mesures

| Configuration | Données | Débit (images/s) | Pic mémoire (Go) | Compilation (s) |
|---|---|---:|---:|---:|
| fp32-synthetic | Synthétiques | 734,8 | 5,95 | — |
| tf32-synthetic | Synthétiques | 1 679,6 | 5,93 | — |
| bf16-synthetic | Synthétiques | 1 987,8 | 3,29 | — |
| bf16-cl-synthetic | Synthétiques | 2 713,1 | 3,29 | — |
| bf16-cl-compile-synthetic | Synthétiques | 4 317,9 | 3,07 | 44,8 |
| bf16-cl-raw-w1 | Raw, 1 worker | 272,4 | 3,29 | — |
| bf16-cl-compile-raw-w1 | Raw, 1 worker | 335,5 | 3,07 | 4,0 |
| bf16-cl-raw-w3 | Raw, 3 workers | 2 878,8 | 3,29 | — |
| bf16-cl-compile-raw-w3 | Raw, 3 workers | 1 311,7 | 3,07 | 3,8 |
| bf16-cl-small-w1 | Small, 1 worker | 484,2 | 3,29 | — |
| bf16-cl-compile-small-w1 | Small, 1 worker | 502,5 | 3,07 | 3,5 |
| bf16-cl-small-w3 | Small, 3 workers | 1 547,6 | 3,29 | — |
| bf16-cl-compile-small-w3 | Small, 3 workers | 2 167,9 | 3,07 | 3,7 |

### 2. Comparaison de fp32-synthetic avec ref-synth

Le débit de `ref-synth` est de 710,5 images/s contre 734,8 images/s pour `fp32-synthetic`, soit un écart relatif de **3,4 %**. Les mesures sont proches, mais montrent une légère variabilité expérimentale. Les petites différences de performance doivent donc être interprétées avec prudence.

### 3. Gains successifs en données synthétiques

- TF32 : gain de **×2,29**.
- BF16 : gain de **×1,18**.
- `channels_last` : gain de **×1,36**.
- `compile` : gain de **×1,59**.

Les Tensor Cores accélèrent les calculs en TF32 et BF16. BF16 réduit également le volume des données manipulées et le trafic mémoire. `channels_last` optimise l'organisation des tenseurs. Enfin, `torch.compile` peut fusionner plusieurs opérations et réduire les lancements de kernels.

### 4. Comparaison synthétique, raw et small

En synthétique, le débit atteint 2 713,1 images/s sans compilation et 4 317,9 images/s avec compilation.

Avec les données réelles, le débit dépend fortement du nombre de workers :
- Raw : 272,4 images/s avec 1 worker et 2 878,8 avec 3 workers.
- Small : 484,2 images/s avec 1 worker et 1 547,6 avec 3 workers.

L'estimation de l'exercice 2 est confirmée : le chargement des données peut devenir le goulot d'étranglement. La compilation améliore légèrement les performances avec 1 worker, mais ses effets sont variables avec 3 workers. La dispersion de `bf16-cl-raw-w1` (224,9 à 309,7 images/s) montre une variabilité importante.

Selon la loi d'Amdahl, accélérer le calcul GPU ne suffit pas si le chargement des données reste limitant.

### 5. Estimation du nombre de workers

Un MIG 1g.12gb dispose d'environ un septième des multiprocesseurs d'un H100. Il faudrait donc environ **21 workers** pour alimenter un H100 entier, en partant de 3 workers pour le MIG. Les 30 cœurs d'Arcadia pourraient théoriquement suffire, sous réserve des limites du stockage et des autres ressources.

### 6. Coût de compilation et amortissement

Les temps de compilation mesurés sont de 44,8 s pour la configuration synthétique, puis entre 3,5 et 4 s pour les configurations réelles. La première compilation est plus longue en raison des coûts d'initialisation.

Avec un gain de débit synthétique de 1 604,8 images/s, il faudrait environ 5 110 itérations pour amortir 44,8 s, contre environ 400 itérations pour 3,5 s. Pour deux époques de 201 pas, la première compilation n'est donc pas rentable ; la seconde est à peine amortie.

### 7. Comparaison des batchs maximaux

Le batch maximal est de 1 128 en FP32 et de 1 104 en BF16. Ce résultat ne correspond pas au gain mémoire théorique du BF16, car les activations, les buffers et les espaces de travail temporaires influencent également la mémoire consommée.

### 8. Temps des époques et validation

Les temps exacts des deux époques et le top-1 de validation doivent être relevés dans la sortie de `train.py`. La première époque est généralement plus lente à cause de l'initialisation et des opérations de préparation. Le facteur d'accélération total se calcule en divisant le temps de l'époque de référence `ref-raw` (19,4 s) par le temps de la deuxième époque.



# TP6 — Exercice 4 : Entraînement distribué multi-nœuds avec DDP

## 1. Transport NCCL et débit réseau

D'après le fichier `logs/nccl-56647.log`, NCCL utilise le transport Socket sur l'interface réseau `ens18`. Aucun périphérique InfiniBand n'a été détecté (`No device found`). La communication entre les deux nœuds passe donc par des sockets réseau.

Le test d'all-reduce porte sur 102 Mo et donne les résultats suivants :
- Durée : 1240,8 ms.
- Débit algorithmique : 0,08 Go/s, soit 0,64 Gbit/s.
- Débit par lien : 0,04 Go/s, soit 0,32 Gbit/s.

Le transport utilisé correspond à une communication via le réseau Ethernet avec des sockets, et non à InfiniBand. Les débits mesurés sont faibles et constituent un facteur limitant pour l'entraînement distribué.

## 2. Tableau des six mesures

Les accélérations sont calculées par rapport à la configuration à un GPU ayant la même accumulation. L'efficacité est calculée par E = S / 2.

| Expérience | Configuration | Temps/pas (ms) | Débit (img/s) | S | E |
|---|---|---:|---:|---:|---:|
| A | All-reduce 102 Mo | 1240,8 | — | — | — |
| B | 1 GPU, accumulation 1 | 138,2 | 462,9 | 1,00 | 50 % |
| C | 2 GPU, accumulation 1 | 865,6 | 147,9 | 0,32 | 16,0 % |
| D | 2 GPU + BF16 hook, accumulation 1 | 463,2 | 276,4 | 0,60 | 29,9 % |
| E | 1 GPU, accumulation 8 | 1052,1 | 486,6 | 1,00 | 50 % |
| F | 2 GPU + BF16 hook, accumulation 8 | 1375,3 | 744,5 | 1,53 | 76,5 % |

Pour l'expérience C :

S = 147,9 / 462,9 = 0,32
E = 0,32 / 2 = 16 %

Pour l'expérience F :

S = 744,5 / 486,6 = 1,53
E = 1,53 / 2 = 76,5 %

Sans optimisation, le passage à deux GPU dégrade les performances : le débit diminue de 462,9 à 147,9 images/s. Avec le hook BF16 et l'accumulation de huit micro-batchs, le débit atteint 744,5 images/s. Le calcul distribué devient alors plus performant que la référence à un GPU.

## 3. Estimation du coût de communication et du recouvrement

Le modèle possède 23,5 millions de paramètres. Chaque paramètre étant représenté par 4 octets en FP32, la taille des gradients est :

23,5 × 10^6 × 4 = 94 × 10^6 octets, soit environ 94 Mo.

Le test NCCL a mesuré 1240,8 ms pour 102 Mo. En supposant une évolution proportionnelle à la taille des données :

tc = 1240,8 × 94 / 102 = 1143,5 ms.

Le temps de référence sur un GPU est t1 = 138,2 ms.

Sans recouvrement :

T = t1 + tc = 138,2 + 1143,5 = 1281,7 ms.

Avec un recouvrement parfait :

T = max(t1, tc) = 1143,5 ms.

Le temps mesuré sur deux GPU sans hook (expérience C) est de 865,6 ms.

La durée recouverte estimée avec le modèle demandé est :

trec = t1 + tc - T2
trec = 138,2 + 1143,5 - 865,6
trec = 416,1 ms.

Cette estimation doit être interprétée avec prudence : elle dépasse le temps de calcul de référence et le temps mesuré est inférieur à la borne théorique de recouvrement parfait. Le test NCCL isolé ne reproduit pas exactement les communications DDP réelles. DDP peut notamment découper les gradients en buckets et recouvrir les communications avec le calcul du backward. Les 416,1 ms constituent donc une estimation issue du modèle simplifié, et non une mesure exacte du temps caché.

## 4. Effet du BF16 hook et risques

Le hook BF16 convertit les gradients FP32 en BF16 avant leur communication. Cette conversion réduit approximativement de moitié la quantité de données à transmettre.

Le débit passe de 147,9 images/s sans hook à 276,4 images/s avec hook, soit une augmentation d'environ 86,9 %.

Cette amélioration s'explique par la réduction du volume de communication. En revanche, la conversion en BF16 entraîne une perte de précision numérique. Elle peut modifier les gradients, affecter la convergence et éventuellement dégrader la qualité finale du modèle. Il faut donc vérifier la stabilité de l'entraînement et les performances sur les données de validation.

## 5. Loi d'Amdahl et prédiction de l'expérience F

La loi d'Amdahl suppose une proportion fixe de calcul parallélisable et une proportion fixe de calcul séquentiel. Elle ne prend pas en compte le coût variable des communications réseau, leur recouvrement avec le calcul, ni les optimisations comme la compression BF16 et l'accumulation des gradients.

Pour prédire l'expérience F, on utilise :

T2 ≈ k × t1 + tc - trec.

Avec :
- k = 8 micro-batchs ;
- t1 = 138,2 ms ;
- tc = 1143,5 / 2 = 571,75 ms avec la compression BF16 ;
- trec = 416,1 ms.

La prédiction est :

T2 ≈ 8 × 138,2 + 571,75 - 416,1
T2 ≈ 1261,3 ms.

Le temps réellement mesuré pour l'expérience F est de 1375,3 ms. L'écart est de 114 ms, soit environ 9 % par rapport à la prédiction.

Cet écart peut s'expliquer par les approximations du modèle, les coûts réels des synchronisations DDP et les différences entre le test NCCL isolé et les communications effectuées pendant l'entraînement.

## 6. Conditions de rentabilité du DDP multi-nœuds

Le DDP multi-nœuds devient rentable lorsque le gain apporté par le calcul parallèle dépasse le coût des communications et des synchronisations réseau.

Deux leviers permettent d'améliorer les performances :

1. L'accumulation des gradients : plusieurs micro-batchs sont traités avant la synchronisation. Cela réduit la fréquence des communications. Son prix est l'augmentation du batch effectif, qui peut modifier la dynamique d'apprentissage.

2. La compression BF16 : elle réduit la quantité de données transmises et donc le coût des communications. Son prix est une perte de précision numérique pouvant affecter la convergence.

Les résultats montrent que ces optimisations sont particulièrement utiles lorsque le réseau constitue un goulot d'étranglement.

## 7. Ajustement du taux d'apprentissage

Le batch effectif est calculé par :

Batch effectif = batch par processus × nombre de processus × accumulation.

Pour deux GPU :
- Accumulation 1 : 64 × 2 = 128.
- Accumulation 8 : 64 × 2 × 8 = 1024.

Le batch initial de référence est 64.

Avec SGD, on peut appliquer la règle de mise à l'échelle linéaire comme point de départ :
- Batch 128 : multiplier le learning rate par 2.
- Batch 1024 : multiplier le learning rate par 16.

Un warm-up est recommandé pour éviter une augmentation trop brutale du taux d'apprentissage.

Avec AdamW, une augmentation plus prudente peut être utilisée, par exemple selon la règle en racine carrée :
- Batch 128 : facteur √2, soit environ 1,41.
- Batch 1024 : facteur √16, soit 4.

Ces facteurs sont des points de départ et doivent être validés expérimentalement en observant la convergence.

## 8. Initialisation, échantillonnage et synchronisation

### Même graine pour l'initialisation

Tous les processus utilisent la même graine afin de garantir une initialisation identique des paramètres du modèle. Les données sont cependant réparties entre les processus grâce au DistributedSampler, afin que chaque GPU traite une partie différente du jeu de données.

### Oubli de sampler.set_epoch(epoch)

Cette instruction permet de modifier l'ordre de mélange des données à chaque époque. Sans elle, le DistributedSampler réutilise le même ordre à chaque époque. Cela réduit la diversité de l'échantillonnage et peut affecter la généralisation du modèle.

### Oubli de model.no_sync()

Pendant l'accumulation des gradients, model.no_sync() désactive les synchronisations DDP pour les micro-batchs intermédiaires. La synchronisation est effectuée uniquement après le dernier micro-batch.

Sans cette instruction, les gradients sont synchronisés après chaque backward. Avec huit micro-batchs, cela entraîne huit synchronisations au lieu d'une seule, ce qui augmente fortement le coût des communications et réduit l'intérêt de l'accumulation.

## Conclusion

Ce TP a permis de mettre en évidence l'importance du réseau dans l'entraînement distribué multi-nœuds. L'utilisation de deux GPU n'améliore pas systématiquement les performances : sans optimisation, le coût des communications peut dépasser le gain du parallélisme.

La compression BF16 et l'accumulation des gradients permettent de réduire le coût des communications et d'améliorer l'efficacité du DDP. La meilleure configuration testée est celle utilisant deux GPU, le hook BF16 et une accumulation de huit micro-batchs. Elle atteint 744,5 images/s, contre 486,6 images/s sur un GPU, soit une accélération de 1,53 et une efficacité parallèle de 76,5 %.

Les performances d'un entraînement distribué dépendent donc autant de la puissance de calcul que de la bande passante réseau, de la fréquence des synchronisations et des optimisations appliquées aux communications.







## Rapport final — Synthèse du TP6

### 1. Tableau comparatif des configurations

Le tableau suivant présente les principales configurations évaluées pendant le TP. Les mesures synthétiques permettent d'isoler les performances de calcul, tandis que les mesures avec données réelles prennent également en compte le chargement des images.

| Configuration | Matériel | Données | Précision | Débit (images/s) | Pic mémoire / taille | Temps par époque |
|---|---|---|---|---:|---:|---:|
| Référence (`ref-synth`) | H100 NVL | Synthétiques | FP32 | 710,5 | 5,95 Go | — |
| Référence (`ref-raw`) | H100 NVL | Raw | FP32 | 696,2 | 5,95 Go | 19,4 s |
| Optimisée (`bf16-cl-compile-synthetic`) | H100 NVL | Synthétiques | BF16 | 4317,9 | 3,07 Go | — |
| Optimisée (`bf16-cl-raw-w3`) | H100 NVL | Raw | BF16 | 2878,8 | 3,29 Go | — |
| DDP (`l4-x1`) | 1 L4 | Synthétiques | BF16 | 462,9 | Non mesuré | — |
| DDP (`l4-x2`) | 2 L4 | Synthétiques | BF16 | 147,9 | Non mesuré | — |
| DDP (`l4-x2-bf16hook`) | 2 L4 | Synthétiques | BF16 + hook | 276,4 | Non mesuré | — |
| DDP (`l4-x1-accum8`) | 1 L4 | Synthétiques | BF16, accumulation 8 | 486,6 | Non mesuré | — |
| DDP (`l4-x2-accum8-bf16hook`) | 2 L4 | Synthétiques | BF16, accumulation 8 + hook | 744,5 | Non mesuré | — |
| Inférence FP32 compilée | CPU, 4 threads | Validation | FP32 | 47,0 | Modèle : 94,3 Mo | — |
| Inférence BF16 | CPU, 4 threads | Validation | BF16 | 84,5 | Modèle : 47,2 Mo | — |
| Inférence INT8 | CPU, 4 threads | Validation | INT8 | 130,0 | Modèle : 23,9 Mo | — |

Pour l'inférence CPU, le débit est estimé à partir de l'inverse de la latence mesurée sur une image : débit ≈ 1000 / latence en ms. La taille indiquée correspond aux tenseurs du modèle mesurés par `weights_mb`, et non au pic mémoire du processus.

La meilleure performance en calcul synthétique est obtenue avec BF16, `channels_last` et `torch.compile`, avec 4317,9 images/s.

Avec les données réelles, la configuration BF16 avec `channels_last` et trois workers, sans compilation, atteint 2878,8 images/s. Cette mesure présente toutefois une variabilité importante : les valeurs observées vont de 2039,2 à 3301,4 images/s. Il faut donc interpréter ce résultat avec prudence.

La meilleure configuration DDP parmi celles testées utilise deux L4, une accumulation de gradients sur 8 itérations et le hook BF16, avec 744,5 images/s.

Enfin, sur CPU, INT8 offre la meilleure latence et la plus faible taille de modèle.

### 2. Analyse des écarts à l'accélération idéale en DDP

L'objectif du DDP est de répartir le calcul entre plusieurs GPU afin d'augmenter le débit. Dans un cas idéal, passer de 1 à 2 GPU devrait presque doubler le débit. Les mesures montrent cependant que ce résultat dépend fortement de la communication et de la configuration utilisée.

**Configuration standard :**

Avec un seul L4, le débit est de 462,9 images/s. Avec deux L4 sans hook BF16, il tombe à 147,9 images/s.

Le speedup mesuré est :

147,9 / 462,9 ≈ 0,32.

L'ajout d'un deuxième GPU ne produit donc pas d'accélération dans cette configuration : le débit total devient même inférieur à celui d'un seul GPU.

Le temps par pas passe également de 138,2 ms avec un L4 à 865,6 ms avec deux L4. Cela montre que le coût des communications et des synchronisations entre les processus devient très important.

**Effet du hook BF16 :**

Avec deux L4, l'utilisation du hook BF16 fait passer le débit de 147,9 à 276,4 images/s. Le temps par pas diminue de 865,6 à 463,2 ms.

Le hook améliore donc nettement les performances en réduisant le coût des échanges de gradients. Toutefois, le débit reste inférieur à celui obtenu avec un seul L4 (462,9 images/s).

**Effet de l'accumulation des gradients :**

Avec une accumulation de 8, le débit sur un L4 est de 486,6 images/s. Avec deux L4, l'accumulation de 8 et le hook BF16, il atteint 744,5 images/s.

Le speedup est alors :

744,5 / 486,6 ≈ 1,53.

Pour deux GPU, l'efficacité de mise à l'échelle est donc d'environ :

1,53 / 2 ≈ 76,5 %.

Cette configuration permet enfin de bénéficier de l'ajout d'un deuxième GPU. L'accumulation des gradients permet de travailler avec un batch effectif plus important : 512 avec un L4 et 1024 avec deux L4.

**Coût de communication :**

Le test NCCL a mesuré un all-reduce de 102 Mo en 1240,8 ms, soit environ 0,08 Go/s de débit algorithmique. La communication utilise Socket sur le réseau Ethernet TCP/IP, sans InfiniBand.

Cette bande passante limitée explique une partie importante des mauvaises performances du DDP standard. Les processus doivent échanger les gradients et se synchroniser à chaque étape de l'entraînement. Si un processus termine ses calculs avant les autres, il doit attendre les processus plus lents.

**Rôle du chargement des données :**

Les benchmarks DDP utilisent des données synthétiques. Le chargement des images n'est donc pas responsable du ralentissement observé dans ces mesures DDP.

Sur H100, la référence synthétique atteint 710,5 images/s, contre 696,2 images/s avec les données réelles, soit une différence d'environ 2 %. Avec trois workers et `pin_memory=True`, le chargement des données ne constitue donc pas un goulot d'étranglement majeur dans cette configuration de référence.

Les écarts de performances observés en DDP proviennent principalement du coût des communications et des synchronisations entre processus, plutôt que du chargement des données.

### 3. Recommandations pour l'entraînement et le déploiement

**Pour entraîner ResNet-50 sur ce cluster :**

Je choisirais un seul GPU NVIDIA H100 NVL avec la configuration suivante :

- Précision BF16 pour réduire la consommation mémoire et accélérer les calculs.
- Format mémoire `channels_last`, mieux adapté aux opérations convolutives.
- Chargement des données réelles avec `pin_memory=True` et 3 workers.
- Batch de 64 pour conserver une configuration comparable aux benchmarks.
- Sans compilation dans cette configuration, car les mesures réelles montrent ici un meilleur débit sans `torch.compile`.

Cette configuration correspond à `bf16-cl-raw-w3`, qui atteint 2878,8 images/s sur les données réelles, contre 696,2 images/s pour la référence FP32 avec données réelles.

Elle permet ainsi d'améliorer fortement le débit tout en réduisant le pic mémoire GPU de 5,95 Go à 3,29 Go.

Même si le DDP permet d'obtenir 744,5 images/s avec deux L4 dans la meilleure configuration testée, le H100 seul reste un meilleur choix pour ce cluster dans les conditions mesurées. Les communications entre les deux L4 limitent fortement le bénéfice du parallélisme distribué.

**Pour déployer le modèle sur CPU :**

Je choisirais la quantification statique INT8 avec 4 threads.

Les résultats obtenus sont :
- Taille du modèle : 23,9 Mo, contre 94,3 Mo en FP32.
- Latence : 7,69 ms par image, contre 21,28 ms en FP32 compilé.
- Top-1 : 99,4 %, identique aux variantes FP32 et BF16.
- Accord avec les prédictions FP32 : 100 % sur les 500 images évaluées.

INT8 réduit donc la taille du modèle d'environ 75 % et divise la latence par environ 2,77 par rapport au FP32 compilé, tout en conservant les mêmes classes prédites sur l'échantillon de validation.

**Conclusion générale :**

Ce TP montre que l'optimisation des performances dépend de plusieurs facteurs complémentaires : la précision numérique, le format mémoire, la compilation, le chargement des données et l'architecture matérielle.

Sur H100, BF16 associé à `channels_last` et à un chargement optimisé constitue la meilleure combinaison mesurée avec les données réelles. Le DDP n'est intéressant sur les deux L4 qu'avec des optimisations de communication et une accumulation des gradients suffisante pour amortir les coûts de synchronisation.

Enfin, pour l'inférence CPU, la quantification statique INT8 offre le meilleur compromis entre taille du modèle, latence et précision.
