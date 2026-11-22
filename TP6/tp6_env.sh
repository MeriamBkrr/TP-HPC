
# TP6 : environnement commun

source ~/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning

export TP6_RAW=/mnt/ssd/imagenet/2012

TP6_SHARED=/mnt/hdd/homes/jromero/tp6/imagenet10_256
export TP6_SMALL=/tmp/$USER/tp6/imagenet10_256

# Copie locale des données sur le noeud
if [ "${TP6_NO_COPY:-0}" != "1" ] && [ ! -d "$TP6_SMALL" ]; then
    mkdir -p "$(dirname "$TP6_SMALL")"
    cp -r "$TP6_SHARED" "$TP6_SMALL.$SLURM_JOB_ID"
    mv -T "$TP6_SMALL.$SLURM_JOB_ID" "$TP6_SMALL" 2>/dev/null || rm -rf "$TP6_SMALL.$SLURM_JOB_ID"
fi

echo "Job $SLURM_JOB_ID sur $(hostname)"
echo "GPU : ${CUDA_VISIBLE_DEVICES:-aucun}"
echo "CPU : $(nproc)"
