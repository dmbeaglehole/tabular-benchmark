sweeps=("5mkato5z" "jalfc0wj" "f95gw8f9" "ok1z50r0" "0wb81axn" "8631xj0j" "8la3gqr3" "lfkh5k8e" "gnrdr2cs" "lrwgcqt2" "zbwr57si" "72omn5yr" "8jbbf39z")


for id in ${sweeps[@]}; do
    echo $id
    sbatch --job-name="$id" delta_setup \
    "wandb agent rfm/feature_kernels_2/$id"
done
