sweeps=("ql723dgt" "radsnftn" "5ekzhoiv" "kcg4t57e" "xwd1c2md" "3q93al19" "y9zq87hy" "sgv2vrgg" "pkhrfd5v" "1kwneq6p" "fgppbfjz" "3eu5bld4")


for id in ${sweeps[@]}; do
    echo $id
    sbatch --job-name="$id" delta_setup \
    "wandb agent rfm/feature_kernels_2/$id"
done
