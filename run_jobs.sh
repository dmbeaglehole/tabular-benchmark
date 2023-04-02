sweeps=("xlk3w2n1" "k5kktfov" "04b3t6ub" "vo4ewo5p" "5p8ylqqk" "hmb78a28" "sh413uqn" "zk3pevbk" "4qxyyz8m" "nm5r3d41" "tfz7v9d5" "cha3130k" "hkjlhl0d" "j2687r1o")



for id in ${sweeps[@]}; do
    echo $id
    sbatch --job-name="$id" delta_setup \
    "wandb agent rfm/feature_kernels_2/$id"
done


#wandb agent rfm/feature_kernels_2/45uvsdjp
