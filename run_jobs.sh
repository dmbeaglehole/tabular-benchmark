sweeps=("mkax5h7o" "hejdvh6k" "2ryx2eru" "r704r2jk" "wqa79xhx" "tdx90559" "4ytlv3lt" "zqttrpxj" "rc0seoio" "ammfejsi" "buypcdps" "6zzoyeef" "fsi59p1j" "uugpgw1i" "u10r65am" "ozc9wvah" "dv3lp2oa" "mwrf1tx0" "tal7tekt" "53pqwa2r") 

for id in ${sweeps[@]}; do
    echo $id
    sbatch --job-name="$id" expanse_setup \
    "wandb agent rfm/feature_kernels_2/$id"
done
