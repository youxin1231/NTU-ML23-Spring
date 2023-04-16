stylegan2_pytorch --data data \
    --name hw6 \
    --results_dir output \
    --models_dir ckpt \
    --batch-size 64 \
    --gradient-accumulate-every 4 \
    --network-capacity 20 \
    --image-size 64 \
    --num-train-steps 100000 \
    --multi-gpus \