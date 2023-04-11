# Download dataset and do preprocessing
bash download.sh

# Back translation and create dataset
bash back_translation.sh

# Train
if [ ! -d ./checkpoints/transformer ]; then
    python3 src/train.py
    python3 ./fairseq/scripts/average_checkpoints.py \
        --inputs checkpoints/transformer \
        --num-epoch-checkpoints 5 \
        --output checkpoints/transformer/avg_last_5_checkpoint.pt
fi

# Test
python3 src/test.py