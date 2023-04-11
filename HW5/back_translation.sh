# Preprocess
python3 src/preprocess.py

# Binarize
data_dir='./DATA/rawdata/'
dataset_name='ted2020'
prefix=$data_dir/$dataset_name

bindir='./DATA/data-bin/'
binpath=$bindir/$dataset_name

if [ ! -d $binpath ]; then
    python3 -m fairseq_cli.preprocess \
            --source-lang zh \
            --target-lang en \
            --trainpref $prefix/train \
            --validpref $prefix/valid \
            --testpref $prefix/test \
            --destdir $binpath \
            --joined-dictionary \
            --workers 16
fi

mono_dataset_name='mono'
mono_prefix=$data_dir/$mono_dataset_name
mono_binpath=$bindir/$mono_dataset_name

if [ ! -d $mono_binpath ]; then
    python -m fairseq_cli.preprocess \
        --source-lang zh \
        --target-lang en \
        --trainpref $mono_prefix/mono.tok \
        --destdir $mono_binpath \
        --srcdict ./DATA/data-bin/ted2020/dict.en.txt \
        --tgtdict ./DATA/data-bin/ted2020/dict.en.txt \
        --workers 16
    cp ./DATA/data-bin/mono/train.zh-en.zh.bin ./DATA/data-bin/ted2020/mono.zh-en.zh.bin
    cp ./DATA/data-bin/mono/train.zh-en.zh.idx ./DATA/data-bin/ted2020/mono.zh-en.zh.idx
fi

# Back translation
if [ ! -d ./checkpoints/transformer_bt ]; then
    python3 src/back_translation_train.py
    python3 ./fairseq/scripts/average_checkpoints.py \
        --inputs checkpoints/transformer_bt \
        --num-epoch-checkpoints 5 \
        --output checkpoints/transformer_bt/avg_last_5_checkpoint.pt
    python3 ./src/back_translation_test.py
fi

# Create dataset
syn_dataset_name='synthetic'
syn_binpath=$bindir/$syn_dataset_name

if [ ! -d $syn_binpath ]; then
    python3 -m fairseq_cli.preprocess \
        --source-lang zh \
        --target-lang en \
        --trainpref $mono_prefix/mono.tok \
        --destdir $syn_binpath \
        --srcdict ./DATA/data-bin/ted2020/dict.en.txt \
        --tgtdict ./DATA/data-bin/ted2020/dict.en.txt \
        --workers 16
fi

if [ ! -d ./DATA/data-bin/ted2020_with_mono ]; then
    cp -r ./DATA/data-bin/ted2020/ ./DATA/data-bin/ted2020_with_mono/

    cp ./DATA/data-bin/synthetic/train.zh-en.zh.bin ./DATA/data-bin/ted2020_with_mono/train1.en-zh.zh.bin
    cp ./DATA/data-bin/synthetic/train.zh-en.zh.idx ./DATA/data-bin/ted2020_with_mono/train1.en-zh.zh.idx
    cp ./DATA/data-bin/synthetic/train.zh-en.en.bin ./DATA/data-bin/ted2020_with_mono/train1.en-zh.en.bin
    cp ./DATA/data-bin/synthetic/train.zh-en.en.idx ./DATA/data-bin/ted2020_with_mono/train1.en-zh.en.idx
fi