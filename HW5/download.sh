# Package
if [ ! -d fairseq ]; then
    git clone https://github.com/pytorch/fairseq.git
    cd fairseq && git checkout 9a1c497
    pip install --upgrade fairseq && cd ..
fi

# Dataset
data_dir='./DATA/rawdata/'
dataset_name='ted2020'
prefix=$data_dir/$dataset_name

mono_dataset_name='mono'
mono_prefix=$data_dir/$mono_dataset_name

if [ ! -d DATA/rawdata ]; then
    mkdir -p $prefix $mono_prefix
    wget https://github.com/figisiwirf/ml2023-hw5-dataset/releases/download/v1.0.1/ml2023.hw5.data.tgz -O ted2020.tgz
    wget https://github.com/figisiwirf/ml2023-hw5-dataset/releases/download/v1.0.1/ml2023.hw5.test.tgz -O test.tgz
    tar -xvf ted2020.tgz -C $prefix && rm ted2020.tgz
    tar -xvf test.tgz -C $prefix && rm test.tgz
    mv $prefix/'raw.en' $prefix/'train_dev.raw.en'
    mv $prefix/'raw.zh' $prefix/'train_dev.raw.zh'
    mv $prefix/'test.en' $prefix/'test.raw.en'
    mv $prefix/'test.zh' $prefix/'test.raw.zh'

    wget https://github.com/figisiwirf/ml2023-hw5-dataset/releases/download/v1.0.1/ted_zh_corpus.deduped.gz -O ted_zh_corpus.deduped.gz
    gzip -fkd ted_zh_corpus.deduped.gz && rm ted_zh_corpus.deduped.gz
    mv ted_zh_corpus.deduped $mono_prefix/train.raw.zh
fi
