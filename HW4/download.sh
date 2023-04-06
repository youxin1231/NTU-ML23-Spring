if [ ! -d Dataset ]; then
    wget https://github.com/googly-mingto/ML2023HW4/releases/download/data/Dataset.tar.gz.partaa
    wget https://github.com/googly-mingto/ML2023HW4/releases/download/data/Dataset.tar.gz.partab
    wget https://github.com/googly-mingto/ML2023HW4/releases/download/data/Dataset.tar.gz.partac
    wget https://github.com/googly-mingto/ML2023HW4/releases/download/data/Dataset.tar.gz.partad

    cat Dataset.tar.gz.part* > Dataset.tar.gz
    rm Dataset.tar.gz.partaa
    rm Dataset.tar.gz.partab
    rm Dataset.tar.gz.partac
    rm Dataset.tar.gz.partad
    # unzip the file
    tar zxf Dataset.tar.gz
    rm Dataset.tar.gz
    rm ._Dataset
fi