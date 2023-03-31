if [ ! -d data ]; then
    mkdir data
    gdown --id '1tbGNwk1yGoCBdu4Gi_Cia7EJ9OhubYD9' --output food11.zip
    unzip -d data food11.zip
    rm food11.zip
fi