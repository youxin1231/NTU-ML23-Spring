if [ ! -d libriphone ]; then
    gdown --id '1qzCRnywKh30mTbWUEjXuNT2isOCAPdO1' --output libriphone.zip
    unzip -q libriphone.zip
    rm libriphone.zip
fi