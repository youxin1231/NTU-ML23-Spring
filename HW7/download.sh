# Download Dataset
if [ ! -d data ]; then
    gdown --id '1TjoBdNlGBhP_J9C66MOY7ILIrydm7ZCS' --output hw7_data.zip
    unzip -o hw7_data.zip -d data
    rm hw7_data.zip
fi