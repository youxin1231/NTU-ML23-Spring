# set up environment
pip install pytorchcv
pip install imgaug

# download
gdown --id 1t2UFQXr1cr5qLMBK2oN2rY1NDypi9Nyw --output data.zip

# if the above link isn't available, try this one
# !wget https://www.dropbox.com/s/lbpypqamqjpt2qz/data.zip

# unzip
unzip ./data.zip

rm ./data.zip