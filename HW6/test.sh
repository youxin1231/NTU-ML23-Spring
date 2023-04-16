rm -rf results && rm -r *.tgz

stylegan2_pytorch \
    --generate \
    --name hw6 \
    --models_dir ckpt \
    --num_generate 1000 \
    --num_image_tiles 1 \

mkdir -p results/ema results/mr
mv results/hw6/*mr* results/mr
mv results/hw6/*ema* results/ema

# Rename outputs
num=1
for file in $(ls -v results/hw6/*.jpg); do
   mv $file results/hw6/$num.jpg
   let num=$num+1
done

cd results/hw6
tar -zcvf ../../hw6.tgz *.jpg > /dev/null
cd ../..

num=1
for file in $(ls -v results/mr/*.jpg); do
   mv $file results/mr/$num.jpg
   let num=$num+1
done

cd results/mr
tar -zcvf ../../mr.tgz *.jpg > /dev/null
cd ../..

num=1
for file in $(ls -v results/ema/*.jpg); do
   mv $file results/ema/$num.jpg
   let num=$num+1
done

cd results/ema
tar -zcvf ../../images.tgz *.jpg > /dev/null
cd ../..