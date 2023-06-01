python3 hw10.py

if [ -d dim_mifgsm ]; then
    cd dim_mifgsm
    tar zcvf ../dim_mifgsm.tgz *  > /dev/null
    cd ..
fi

python3 report.py