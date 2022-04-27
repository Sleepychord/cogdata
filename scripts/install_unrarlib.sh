#!/bin/bash

mkdir -p unrar_tmp
cd unrar_tmp
wget https://www.rarlab.com/rar/unrarsrc-6.0.7.tar.gz
tar zxvf unrarsrc-6.0.7.tar.gz
cd unrar
make lib
make install-lib
if [ $? -eq 0 ]; then
    echo "successfully install unrar lib."
else
    echo "Permission denied. Please use sudo to run this script, or manually move ./lib/libunrar.so to /usr/lib."
    mkdir -p ../../lib
    cp libunrar.so ../../lib
fi
cd ../..
rm -rf unrar_tmp