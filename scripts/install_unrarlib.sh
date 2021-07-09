#!/bin/bash

mkdir -p unrar_tmp
cd unrar_tmp
wget https://www.rarlab.com/rar/unrarsrc-6.0.7.tar.gz
tar zxvf unrarsrc-6.0.7.tar.gz
cd unrar
make lib
mkdir -p ../../lib
cp libunrar.so ../../lib
cd ../..
rm -rf unrar_tmp