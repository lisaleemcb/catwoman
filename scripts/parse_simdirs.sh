#!/bin/bash

path=$(pwd)
ref_file=/Users/emcbride/catwoman/refs/LoReLi_params_ref.txt
echo $path

for f in $path/*; do
    cd $f
    file=runtime*
    bash /Users/emcbride/catwoman/scripts/convert_pnames.sh $file $ref_file
    cd $path
done
