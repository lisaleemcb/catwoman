#!/bin/bash

path=$(pwd)
ref_file=/obs/emcbride/catwoman/refs/LoReLi_params_ref.txt

echo $path

for f in $path/*; do
    cd $f
    echo "$(pwd)"
    file=runtime*
    bash /obs/emcbride/catwoman/scripts/convert_pnames.sh $file $ref_file
    cd $path
done
