#!bin/bash

path=$(pwd)
ref_file=/obs/emcbride/catwoman/refs/LoReLi_params_ref.txt

echo $path

for f in $path/*; do
    bash /obs/emcbride/catwoman/scripts/convert_pnames.sh $f $ref_file
    echo "copying $f"
done

