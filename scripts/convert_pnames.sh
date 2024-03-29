#!/bin/bash
#================================================================
# HEADER
#================================================================
# <purpose of this script>
# Copyright (C) <current date> Lisa McBride (github: lisaleemcb)
# Permission to copy and modify is granted under the MIT license
# Last revised <data revised>

# Options:
#   -h, --help    Display this message.
#   -n            Dry-run; only show what would be done.
#

file=$1
ref=$2

#source /Users/emcbride/.bash_profile
#mamba activate kSZ

#echo which mamba

new_file="$1"_reformatted.txt

echo "Acting on file $file"

# this seems awful but for now, ça marche
nrows_file=$(egrep -cv '^#|^$' "$file")
nrows_ref=$(egrep -cv '^#|^$' "$ref")
if [ $nrows_file -ne $nrows_ref ]; then
    echo "The number of parameters between the file $file and the reference $ref do not match. Are you sure you are processing the correct pair?"
    echo "Your file has $nrows_file and your reference has $nrows_ref."
    exit 1
fi

N_file=$(sed -n '$=' "$file")
N_ref=$(sed -n '$=' "$ref")
start_file=$(($N_file - $nrows_file))
start_ref=$(($N_ref - $nrows_ref))

echo "hi, testing!"

for ((i = 0 ; i < (("$nrows_file"+1)) ; i++))
do
    #echo "processing the "$i"th line"
    pname_new=$(awk -v var=$(($i+$start_ref)) 'NR==var {print $1}' $ref)
    # echo "$pname_new"
    if [[ $pname_new == 'box_size' ]]
        then
           sed -nEe "s/,/./g" -e "s/\s+/ /g" -e ""$(($i+$start_file))" s/\/.*/\t"$pname_new"/gp" $file
    elif [[ $pname_new == 'elasticity_params' ]]
        then
        sed -nEe "s/\s+/ /g" -e "$(($(($i+$start_file)))) s/([0-9.]+) ([0-9.]+)/(\1, \2)/" -e "$(($(($i+$start_file)))) s/\/.*/\t$pname_new/gp" "$file"
    else
        sed -nEe "s/\s+/ /g" -e ""$(($i+$start_file))" s/\/.*/\t"$pname_new"/gp" $file
    fi
done > "$new_file"


echo "file $new_file saved at $(pwd)"
