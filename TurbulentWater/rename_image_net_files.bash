#!/bin/bash

declare DATA_ROOT='data'
declare IMAGE_NET_ROOT="${DATA_ROOT}/ImageNet"
declare -a SETS=('test' 'train')
declare -a EXTENSIONS=('.jpg' '.jpeg' '.JPG')
declare -a GOOD_EXTENSION='.JPEG'

for set in "${SETS[@]}"
do
    echo ${set}

    for extension in "${EXTENSIONS[@]}"
    do
        echo ${extension}

        for file_path in "${IMAGE_NET_ROOT}/${set}"/*/*"${extension}"
        do
#            echo ${file_path}
#            echo ${#file_path}
#            echo ${#extension}
            prefix_length=$(( ${#file_path} - ${#extension} ))
#            echo ${prefix_length}
            prefix=${file_path:0:${prefix_length}}
#            echo ${prefix}echo ${prefix}
            new_path="${prefix}${GOOD_EXTENSION}"
            echo "${file_path} -> ${new_path}"
            mv "${file_path}" "${new_path}"
        done
    done
done
