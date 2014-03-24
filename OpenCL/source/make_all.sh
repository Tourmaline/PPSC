#!/bin/bash

LIST=(VectorUpdate MatVecMul MatMatMul SparseMatMatMul DotProduct VectorUpdate_vec DotProduct_vec)

for item in ${LIST[*]}
do
    printf "make %s\n" $item
    cd $item
    make
    cd ..
echo '-----------------'
echo '-----------------'

done

