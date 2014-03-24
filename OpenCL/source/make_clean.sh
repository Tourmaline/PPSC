#!/bin/bash

LIST=(VectorUpdate MatVecMul MatMatMul SparseMatMatMul DotProduct DotProduct_vec VectorUpdate_vec)

for item in ${LIST[*]}
do
    printf "make %s\n" $item
    cd $item
    make clean
    cd ..
echo '-----------------'
echo '-----------------'

done

