#!/bin/bash

if [ $# -eq 0 ]
then
      echo $0:" normal/cuda/both"
      exit
fi


if [ $1 = cuda ]
then
    nvcc -g -G -O3 -Wno-deprecated-gpu-targets -o ../CannyCUDA/canny_edgeCUDA ../CannyCUDA/canny_edge.cu ../CannyCUDA/canny.cu ../CannyCUDA/hysteresis.cu ../CannyCUDA/pgm_io.cu -lm
    ../CannyCUDA/canny_edgeCUDA sunrise8k.pgm 10 0.9 0.95
elif [ $1 = normal ]
then
    gcc -ggdb3 -o ../Canny/canny_edge ../Canny/canny_edge.c ../Canny/hysteresis.c ../Canny/pgm_io.c -lm -O3
    ../Canny/canny_edge sunrise8k.pgm 10 0.9 0.95
elif [ $1 = both ]
then
	nvcc -g -G -O3 -Wno-deprecated-gpu-targets -o ../CannyCUDA/canny_edgeCUDA ../CannyCUDA/canny_edge.cu ../CannyCUDA/canny.cu ../CannyCUDA/hysteresis.cu ../CannyCUDA/pgm_io.cu -lm
    ../CannyCUDA/canny_edgeCUDA sunrise8k.pgm 10 0.9 0.95
    mv sunrise8k.pgm_s_10.00_l_0.90_h_0.95.pgm primer.pgm

    gcc -ggdb3 -o ../Canny/canny_edge ../Canny/canny_edge.c ../Canny/hysteresis.c ../Canny/pgm_io.c -lm -O3
    ../Canny/canny_edge sunrise8k.pgm 10 0.9 0.95
    mv sunrise8k.pgm_s_10.00_l_0.90_h_0.95.pgm segundo.pgm

    if cmp primer.pgm segundo.pgm
    then
    	echo Iguales!
    else
    	echo Distintos!
    fi
else
    echo $0:" normal/cuda/both"
fi
