#!/bin/zsh

source ~/envs/ditherMe/bin/activate
echo $1|entr sh -c 'clear;python $1'

