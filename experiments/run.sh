#!/usr/bin/env bash

echo "Spawning 100 processes"
for i in {1..100} ;
do
    ( ./my_script.sh  & );
done
wait
echo "hello world"