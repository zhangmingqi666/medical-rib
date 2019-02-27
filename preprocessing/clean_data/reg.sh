#!/usr/bin/env bash

if [[ $1 =~ ^[a-zA-Z1-9].*nii$ ]]
then
    echo "$1 match"
else
    echo "$1 not match"
fi