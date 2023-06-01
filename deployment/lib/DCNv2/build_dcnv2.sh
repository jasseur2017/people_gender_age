#!/bin/bash

if [ ! -f build/dcn_plugin.so ]; then
    echo "compiling dcn_plugin.so"
    make clean
    make
else
    echo "dcn_plugin.so already exists"
fi
