#!/bin/bash
rm goselector.cpython-39-x86_64-linux-gnu*
rm -rf build
set -e 
CURRENT_DIR=$(cd "$(dirname "$0")";pwd)
echo $CURRENT_DIR
cd $CURRENT_DIR/go_selector \
    && go build -ldflags '-w -s' -buildmode=c-archive -o $CURRENT_DIR/bind11/libgo.a . 
cd $CURRENT_DIR \
    && python $CURRENT_DIR/bind11/setup.py build_ext --inplace 