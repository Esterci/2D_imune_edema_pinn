#!/bin/bash

if [ -d build ]; then
	rm -rf build
	mkdir build
else
	mkdir build
fi

if [ -d cb_over_time ]; then
	rm -rf cb_over_time
	mkdir cb_over_time
else
	mkdir cb_over_time
fi

if [ -d cb_over_time ]; then
	rm -rf cn_over_time
	mkdir cn_over_time
else
	mkdir cn_over_time
fi

cd build

cmake ../.

make clean all

./mmfe

