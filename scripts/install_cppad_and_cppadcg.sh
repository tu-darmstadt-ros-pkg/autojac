#!/bin/bash
echo "Remove previous installs"
rm -r -f temp_cppad_install
rm -r -f lib
cd include
rm -r cppad
echo "Install CppAD"
cd ..
mkdir temp_cppad_install
mkdir lib
cd temp_cppad_install
# use latest release tested
git clone --branch 20240000.5 https://github.com/coin-or/CppAD.git cppad 
cd cppad
mkdir build
cd build
cmake -D cppad_prefix="" ..
make DESTDIR=install install
cd install/include
cp -r cppad ../../../../../include
cd ..
cp -r lib/. ../../../../lib/
cd ../../..
echo "Install CppADCodeGen"
git clone https://github.com/joaoleal/CppADCodeGen.git CppADCodeGen
cd CppADCodeGen
# use latest version tested
git checkout e15c57207ea42a9572e9ed44df1894e09f7ce67e
mkdir build
cd build
cmake -D CMAKE_INSTALL_PREFIX="" -D CPPAD_INCLUDE_DIR=`pwd`/../../cppad/build/install/include ..
make DESTDIR=install install
cd install/include/cppad
cp -r cg ../../../../../../include/cppad
cp cg.hpp ../../../../../../include/cppad
cd ../../../../../..
rm -r -f temp_cppad_install
echo "Installation finished"
