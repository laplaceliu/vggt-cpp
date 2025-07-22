#!/bin/bash

THIRDPARTY_SOURCE_DIR=$(pwd)/thirdparty
DEPENDENCIES_INSTALL_DIR=$(pwd)/dependencies
export CMAKE_LIBRARY_PATH=$DEPENDENCIES_INSTALL_DIR/lib:$CMAKE_LIBRARY_PATH
export CMAKE_INCLUDE_PATH=$DEPENDENCIES_INSTALL_DIR/include:$CMAKE_INCLUDE_PATH

build_argparse() {
    TMP_DIR=$(mktemp -d)

    NAME=argparse-3.2
    SOURCE=$NAME.tar.gz
    tar zxf $THIRDPARTY_SOURCE_DIR/$SOURCE -C $TMP_DIR

    pushd $TMP_DIR/$NAME
    mkdir build && cd build
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=$DEPENDENCIES_INSTALL_DIR \
        -DCMAKE_BUILD_TYPE=MinSizeRel \
        -DARGPARSE_BUILD_SAMPLES=off \
        -DARGPARSE_BUILD_TESTS=off

    make -j
    make install
    popd

    rm -rf $TMP_DIR
}
build_cgal() {
    TMP_DIR=$(mktemp -d)

    NAME=cgal-6.0.1
    SOURCE=$NAME.tar.gz
    tar zxf $THIRDPARTY_SOURCE_DIR/$SOURCE -C $TMP_DIR

    pushd $TMP_DIR/$NAME
    mkdir build && cd build
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=$DEPENDENCIES_INSTALL_DIR \
        -DCMAKE_BUILD_TYPE=MinSizeRel

    make -j
    make install
    popd

    rm -rf $TMP_DIR
}

build_freeimage() {
    TMP_DIR=$(mktemp -d)

    NAME=FreeImage-3.19.10
    SOURCE=$NAME.tar.gz
    tar zxf $THIRDPARTY_SOURCE_DIR/$SOURCE -C $TMP_DIR

    pushd $TMP_DIR/$NAME
    mkdir build && cd build
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=$DEPENDENCIES_INSTALL_DIR \
        -DCMAKE_BUILD_TYPE=MinSizeRel

    make -j
    make install
    popd

    rm -rf $TMP_DIR
}

build_metis() {
    TMP_DIR=$(mktemp -d)
    pushd $TMP_DIR

    NAME=GKlib
    git clone https://github.com/KarypisLab/$NAME.git
    pushd $TMP_DIR/$NAME
    make config prefix=$DEPENDENCIES_INSTALL_DIR
    make -j
    make install
    popd

    NAME=METIS
    git clone https://github.com/KarypisLab/$NAME.git

    pushd $TMP_DIR/$NAME
    make config \
    prefix=$DEPENDENCIES_INSTALL_DIR \
    gklib_path=$DEPENDENCIES_INSTALL_DIR
    make -j
    make install
    popd

    popd

    rm -rf $TMP_DIR
}

build_openblas() {
    TMP_DIR=$(mktemp -d)

    NAME=OpenBLAS-0.3.30
    SOURCE=$NAME.tar.gz
    tar zxf $THIRDPARTY_SOURCE_DIR/$SOURCE -C $TMP_DIR

    pushd $TMP_DIR/$NAME
    mkdir build && cd build
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=$DEPENDENCIES_INSTALL_DIR \
        -DCMAKE_BUILD_TYPE=MinSizeRel

    make -j
    make install
    popd

    rm -rf $TMP_DIR
}

build_boost() {
    TMP_DIR=$(mktemp -d)

    NAME=boost_1_88_0
    SOURCE=$NAME.tar.gz
    tar zxf $THIRDPARTY_SOURCE_DIR/$SOURCE -C $TMP_DIR

    pushd $TMP_DIR/$NAME
    ./bootstrap.sh
    ./b2 install --prefix=$DEPENDENCIES_INSTALL_DIR

    popd

    rm -rf $TMP_DIR
}

build_ceres() {
    TMP_DIR=$(mktemp -d)

    NAME=ceres-solver-2.2.0
    SOURCE=$NAME.tar.gz
    tar zxf $THIRDPARTY_SOURCE_DIR/$SOURCE -C $TMP_DIR

    pushd $TMP_DIR/$NAME
    mkdir build && cd build
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=$DEPENDENCIES_INSTALL_DIR \
        -DCMAKE_BUILD_TYPE=MinSizeRel \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARK=OFF \
        -DEIGENMETIS=OFF \
        -DUSE_CUDA=ON

    make -j
    make install
    popd

    rm -rf $TMP_DIR
}

build_mpfr() {
    TMP_DIR=$(mktemp -d)

    NAME=mpfr
    pushd $TMP_DIR
    git clone https://github.com/aixoss/$NAME.git

    pushd $TMP_DIR/$NAME
    ./configure --prefix=$DEPENDENCIES_INSTALL_DIR
    make -j
    make install
    popd
    popd

    rm -rf $TMP_DIR
}

build_poselib() {
    TMP_DIR=$(mktemp -d)

    NAME=PoseLib-9c8f3ca
    SOURCE=$NAME.tar.gz
    tar zxf $THIRDPARTY_SOURCE_DIR/$SOURCE -C $TMP_DIR

    pushd $TMP_DIR/$NAME
    mkdir build && cd build

    cmake .. \
        -DCMAKE_INSTALL_PREFIX=$DEPENDENCIES_INSTALL_DIR \
        -DCMAKE_BUILD_TYPE=MinSizeRel \
        -DWITH_BENCHMARK=OFF \
        -DPYTHON_PACKAGE=OFF

    make -j8
    make install
    popd

    rm -rf $TMP_DIR
}

build_flann() {
    # build_cgal
    # build_ceres
    # build_mpfr
    # build_freeimage
    # build_metis
    TMP_DIR=$(mktemp -d)

    NAME=flann-1.9.2
    SOURCE=$NAME.tar.gz
    tar zxf $THIRDPARTY_SOURCE_DIR/$SOURCE -C $TMP_DIR

    pushd $TMP_DIR/$NAME
    mkdir build && cd build
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=$DEPENDENCIES_INSTALL_DIR \
        -DCMAKE_BUILD_TYPE=MinSizeRel \
        -DCMAKE_CUDA_ARCHITECTURES=native \
        -DBUILD_MATLAB_BINDINGS=OFF \
        -DBUILD_PYTHON_BINDINGS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_DOC=OFF

    make -j
    make install
    popd

    rm -rf $TMP_DIR
}

build_colmap() {
    # build_cgal
    # build_ceres
    # build_mpfr
    # build_freeimage
    # build_metis
    # build_flann
    TMP_DIR=$(mktemp -d)

    NAME=colmap-3.10
    SOURCE=$NAME.tar.gz
    tar zxf $THIRDPARTY_SOURCE_DIR/$SOURCE -C $TMP_DIR

    pushd $TMP_DIR/$NAME
    mkdir build && cd build
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=$DEPENDENCIES_INSTALL_DIR \
        -DCMAKE_BUILD_TYPE=MinSizeRel \
        -DCMAKE_CUDA_ARCHITECTURES=native \
        -DBLA_VENDOR=Intel10_64lp

    make -j
    make install
    popd

    rm -rf $TMP_DIR
}

build_eigen() {
    TMP_DIR=$(mktemp -d)

    NAME=eigen-3.3.9
    SOURCE=$NAME.tar.gz
    tar zxf $THIRDPARTY_SOURCE_DIR/$SOURCE -C $TMP_DIR

    pushd $TMP_DIR/$NAME
    mkdir build && cd build
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=$DEPENDENCIES_INSTALL_DIR \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DEIGEN_BUILD_DOC=OFF
    make -j
    make install
    popd

    rm -rf $TMP_DIR
}

build_eigen_master() {
    TMP_DIR=$(mktemp -d)

    NAME=eigen
    pushd $TMP_DIR
    git clone https://gitlab.com/libeigen/$NAME.git

    pushd $TMP_DIR/$NAME
    mkdir build && cd build
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=$DEPENDENCIES_INSTALL_DIR \
        -DCMAKE_BUILD_TYPE=MinSizeRel \
        -DBUILD_TESTING=OFF \
        -DEIGEN_BUILD_DOC=OFF
    make -j
    make install
    popd
    popd

    rm -rf $TMP_DIR
}

build_gflags() {
    TMP_DIR=$(mktemp -d)

    NAME=gflags-2.2.2
    SOURCE=$NAME.tar.gz
    tar zxf $THIRDPARTY_SOURCE_DIR/$SOURCE -C $TMP_DIR

    pushd $TMP_DIR/$NAME
    mkdir build && cd build

    cmake .. \
        -DCMAKE_INSTALL_PREFIX=$DEPENDENCIES_INSTALL_DIR \
        -DCMAKE_BUILD_TYPE=MinSizeRel \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_TESTING=OFF


    make -j
    make install
    popd

    rm -rf $TMP_DIR
}

build_gtest() {
    TMP_DIR=$(mktemp -d)

    NAME=googletest-1.17.0
    SOURCE=$NAME.tar.gz
    tar zxf $THIRDPARTY_SOURCE_DIR/$SOURCE -C $TMP_DIR

    pushd $TMP_DIR/$NAME
    mkdir build && cd build

    cmake .. \
        -DCMAKE_INSTALL_PREFIX=$DEPENDENCIES_INSTALL_DIR \
        -DCMAKE_BUILD_TYPE=MinSizeRel

    make -j
    make install
    popd

    rm -rf $TMP_DIR
}

build_glog() {
    TMP_DIR=$(mktemp -d)

    NAME=glog-0.7.1
    SOURCE=$NAME.tar.gz
    tar zxf $THIRDPARTY_SOURCE_DIR/$SOURCE -C $TMP_DIR

    pushd $TMP_DIR/$NAME
    mkdir build && cd build
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=$DEPENDENCIES_INSTALL_DIR \
        -DCMAKE_BUILD_TYPE=MinSizeRel \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_SHARED_LIBS=OFF


    make -j
    make install
    popd

    rm -rf $TMP_DIR
}

build_opencv() {
    TMP_DIR=$(mktemp -d)

    CONTRIB_NAME=opencv_contrib-4.12.0
    CONTRIB_SOURCE=$CONTRIB_NAME.tar.gz
    tar zxf $THIRDPARTY_SOURCE_DIR/$CONTRIB_SOURCE -C $TMP_DIR

    OPENCV_NAME=opencv-4.12.0
    OPENCV_SOURCE=$OPENCV_NAME.tar.gz
    tar zxf $THIRDPARTY_SOURCE_DIR/$OPENCV_SOURCE -C $TMP_DIR

    pushd $TMP_DIR/opencv-4.12.0
    mkdir build && cd build
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=$DEPENDENCIES_INSTALL_DIR \
        -DCMAKE_BUILD_TYPE=MinSizeRel \
        -DOPENCV_ENABLE_NONFREE=ON \
        -DOPENCV_DNN_CUDA=ON \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=ON \
        -DWITH_CUBLAS=ON \
        -DWITH_NVCUVID=ON \
        -DWITH_NVCUVENC=ON \
        -DWITH_EIGEN=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_DOCS=OFF \
        -DBUILD_opencv_alphamat=OFF \
        -DBUILD_opencv_aruco=ON \
        -DBUILD_opencv_bgsegm=OFF \
        -DBUILD_opencv_bioinspired=OFF \
        -DBUILD_opencv_ccalib=OFF \
        -DBUILD_opencv_cnn_3dobj=OFF \
        -DBUILD_opencv_cvv=ON \
        -DBUILD_opencv_datasets=OFF \
        -DBUILD_opencv_dnn_objdetect=OFF \
        -DBUILD_opencv_dnn_superres=OFF \
        -DBUILD_opencv_dnns_easily_fooled=OFF \
        -DBUILD_opencv_dpm=OFF \
        -DBUILD_opencv_face=OFF \
        -DBUILD_opencv_freetype=OFF \
        -DBUILD_opencv_fuzzy=OFF \
        -DBUILD_opencv_hdf=OFF \
        -DBUILD_opencv_hfs=OFF \
        -DBUILD_opencv_img_hash=OFF \
        -DBUILD_opencv_intensity_transform=OFF \
        -DBUILD_opencv_julia=OFF \
        -DBUILD_opencv_line_descriptor=ON \
        -DBUILD_opencv_matlab=OFF \
        -DBUILD_opencv_mcc=OFF \
        -DBUILD_opencv_optflow=ON \
        -DBUILD_opencv_ovis=OFF \
        -DBUILD_opencv_phase_unwrapping=OFF \
        -DBUILD_opencv_plot=OFF \
        -DBUILD_opencv_quality=OFF \
        -DBUILD_opencv_rapid=OFF \
        -DBUILD_opencv_reg=OFF \
        -DBUILD_opencv_rgbd=OFF \
        -DBUILD_opencv_saliency=OFF \
        -DBUILD_opencv_sfm=OFF \
        -DBUILD_opencv_shape=OFF \
        -DBUILD_opencv_stereo=OFF \
        -DBUILD_opencv_structured_light=OFF \
        -DBUILD_opencv_superres=OFF \
        -DBUILD_opencv_surface_matching=OFF \
        -DBUILD_opencv_text=OFF \
        -DBUILD_opencv_tracking=OFF \
        -DBUILD_opencv_videostab=ON \
        -DBUILD_opencv_viz=OFF \
        -DBUILD_opencv_wechat_qrcode=OFF \
        -DBUILD_opencv_xfeatures2d=ON \
        -DBUILD_opencv_ximgproc=OFF \
        -DBUILD_opencv_xobjdetect=OFF \
        -DBUILD_opencv_xphoto=OFF \
        -DOPENCV_EXTRA_MODULES_PATH=$TMP_DIR/$CONTRIB_NAME/modules
    make -j
    make install
    popd

    rm -rf $TMP_DIR
}

build_spdlog() {
    NAME=spdlog-1.12.0
    SOURCE=$NAME.tar.gz
    TMP_DIR=$(mktemp -d)

    tar zxf $THIRDPARTY_SOURCE_DIR/$SOURCE -C $TMP_DIR

    pushd $TMP_DIR/$NAME
    mkdir build && cd build

    cmake .. \
        -DCMAKE_INSTALL_PREFIX=$DEPENDENCIES_INSTALL_DIR \
        -DCMAKE_BUILD_TYPE=MinSizeRel \
        -DSPDLOG_BUILD_EXAMPLE=OFF \
        -DSPDLOG_BUILD_TESTS=OFF \
        -DSPDLOG_BUILD_BENCH=OFF

    make -j
    make install
    popd

    rm -rf $TMP_DIR
}

default() {
    build_eigen_master
    build_colmap
    build_gtest
    build_eigen
    build_opencv
    build_spdlog
}

if [ $# -eq 0 ]; then
    default
else
    for param in "$@"; do
        if [ $param = "argparse" ]; then
            build_argparse
        elif [ $param = "eigen" ]; then
            build_eigen
        elif [ $param = "eigen_master" ]; then
            build_eigen_master
        elif [ $param = "boost" ]; then
            build_boost
        elif [ $param = "mpfr" ]; then
            build_mpfr
        elif [ $param = "cgal" ]; then
            build_cgal
        elif [ $param = "flann" ]; then
            build_flann
        elif [ $param = "metis" ]; then
            build_metis
        elif [ $param = "poselib" ]; then
            build_poselib
        elif [ $param = "openblas" ]; then
            build_openblas
        elif [ $param = "freeimage" ]; then
            build_freeimage
        elif [ $param = "ceres" ]; then
            build_ceres
        elif [ $param = "colmap" ]; then
            build_colmap
        elif [ $param = "gflags" ]; then
            build_gflags
        elif [ $param = "glog" ]; then
            build_glog
        elif [ $param = "gtest" ]; then
            build_gtest
        elif [ $param = "opencv" ]; then
            build_opencv
        elif [ $param = "spdlog" ]; then
            build_spdlog
        else
            echo "unknown param: $param"
        fi
    done
fi
