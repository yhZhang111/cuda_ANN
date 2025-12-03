WORK_DIR:=./gpu_ann_ivf
TARGET_NAME:=brute_nn

all: build
	cd ${WORK_DIR} && ./build/${TARGET_NAME}

build: gen-cmake
	cd ${WORK_DIR}/build && make

gen-cmake:
	cd ${WORK_DIR} && mkdir -p build
	cd ${WORK_DIR}/build && cmake ../
