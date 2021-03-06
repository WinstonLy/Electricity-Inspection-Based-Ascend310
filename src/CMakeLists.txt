# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(acl_cam)

# Compile options
add_compile_options(-std=c++17 -g -O0)



add_definitions(-DENABLE_DVPP_INTERFACE)

#set(CMAKE_SKIP_RPATH TRUE)



set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../out)

if (EXISTS /usr/local/Ascend/acllib/)
        set(ACL_PATH "/usr/local/Ascend/acllib")
        message(STATUS "set ACL_PATH: /usr/local/Ascend/acllib")
else ()
        set(ACL_PATH "/home/HwHiAiUser/Ascend/acllib")
        message(STATUS "set ACL_PATH to default path: /home/HwHiAiUser/Ascend/acllib")    
endif()

# Header path
include_directories(
        /home/winston/ascend_ddk/arm/include/
	    #/home/HwHiAiUser/ascend_ddk/arm/include/
	    /usr/local/Ascend/include # for peripheral_api.h
	    ${ACL_PATH}/include # acl hearders
        /usr/include/
        ../inc/
        ${PYTHON_INCLUDE_DIRS}
)

# add host lib path
link_directories(

        /home/winston/ascend_ddk/arm/lib/
        #/home/HwHiAiUser/ascend_ddk/arm/lib/
	    /usr/lib64/
        /usr/local/Ascend/acllib/lib64/stub/
	    ${ACL_PATH}/lib64/
)

#add_executable(video_main
#        video_main.cpp
#        dvpp_vdec.cpp
#        dvpp_venc.cpp
#        dvpp_vpc.cpp
#        ffmpeg_io.cpp
#        sample_objection.cpp
#        model_infer.cpp
#        dvpp_jpege.cpp
#        yolov4_post.cpp
#        utils.cpp)
#
#target_link_libraries(video_main
#        ascendcl
#        acl_dvpp
#        runtime
#        /usr/lib64/libslog.so
#        /usr/lib64/libc_sec.so
#        /usr/lib64/libmedia_mini.so
#        pthread
#        opencv_core
#        opencv_imgproc
#        opencv_videoio
#        avformat
#        avutil
#        avcodec
#        protobuf 
#        presenteragent)
#        
#add_executable(image_main
#        image_main.cpp
#        dvpp_vpc.cpp
#        ffmpeg_io.cpp
#        sample_objection.cpp
#        model_infer.cpp
#        dvpp_jpege.cpp
#        yolov4_post.cpp
#        dvpp_jpegd.cpp
#        utils.cpp)
#
#target_link_libraries(image_main
#        ascendcl
#        acl_dvpp
#        runtime
#        /usr/lib64/libslog.so
#        /usr/lib64/libc_sec.so
#        /usr/lib64/libmedia_mini.so
#        pthread
#        opencv_core
#        opencv_imgproc
#        opencv_videoio
#        avformat
#        avutil
#        avcodec
#        protobuf 
#        presenteragent)
add_executable(multi_main
        multi_main.cpp
        dvpp_vdec.cpp
        dvpp_venc.cpp
        dvpp_vpc.cpp
        ffmpeg_io.cpp
        sample_objection.cpp
        model_infer.cpp
        dvpp_jpege.cpp
        yolov4_post.cpp
        utils.cpp)

target_link_libraries(multi_main
        ascendcl
        acl_dvpp
        runtime
        /usr/lib64/libslog.so
        /usr/lib64/libc_sec.so
        /usr/lib64/libmedia_mini.so
        pthread
        opencv_core
        opencv_imgproc
        opencv_videoio
        avformat
        avutil
        avcodec
        protobuf 
        presenteragent)