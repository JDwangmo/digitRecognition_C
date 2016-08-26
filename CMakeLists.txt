cmake_minimum_required(VERSION 3.6)
project(digitRecognition_C)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#set(SOURCE_FILES "CNN/main.cpp")
aux_source_directory(CNN SOURCE_DIR)
add_executable(CNN ${SOURCE_DIR})
