project( darkchannel )
cmake_minimum_required( VERSION 2.8 )
add_compile_options(-std=c++11 -g)

find_package( OpenCV REQUIRED )                       # 不使用 find_package

message( STATUS "OpenCV library status:" )              # 输出一下得到的变量
message( STATUS "    version: ${OpenCV_VERSION}" )
message( STATUS "    libraries: ${OpenCV_LIBS}" )
message( STATUS "    include path: ${OpenCV_INCLUDE_DIRS}" )

include_directories( ${OpenCV_INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/../include/ )           # 引入 OpenCV 头文件目录
aux_source_directory(${PROJECT_BINARY_DIR}/../src/ SRCS)


add_executable( ${PROJECT_NAME} ${SRCS} )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ) # exe 链接 OpenCV