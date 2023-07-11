cmake_minimum_required(VERSION 3.0.0)
project(yolo VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
aux_source_directory(src SOURCES)
add_executable(yolo ${SOURCES})
target_include_directories(yolo PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_directories(yolo PRIVATE ${PROJECT_SOURCE_DIR}/lib)
file(GLOB LIB_FILES ${PROJECT_SOURCE_DIR}/lib/*.lib)

foreach(item IN LISTS LIB_FILES)
    get_filename_component(_filename ${item} NAME_WE)
    list(APPEND LIB_NAMES ${_filename})
    message(STATUS "Load Dll Name: ${_filename}")
endforeach()

target_link_libraries(yolo PRIVATE ${LIB_NAMES})

file(GLOB DLL_FILES ${PROJECT_SOURCE_DIR}/lib/*.dll)
file(COPY ${DLL_FILES} DESTINATION ${PROJECT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
file(COPY model DESTINATION ${PROJECT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
file(COPY assets DESTINATION ${PROJECT_BINARY_DIR}/${CMAKE_BUILD_TYPE})

set(CMAKE_INSTALL_PREFIX output)

install(TARGETS yolo RUNTIME DESTINATION .)
install(DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/model DESTINATION .)
install(DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/assets DESTINATION .)
install(FILES ${DLL_FILES} DESTINATION .)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()