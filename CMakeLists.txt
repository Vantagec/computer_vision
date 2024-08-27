#cmake_minimum_required(VERSION 3.5)

#project(MNIST13Otus_neyrosety LANGUAGES CXX)

#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

#add_executable(MNIST13Otus_neyrosety main.cpp readerMNIST.h readerMNIST.cpp container.h container.cpp classifierFashion.h classifierFashion.cpp)



cmake_minimum_required(VERSION 3.2)

project(MNIST13Otus_neyrosety VERSION 0.0 ${PATCH_VERSION})

configure_file(version.h.in version.h)

add_executable(MNIST13Otus_neyrosety main.cpp readerMNIST.h readerMNIST.cpp container.h container.cpp classifierFashion.h classifierFashion.cpp)

set_target_properties(MNIST13Otus_neyrosety PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        COMPILE_OPTIONS "-Wpedantic;-Wall;-Wextra"
        )

install(TARGETS MNIST13Otus_neyrosety RUNTIME DESTINATION bin)

set(CPACK_GENERATOR DEB)

set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")

set(CPACK_PACKAGE_CONTACT Alttaira@yandex.ru)

include(CPack)
