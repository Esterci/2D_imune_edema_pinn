# cmake_minimum_required(VERSION 3.19.4)

# project(Modelagem-Mecanica-na-Fomacao-de-Edemas-Joao-Victor)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lm -O3 -fopenmp -march=native -mtune=native -funroll-loops")

# file(GLOB source_files
#         "*.h"
#         "*.c"
#         "source/*.c"
#         "include/*.h"
# )

# add_executable(mmfe ${source_files})

# target_link_libraries(mmfe m)
# include_directories(./include)
cmake_minimum_required(VERSION 3.19.4)

project(Modelagem-Mecanica-na-Fomacao-de-Edemas-Joao-Victor)

# Find OpenMP package
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lm -O3 -fopenmp -march=native -mtune=native -funroll-loops")

file(GLOB source_files
        "*.h"
        "*.c"
        "source/*.c"
        "include/*.h"
)

add_executable(mmfe ${source_files})

# Link OpenMP
target_link_libraries(mmfe PRIVATE OpenMP::OpenMP_CXX m)

include_directories(./include)

