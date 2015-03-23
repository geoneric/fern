add_subdirectory(image)
add_subdirectory(source)
file(GLOB_RECURSE SPHINX_SOURCES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    *.rst
    *.cc
    *.h
    *.png
)
set(SPHINX_TARGET
    generic_algorithms
)
include(SphinxDoc)
