add_subdirectory(algorithm)
add_subdirectory(io)

file(GLOB_RECURSE SPHINX_SOURCES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    *.rst
    *.cc
)
set(SPHINX_TARGET
    fern_manual
)
include(SphinxDoc)
