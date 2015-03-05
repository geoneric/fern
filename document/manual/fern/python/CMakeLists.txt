# add_subdirectory(source)

file(GLOB_RECURSE FERN_PYTHON_SPHINX_SOURCES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    *.rst
    *.cc
)
