add_subdirectory(image)
add_subdirectory(source)
file(GLOB_RECURSE sources
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    *.rst
    *.cc
    *.h
    *.png
)
create_sphinx_docs(
    TARGET post_generic_algorithms
    SOURCES ${sources})
add_dependencies(post_generic_algorithms post_generic_algorithms_images)
