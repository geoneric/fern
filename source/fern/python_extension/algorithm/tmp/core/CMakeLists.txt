set(SOURCES
    add
    sqrt
    util
)
add_library(fern_python_extension_algorithm_core ${SOURCES})
target_link_libraries(fern_python_extension_algorithm_core
    ${PYTHON_LIBRARIES}
    stdc++
)
