SET(SOURCES
    add
    sqrt
    util
)
ADD_LIBRARY(fern_python_extension_algorithm_core ${SOURCES})
TARGET_LINK_LIBRARIES(fern_python_extension_algorithm_core
    ${PYTHON_LIBRARIES}
    stdc++
)
