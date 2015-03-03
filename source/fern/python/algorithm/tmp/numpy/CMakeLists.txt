if(FERN_TEST)
    add_subdirectory(test)
endif()

set(LIB_SOURCES
    add
    sqrt
    util
)
add_library(fern_python_extension_algorithm_numpy ${LIB_SOURCES})
target_link_libraries(fern_python_extension_algorithm_numpy
    fern_python_extension_core
    fern_algorithm_core
    fern_feature_core
    stdc++
)
set(EXTENSION_SOURCES
    add_overloads
    algorithm
    python_extension
    sqrt_overloads
    wrapped_data_type
)
add_library(_fern_algorithm_numpy SHARED ${EXTENSION_SOURCES})
target_link_libraries(_fern_algorithm_numpy
    fern_python_extension_algorithm_numpy
    fern_python_extension_algorithm_core
    fern_python_extension_core
    stdc++
)
configure_python_extension(_fern_algorithm_numpy "_fern_algorithm_numpy")
set_target_properties(_fern_algorithm_numpy
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/fern/algorithm/numpy
)
configure_file(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/algorithm/numpy/__init__.py
    COPYONLY
)
