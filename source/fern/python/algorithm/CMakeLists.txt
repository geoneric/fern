add_subdirectory(algebra)
add_subdirectory(core)
add_subdirectory(space)
add_test_conditionally(test)


set(EXTENSION_SOURCES
    execution_policy
    python_extension
)
add_library(_fern_algorithm SHARED
    ${EXTENSION_SOURCES}
)
target_link_libraries(_fern_algorithm
    fern_algorithm
    ${Boost_PYTHON_LIBRARY}
    ${PYTHON_LIBRARIES}
    stdc++
)
configure_python_extension(_fern_algorithm "_fern_algorithm")
set_target_properties(_fern_algorithm
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/fern/algorithm
)
configure_file(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/algorithm/__init__.py
    COPYONLY
)
add_custom_target(fern_algorithm_python_package
    DEPENDS
        ${PROJECT_BINARY_DIR}/bin/fern/algorithm/__init__.py
)
add_dependencies(fern_algorithm_python_package
    fern_algorithm_algebra_python_package
    fern_algorithm_core_python_package
    fern_algorithm_space_python_package
    _fern_algorithm
)
