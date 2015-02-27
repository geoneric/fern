add_subdirectory(algebra)
add_subdirectory(core)
add_subdirectory(space)
if(FERN_TEST)
    add_subdirectory(test)
endif()


set(EXTENSION_SOURCES
    execution_policy
    python_extension
)
add_library(_fern_algorithm SHARED ${EXTENSION_SOURCES})
target_link_libraries(_fern_algorithm
    fern_algorithm_policy
    fern_core
    ${Boost_PYTHON_LIBRARY}
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
