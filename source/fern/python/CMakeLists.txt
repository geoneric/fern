add_subdirectory(algorithm)
add_subdirectory(core)
add_subdirectory(feature)
add_subdirectory(io)
if(FERN_TEST)
    add_subdirectory(test)
endif()


configure_file(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/__init__.py
    COPYONLY
)
add_custom_target(fern_python_package
    DEPENDS
        ${PROJECT_BINARY_DIR}/bin/fern/__init__.py
)
add_dependencies(fern_python_package
    fern_algorithm_python_package
    fern_feature_python_package
    fern_io_python_package
)
