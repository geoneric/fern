add_test_conditionally(test)

set(EXTENSION_SOURCES
    gdal
    python_extension
)
add_library(_fern_io SHARED
    ${EXTENSION_SOURCES}
)
target_link_libraries(_fern_io
    fern_python_extension_feature
    fern_feature
    fern_io
    ${Boost_PYTHON_LIBRARY}
    ${PYTHON_LIBRARIES}
    stdc++
)
configure_python_extension(_fern_io "_fern_io")
set_target_properties(_fern_io
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/fern/io
)
configure_file(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/io/__init__.py
    COPYONLY
)
add_custom_target(fern_io_python_package
    DEPENDS
        ${PROJECT_BINARY_DIR}/bin/fern/io/__init__.py
)
add_dependencies(fern_io_python_package
    _fern_io
)
