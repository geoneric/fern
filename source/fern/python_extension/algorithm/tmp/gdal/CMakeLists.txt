add_subdirectory(test)
set(SOURCES
    add
    add_overloads
    algorithm
    python_extension
    slope
    slope_overloads
    util
    wrapped_data_type
)
add_library(fern_algorithm_gdal SHARED ${SOURCES})
target_link_libraries(fern_algorithm_gdal
    _fern_algorithm_numpy
    fern_python_extension_algorithm_core
    fern_python_extension_core
    fern_algorithm_core
    fern_feature_core
    ${GDAL_LIBRARY}
    stdc++
)
configure_python_extension(fern_algorithm_gdal "_fern_algorithm_gdal")
set_target_properties(fern_algorithm_gdal
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin/fern/algorithm/gdal
)
configure_file(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/algorithm/gdal/__init__.py
    COPYONLY
)
