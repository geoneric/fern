SET(SOURCES
    algorithm
    python_extension
)
ADD_LIBRARY(fern_algorithm_gdal SHARED ${SOURCES})
ADD_DEPENDENCIES(fern_algorithm_gdal swig_runtime)
TARGET_LINK_LIBRARIES(fern_algorithm_gdal
    fern_algorithm_core
    fern_feature_core
    ${GDAL_LIBRARY}
    stdc++
)
CONFIGURE_PYTHON_EXTENSION(fern_algorithm_gdal "fern_algorithm_gdal")
