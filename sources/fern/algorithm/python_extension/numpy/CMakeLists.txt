### SET(SOURCES
###     fern_algorithm_numpy.pyx
### )
### SET_SOURCE_FILES_PROPERTIES(${SOURCES}
###     PROPERTIES
###         CYTHON_IS_CXX TRUE
### )
### CYTHON_ADD_MODULE(
###     fern_algorithm_numpy
###     ${SOURCES}
### )


### ADD_SUBDIRECTORY(test)

SET(SOURCES
    algorithm
    python_extension
)
ADD_LIBRARY(fern_algorithm_numpy SHARED ${SOURCES})
TARGET_LINK_LIBRARIES(fern_algorithm_numpy
    fern_algorithm_core
    fern_feature_core
    stdc++
)
CONFIGURE_PYTHON_EXTENSION(fern_algorithm_numpy "fern_algorithm_numpy")
# INSTALL(
#     TARGETS fern_algorithm_core
#     EXPORT fern_targets
#     ARCHIVE DESTINATION lib
# )


### import numpy
### from fern_algorithm import add
### 
### 
### # ndarray
### array1 = numpy.array([
###     [-2.0, -1.0],
###     [ 0.0,  9.0],
###     [ 1.0,  2.0]
### ])
### print array1
### 
### 
### # fern algorithm
### array2 = fern_algorithm.add(array, 2)
### 
### print array2
