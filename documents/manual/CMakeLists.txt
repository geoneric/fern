ADD_SUBDIRECTORY(fern)

INSTALL(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fern/algorithm/_build/html/
    DESTINATION doc/manual/algorithm
)
