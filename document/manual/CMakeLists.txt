add_subdirectory(fern)

install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fern/_build/html/
    DESTINATION doc/fern/manual
    CONFIGURATIONS Release
)
