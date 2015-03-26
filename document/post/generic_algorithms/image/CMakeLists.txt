add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/field_raster.png
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/field_raster.py
        ${CMAKE_CURRENT_BINARY_DIR}/field_raster.png
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/field_raster.py
)

add_custom_target(post_generic_algorithms_images
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/field_raster.png
)
