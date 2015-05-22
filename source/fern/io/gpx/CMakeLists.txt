add_test_conditionally(test)

add_tree_generation_command(gpx-1.0
    --namespace-map http://www.topografix.com/GPX/1/0=gpx_1_0)
add_tree_generation_command(gpx-1.1
    --namespace-map http://www.topografix.com/GPX/1/1=gpx_1_1)

set(SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/gpx-1.0.cxx
    ${CMAKE_CURRENT_BINARY_DIR}/gpx-1.1.cxx
    parse
)
add_object_library(
    TARGET fern_io_gpx_objects
    SOURCES ${SOURCES}
    LIBRARY fern_io)
