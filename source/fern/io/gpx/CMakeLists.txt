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
add_library(fern_io_gpx
    ${SOURCES}
)
target_link_libraries(fern_io_gpx
    fern_core
    ${XercesC_LIBRARIES}
)
