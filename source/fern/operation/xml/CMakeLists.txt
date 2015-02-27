if(FERN_TEST)
    add_subdirectory(test)
endif()

add_parser_generation_command(operation
    --namespace-map http://www.geoneric.eu/fern=fern)

set(SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/operation-pskel.cxx
    operation_xml_parser
)
add_library(fern_operation_xml
    ${SOURCES}
)
target_link_libraries(fern_operation_xml
    fern_operation_core
    # fern_operation_std
    ${EXPAT_LIBRARY}
)
