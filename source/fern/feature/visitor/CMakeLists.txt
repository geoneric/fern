add_subdirectory(test)

set(SOURCES
    attribute_visitor
    attribute_type_visitor
)
add_library_and_object_library(fern_feature_visitor ${SOURCES})
target_link_libraries(fern_feature_visitor
    fern_core
)
