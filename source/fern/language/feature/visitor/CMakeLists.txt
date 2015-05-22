add_test_conditionally(test)

set(SOURCES
    attribute_visitor
    attribute_type_visitor
)
add_object_library(
    TARGET fern_language_feature_visitor_objects
    SOURCES ${SOURCES}
    LIBRARY fern_language)
# target_link_libraries(fern_language_feature_visitor
#     fern_core
# )
