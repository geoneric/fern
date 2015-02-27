if(FERN_TEST)
    add_subdirectory(test)
endif()

set(SOURCES
    argument
    argument_type
    attribute_argument
    expression_type_calculation
    feature_argument
    operation
    operations
    parameter
    result
    expression_types
)
add_library_and_object_library(fern_operation_core ${SOURCES})
target_link_libraries(fern_operation_core
    fern_feature_visitor
    fern_core
)
