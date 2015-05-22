add_test_conditionally(test)

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
add_object_library(
    TARGET fern_language_operation_core_objects
    SOURCES ${SOURCES}
    LIBRARY fern_language)
