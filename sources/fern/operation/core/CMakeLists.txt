ADD_SUBDIRECTORY(test)

SET(SOURCES
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
ADD_LIBRARY_AND_OBJECT_LIBRARY(fern_operation_core ${SOURCES})
TARGET_LINK_LIBRARIES(fern_operation_core
    fern_feature_visitor
    fern_core
)
