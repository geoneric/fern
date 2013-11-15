SET(TEST_NAMES
    array_test
    array_value_test
    constant_attribute_test
    feature_test
    masked_array_test
    masked_array_value_test
    spatial_attribute_test
    spatial_domain_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(feature_core_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(feature_core_${NAME}
        fern_feature_core
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(NAME feature_core_${NAME} COMMAND feature_core_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
