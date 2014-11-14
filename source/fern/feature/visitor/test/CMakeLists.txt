SET(TEST_NAMES
    attribute_type_visitor_test
)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(feature_visitor ${NAME} fern_feature_visitor)
ENDFOREACH()
