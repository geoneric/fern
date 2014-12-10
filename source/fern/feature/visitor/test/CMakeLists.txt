SET(TEST_NAMES
    attribute_type_visitor_test
)
FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(
        SCOPE feature_visitor
        NAME ${NAME}
        LINK_LIBRARIES fern_feature_visitor)
ENDFOREACH()
