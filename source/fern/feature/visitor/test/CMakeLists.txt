set(TEST_NAMES
    attribute_type_visitor_test
)
foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE feature_visitor
        NAME ${NAME}
        LINK_LIBRARIES fern_feature_visitor)
endforeach()
