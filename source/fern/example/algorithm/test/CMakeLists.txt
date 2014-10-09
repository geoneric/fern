SET(TEST_NAMES
    raster_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(example_algorithm_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(example_algorithm_${NAME}
        # fern_core
        # fern_algorithm_core
        # fern_feature_core  # fern::extents
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
        stdc++
    )
    ADD_TEST(
        NAME example_algorithm_${NAME}
        COMMAND example_algorithm_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
