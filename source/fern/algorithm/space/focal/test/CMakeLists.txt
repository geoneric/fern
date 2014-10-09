SET(TEST_NAMES
    slope_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(algorithm_spatial_focal_${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(algorithm_spatial_focal_${NAME}
        fern_core
        fern_algorithm_core
        fern_feature_core  # fern::extents
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    )
    ADD_TEST(
        NAME algorithm_spatial_focal_${NAME}
        COMMAND algorithm_spatial_focal_${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()
