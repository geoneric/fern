IF(FERN_ALGORITHM)
    SET(TEST_NAMES
        array_test
        masked_array_test
        masked_constant_test
        masked_raster_test
        raster_test
    )
ENDIF()
### ELSE()
###     SET(TEST_NAMES
###         array_test
###         array_value_test
###         constant_attribute_test
###         feature_test
###         masked_array_test
###         masked_array_value_test
###         masked_constant_test
###         masked_raster_test
###         spatial_attribute_test
###         spatial_domain_test
###     )
### ENDIF()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST2(feature_core ${NAME} fern_feature_core)
ENDFOREACH()
