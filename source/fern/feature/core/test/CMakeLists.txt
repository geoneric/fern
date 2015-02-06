if(FERN_ALGORITHM)
    set(TEST_NAMES
        array_test
        masked_array_test
        masked_constant_test
        masked_raster_test
        raster_test
    )
endif()
### else()
###     set(TEST_NAMES
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
### endif()

foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE feature_core
        NAME ${NAME}
        LINK_LIBRARIES fern_feature_core)
endforeach()
