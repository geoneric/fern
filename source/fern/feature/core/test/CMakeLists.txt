set(TEST_NAMES
    array_test
    masked_array_test
    masked_raster_test
    masked_scalar_test
    raster_test
)

foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE feature_core
        NAME ${NAME}
        LINK_LIBRARIES fern_feature_core)
endforeach()
