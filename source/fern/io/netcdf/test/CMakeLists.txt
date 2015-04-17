set(TEST_NAMES
    read_test
)

foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE io_netcdf
        NAME ${NAME}
        # LINK_LIBRARIES fern_core)
        )
endforeach()
