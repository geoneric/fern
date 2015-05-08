set(TEST_NAMES
    symbol_table_test
)

foreach(NAME ${TEST_NAMES})
    add_unit_test2(
        SCOPE language_core
        NAME ${NAME})
endforeach()
