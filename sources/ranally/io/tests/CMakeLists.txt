SET(TEST_NAMES
    hdf5_dataset_driver_test
    hdf5_dataset_test
    ogr_dataset_driver_test
)

MACRO(ADD_UNIT_TEST NAME)
    ADD_EXECUTABLE(${NAME} ${NAME})
    TARGET_LINK_LIBRARIES(${NAME}
        RanallyIO
    )
    ADD_TEST(NAME ${NAME} COMMAND ${NAME})
ENDMACRO()

FOREACH(NAME ${TEST_NAMES})
    ADD_UNIT_TEST(${NAME})
ENDFOREACH()

### ADD_CUSTOM_TARGET(copyTestData
###     COMMAND ${CMAKE_COMMAND} -E copy Point.json ${CMAKE_CURRENT_BINARY_DIR}
### 
###     COMMAND ${CMAKE_COMMAND} -E copy Point.json
###         ${CMAKE_CURRENT_BINARY_DIR}/ReadOnlyPoint.json
###     COMMAND chmod 444 ${CMAKE_CURRENT_BINARY_DIR}/ReadOnlyPoint.json
### 
###     COMMAND ${CMAKE_COMMAND} -E copy Point.json
###         ${CMAKE_CURRENT_BINARY_DIR}/WriteOnlyPoint.json
###     COMMAND chmod 222 ${CMAKE_CURRENT_BINARY_DIR}/WriteOnlyPoint.json
### 
###     COMMAND ${CMAKE_COMMAND} -E copy Point.json
###         ${CMAKE_CURRENT_BINARY_DIR}/ExecuteOnlyPoint.json
###     COMMAND chmod 111 ${CMAKE_CURRENT_BINARY_DIR}/ExecuteOnlyPoint.json
### 
###     COMMAND ${CMAKE_COMMAND} -E make_directory
###         ${CMAKE_CURRENT_BINARY_DIR}/ReadOnlyDir
###     COMMAND chmod 666 ${CMAKE_CURRENT_BINARY_DIR}/ReadOnlyDir
### 
###     WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/Data/Tests
###     COMMENT "Copying test data" VERBATIM
### )
### TODO ADD_DEPENDENCIES(runTestRanallyIO
### TODO     copyTestData
### TODO )
