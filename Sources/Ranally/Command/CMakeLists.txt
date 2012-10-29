INCLUDE(UseBoost)
INCLUDE(UseExpat)
INCLUDE(UseHdf5)
INCLUDE(UseIcu)
INCLUDE(UseLoki)
INCLUDE(UsePython)

INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_BINARY_DIR}/../..
)
ADD_EXECUTABLE(ranally
    Command.cc
    ConvertCommand.cc
    DescribeCommand.cc
    ExecuteCommand.cc
    ImportCommand.cc
    Ranally.cc
)
ADD_DEPENDENCIES(ranally
    Operation-xml.h
)
TARGET_LINK_LIBRARIES(ranally
    RanallyInterpreter
    RanallyIO
    RanallyLanguage
    RanallyPython
    RanallyOperation
    RanallyUtil
    ${PYTHON_LIBRARIES}
    ${BOOST_FILESYSTEM_LIBRARY}
    ${BOOST_PROGRAM_OPTIONS_LIBRARY}
    ${BOOST_REGEX_LIBRARY}
    ${BOOST_SYSTEM_LIBRARY}
    ${EXPAT_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${ICU_LIBRARIES}
)
INSTALL(
    TARGETS ranally
    RUNTIME DESTINATION bin
)
IF(UNIX AND NOT APPLE)
    SET_TARGET_PROPERTIES(ranally
        PROPERTIES
            INSTALL_RPATH "$ORIGIN/../lib"
      )
ENDIF(UNIX AND NOT APPLE)
