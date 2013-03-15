ADD_SUBDIRECTORY(test)

SET(SOURCES
    exception
    io_error
    messages
    parse_error
    script_error
    stack
    string
    validate_error
)
ADD_LIBRARY(ranally_core
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(ranally_core
    ${ICU_LIBRARIES}
)
