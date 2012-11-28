ADD_SUBDIRECTORY(test)

SET(SOURCES
    exception
    messages
    string
)
ADD_LIBRARY(ranally_core
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(ranally_core
    ${ICU_LIBRARIES}
)
