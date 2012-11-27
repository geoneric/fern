ADD_SUBDIRECTORY(test)

SET(SOURCES
    exception
    messages
    string
)
ADD_LIBRARY(RanallyCore
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(RanallyCore
    ${ICU_LIBRARIES}
)
