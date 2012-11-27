ADD_SUBDIRECTORY(test)

SET(SOURCES
    messages
    string
)
ADD_LIBRARY(RanallyCore
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(RanallyCore
    ${ICU_LIBRARIES}
)
