ADD_SUBDIRECTORY(test)

SET(SOURCES
    dataset
    driver
    file
)
ADD_LIBRARY(fern_io_core
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(fern_io_core
    fern_core
)
