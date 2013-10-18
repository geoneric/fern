ADD_SUBDIRECTORY(test)

SET(SOURCES
    dataset
    driver
    file
)
ADD_LIBRARY(geoneric_io_core
    ${SOURCES}
)
TARGET_LINK_LIBRARIES(geoneric_io_core
    geoneric_core
)
