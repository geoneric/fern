ADD_SUBDIRECTORY(test)

SET(SOURCES
    back_end
)
ADD_LIBRARY(fern_back_end
    ${SOURCES}
)
# TARGET_LINK_LIBRARIES(fern_back_end
# )
