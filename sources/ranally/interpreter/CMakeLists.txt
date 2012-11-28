ADD_SUBDIRECTORY(test)

SET(SOURCES
    interpreter
)
ADD_LIBRARY(ranally_interpreter
    ${SOURCES}
)
ADD_DEPENDENCIES(ranally_interpreter
    # This header is generated in the operation project.
    operation-xml.h
)
