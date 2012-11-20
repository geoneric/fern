ADD_SUBDIRECTORY(Tests)

SET(SOURCES
    interpreter
)
ADD_LIBRARY(RanallyInterpreter
    ${SOURCES}
)
ADD_DEPENDENCIES(RanallyInterpreter
    # This header is generated in the Operation project.
    operation-xml.h
)
