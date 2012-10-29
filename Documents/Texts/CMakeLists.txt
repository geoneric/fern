ADD_SUBDIRECTORY(Project)
IF(NOT APPLE)
    # TODO Issue calling dot.
    ADD_SUBDIRECTORY(DataModel)
ENDIF()
ADD_SUBDIRECTORY(Development)
