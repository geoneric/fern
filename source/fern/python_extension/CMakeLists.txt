ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/swig_runtime.h
    COMMAND echo "\\#pragma once" > swig_runtime.h
    COMMAND echo "\\#pragma GCC system_header" >> swig_runtime.h
    COMMAND echo "/*! @ingroup MyGlobals" >> swig_runtime.h
    COMMAND echo " * @{ */" >> swig_runtime.h
    COMMAND ${SWIG_EXECUTABLE} -python -external-runtime swig_runtime.h.tmp
    COMMAND cat swig_runtime.h.tmp >> swig_runtime.h
    COMMAND rm swig_runtime.h.tmp
    COMMAND echo "/*! @} */" >> swig_runtime.h
)
ADD_CUSTOM_TARGET(swig_runtime DEPENDS swig_runtime.h)


ADD_SUBDIRECTORY(algorithm)


CONFIGURE_FILE(__init__.py.in
    ${PROJECT_BINARY_DIR}/bin/fern/__init__.py
    COPYONLY
)
