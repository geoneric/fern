SET(LIBRARY_OUTPUT_PATH
    ${PROJECT_BINARY_DIR}/bin
    CACHE PATH
    "Single directory for all libraries."
)
SET(EXECUTABLE_OUTPUT_PATH
    ${PROJECT_BINARY_DIR}/bin
    CACHE PATH
    "Single directory for all executables."
)
MARK_AS_ADVANCED(
    LIBRARY_OUTPUT_PATH
    EXECUTABLE_OUTPUT_PATH
)


IF(UNIX AND NOT CYGWIN)
    IF(APPLE)
        SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
        SET(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")
    ELSE()
        SET(CMAKE_SKIP_BUILD_RPATH FALSE)
        SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
        SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
        SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
    ENDIF()
ENDIF()


INCLUDE(CheckCXXCompilerFlag)


IF(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR
        ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    # The code assumes integer overflow and underflow wraps. This is not
    # guaranteed by the standard. Gcc may assume overflow/underflow will not
    # happen and optimize the code accordingly. That's why we added
    # -fno-strict-overflow. It would be better if we don't assume
    # over/underflow wraps.
    # See http://www.airs.com/blog/archives/120
    # See out of range policy of add algorithm for signed integrals.
    #
    # Add as many warning options as possible/useful:
    # - https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
    # TODO Maybe add:
    # -Wconversion
    # -Wsign-conversion
    SET(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wcast-qual -Wzero-as-null-pointer-constant -Wwrite-strings -Werror=strict-aliasing -pedantic -fno-strict-overflow -ftemplate-backtrace-limit=0"
    )
    IF(APPLE)
        SET(CMAKE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs"
        )
    ENDIF()
ENDIF()

IF(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    SET(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -std=c++14"
    )
ELSEIF(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    SET(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -stdlib=libc++ -std=c++1y"
    )
ENDIF()


SET(CMAKE_CXX_FLAGS_RELEASE
    # Disable range checks in release builds.
    "${CMAKE_CXX_FLAGS_RELEASE} -DBOOST_DISABLE_ASSERTS"
)

# Add the PIC compiler flag if needed.
IF(UNIX AND NOT WIN32)
    IF(CMAKE_SIZEOF_VOID_P MATCHES "8")
        CHECK_CXX_COMPILER_FLAG("-fPIC" WITH_FPIC)
        IF(WITH_FPIC)
            ADD_DEFINITIONS(-fPIC)
        ENDIF()
    ENDIF()
ENDIF()
