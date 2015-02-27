if(FERN_TEST)
    add_subdirectory(test)
endif()

set(SOURCES
    back_end
)
add_library(fern_back_end
    ${SOURCES}
)
# target_link_libraries(fern_back_end
# )
