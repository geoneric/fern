add_subdirectory(test)

set(SOURCES
    back_end
)
add_library(fern_back_end
    ${SOURCES}
)
# target_link_libraries(fern_back_end
# )
