include_directories(
    ${CMAKE_SOURCE_DIR}/source
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(pow_3
    main
)
target_link_libraries(pow_3
    fern_feature_core
    stdc++
)
