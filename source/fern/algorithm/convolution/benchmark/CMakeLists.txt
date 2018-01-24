add_executable(convolve_benchmark
    convolve_benchmark
)

target_link_libraries(convolve_benchmark
    PRIVATE
        benchmark::benchmark
)
