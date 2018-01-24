// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include <benchmark/benchmark.h>


// template<
//     typename... T>
// class Fixture;


template<
    typename T>
class Fixture:
    public benchmark::Fixture
{

public:

    T blah1 = 5;

private:


};


// template<
//     typename T1,
//     typename T2>
// class Fixture:
//     public benchmark::Fixture
// {
// 
// public:
// 
//     T1 blah1 = 5;
//     T2 blah2 = 6;
// 
// private:
// 
// 
// };


// See paper for how to report HPC results
// See paper about execution policies, executors, etc. Verify our
// approach is similar.

// benchmarks:
// - kernel value types
//     - same raster
//     - different kernel value types (int, float, bool)
//     - divide by weights
//     - sequential
// - compile-time vs runtime kernels
//     - same raster
//     - different kernels
//     - divide by weights
// - grain size
//     - same raster
//     - same kernel
//     - max number of worker threads
//     - different grain size
// - scaling
//     - same raster
//     ₋ same kernel
//     - same grain size
//     - different number of worker threads


BENCHMARK_TEMPLATE_F(Fixture, convolve, double)(
    benchmark::State& state)
{


    auto meh = blah1;
    for(auto _: state) {
        std::vector<double>(1000);
    }
}


// BENCHMARK(slope);
BENCHMARK_MAIN();


// If the benchmarked code itself uses threads and you want to compare
// it to single-threaded code, you may want to use real-time ("wallclock")
// measurements for latency comparisons:
// 
// BENCHMARK(BM_test)->Range(8, 8<<10)->UseRealTime();
// Without UseRealTime, CPU time is used by default.
//
// nr_iterations
// nr_repetitions
// wall_clock vs cpu time
