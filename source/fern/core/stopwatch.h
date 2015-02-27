#pragma once
#include <boost/timer/timer.hpp>


namespace fern {

// uint64_t           rdtsc               ();


//! Class for timing snippets of code.
/*!
    A stopwatch runs a function and keeps track of the time spent. After
    creation, the stopwatch can be querried for the amount of time spent.

    The rule of thumb is:

    - wall time < user time: The process is CPU bound and takes advantage of
        parallel execution on multiple cores/CPUs.
    - wall time ≈ user time: The process is CPU bound and takes no advantage of
        parallel execution.
    - wall time > user time: The process is I/O bound. Execution on multiple
        cores would be of little to no advantage.
*/
class Stopwatch
{

public:

    using nanosecond_type = boost::timer::nanosecond_type;

    template<
        class Function,
        class... Arguments>
    explicit       Stopwatch           (Function&& function,
                                        Arguments&&... arguments);

                   Stopwatch           (Stopwatch const& stopwatch)=delete;

                   Stopwatch           (Stopwatch&& stopwatch)=delete;

    Stopwatch&     operator=           (Stopwatch const& stopwatch)=delete;

    Stopwatch&     operator=           (Stopwatch&& stopwatch)=delete;

                   ~Stopwatch          ()=default;

    nanosecond_type wall_time          () const;

    nanosecond_type user_time          () const;

    nanosecond_type system_time        () const;

    // uint64_t       clock_ticks         () const;

private:

    boost::timer::cpu_times _cpu_times;

    // uint64_t       _clock_ticks;

};


//! Construct a stopwatch, given a \a function with its \a arguments.
/*!
  \tparam    Function Type of function to execute.
  \tparam    Arguments Type of arguments to pass to \a function.
  \param     function Function to call.
  \param     arguments Arguments to pass to \a function.

  The \a function passed in is executed immediately, blocking the constructor
  until the \a function has finished.
*/
template<
    class Function,
    class... Arguments>
inline Stopwatch::Stopwatch(
    Function&& function,
    Arguments&&... arguments)
{
    // Execute function while keeping track of the time spent executing it.
    boost::timer::cpu_timer timer;

    // _clock_ticks = rdtsc();
    function(arguments...);
    // _clock_ticks = rdtsc() - _clock_ticks;

    timer.stop();
    _cpu_times = timer.elapsed();
}

} // namespace fern
