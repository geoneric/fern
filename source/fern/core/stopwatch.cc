// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/core/stopwatch.h"
/// #ifdef __GNUC__
/// #include <x86intrin.h>
/// #endif


namespace fern {

/// //! Return the CPU time stamp counter.
/// /*!
///   \tparam    .
///   \param     .
///   \return    .
///   \exception .
///   \warning   .
///   \sa        .
/// */
/// uint64_t rdtsc()
/// {
///     // http://msdn.microsoft.com/en-us/library/ee417693(v=vs.85).aspx
///     return __rdtsc();
/// }


//! Return time as would be measured by an ordinary clock.
/*!
  \return    Number of nanoseconds.
  \sa        user_time(), system_time()
*/
Stopwatch::nanosecond_type Stopwatch::wall_time() const
{
    return _cpu_times.wall;
}


//! Return "CPU time charged for the execution of user instructions of the calling process."
/*!
  \return    Number of nanoseconds.
  \sa        wall_time(), system_time()

  http://pubs.opengroup.org/onlinepubs/9699919799/functions/times.html
*/
Stopwatch::nanosecond_type Stopwatch::user_time() const
{
    return _cpu_times.user;
}


//! Return "CPU time charged for the execution by the system on behalf of the calling process."
/*!
  \return    Number of nanoseconds.
  \sa        wall_time(), user_time()

  http://pubs.opengroup.org/onlinepubs/9699919799/functions/times.html
*/
Stopwatch::nanosecond_type Stopwatch::system_time() const
{
    return _cpu_times.system;
}


/// uint64_t Stopwatch::clock_ticks() const
/// {
///     return _clock_ticks;
/// }

} // namespace fern
