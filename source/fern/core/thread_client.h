#pragma once
#include "fern/core/thread_pool.h"


namespace fern {

//! Class for managing resources used by threading.
/*!
    \sa            ThreadPool

    Instantiating this class will create a thread pool that can be used for
    executing concurrent tasks.
*/
class ThreadClient
{

public:

    static size_t  hardware_concurrency();

    static ThreadPool&
                   pool                ();

                   ThreadClient        (size_t nr_threads=
                                          hardware_concurrency());

                   ~ThreadClient       ();

private:

};

} // namespace fern
