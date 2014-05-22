#pragma once
#include "fern/core/thread_pool.h"


namespace fern {

class ThreadClient
{

public:

    static ThreadPool& pool            ();

                   ThreadClient        (size_t nr_threads=
                                          std::thread::hardware_concurrency());

                   ~ThreadClient       ();

private:

};

} // namespace fern
