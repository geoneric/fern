#include "fern/core/join_threads.h"


namespace fern {

JoinThreads::JoinThreads(
    std::vector<std::thread>& threads)

    : _threads(threads)

{
}


JoinThreads::~JoinThreads()
{
    for(auto& thread: _threads) {
        if(thread.joinable()) {
            thread.join();
        }
    }
}

} // namespace fern
