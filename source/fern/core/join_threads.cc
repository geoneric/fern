// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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
