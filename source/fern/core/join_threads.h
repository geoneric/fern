// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <thread>
#include <vector>


namespace fern {

class JoinThreads
{

public:

    explicit       JoinThreads         (std::vector<std::thread>& threads);

                   ~JoinThreads        ();

private:

    std::vector<std::thread>& _threads;

};


} // namespace fern
