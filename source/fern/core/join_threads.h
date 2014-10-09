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
