#include "fern/command/message.h"
#include <iostream>
#include "fern/configure.h"


namespace fern {

void show_version()
{
    std::cout << "fern " << FERN_VERSION << "\n";
    std::cout << FERN_COPYRIGHT << "\n";
}

} // namespace fern
