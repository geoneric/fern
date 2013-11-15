#include "fern/command/message.h"
#include <iostream>
#include "fern/configure.h"


namespace fern {

void show_version()
{
    std::cout << "geoneric " << GEONERIC_VERSION << "\n";
    std::cout << GEONERIC_COPYRIGHT << "\n";
}

} // namespace fern
