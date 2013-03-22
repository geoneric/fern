#include "ranally/command/message.h"
#include <iostream>
#include "ranally/configure.h"


namespace ranally {

void show_version()
{
    std::cout << "ranally " << RANALLY_VERSION << "\n";
    std::cout << RANALLY_COPYRIGHT << "\n";
}

} // namespace ranally
