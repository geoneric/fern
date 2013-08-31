#include "geoneric/command/message.h"
#include <iostream>
#include "geoneric/configure.h"


namespace geoneric {

void show_version()
{
    std::cout << "geoneric " << GEONERIC_VERSION << "\n";
    std::cout << GEONERIC_COPYRIGHT << "\n";
}

} // namespace geoneric
