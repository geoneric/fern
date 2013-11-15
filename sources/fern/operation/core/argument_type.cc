#include "fern/operation/core/argument_type.h"


namespace fern {

std::ostream& operator<<(
    std::ostream& stream,
    ArgumentType const& argument_type)
{
    switch(argument_type) {
        case ArgumentType::AT_ATTRIBUTE: {
            stream << "AT_ATTRIBUTE";
            break;
        }
        case ArgumentType::AT_FEATURE: {
            stream << "AT_FEATURE";
            break;
        }
    }

    return stream;
}

} // namespace fern
