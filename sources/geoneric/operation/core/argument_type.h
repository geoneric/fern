#pragma once
#include <iostream>


namespace geoneric {

enum class ArgumentType {

    AT_ATTRIBUTE,

    AT_FEATURE

};


std::ostream&      operator<<          (std::ostream& stream,
                                        ArgumentType const& argument_type);

} // namespace geoneric
