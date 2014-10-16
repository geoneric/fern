#pragma once
#include <string>


namespace fern {

void               raise_runtime_error (std::string const& message);

void               raise_unsupported_argument_type_exception(
                                        std::string const& type_represenation);

} // namespace fern
