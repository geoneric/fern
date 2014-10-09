#pragma once
#include <memory>
#include "fern/operation/core/operations.h"


namespace fern {

// extern std::shared_ptr<Operations> operations;

std::shared_ptr<Operations> const& operations ();

} // namespace fern
