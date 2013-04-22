#pragma once
#include <memory>
#include "ranally/operation/core/operations.h"


namespace ranally {

// extern std::shared_ptr<Operations> operations;

std::shared_ptr<Operations> const& operations ();

} // namespace ranally
