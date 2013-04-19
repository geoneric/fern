#include "ranally/operation/std/operations.h"
#include "ranally/operation/std/abs.h"

namespace ranally {

std::shared_ptr<Operations> operations(new Operations({
    std::shared_ptr<Operation>(new Abs())
}));

} // namespace ranally
