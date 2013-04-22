#include "ranally/operation/std/operations.h"
#include "ranally/operation/std/abs.h"
#include "ranally/operation/std/add.h"
#include "ranally/operation/std/int32.h"

namespace ranally {

// std::shared_ptr<Operations> operations(new Operations({
//     std::shared_ptr<Operation>(new Abs())
// }));

namespace detail {

static std::shared_ptr<Operations> operations;

} // namespace detail


std::shared_ptr<Operations> const& operations()
{
    if(!detail::operations) {
        detail::operations.reset(new Operations({
            std::shared_ptr<Operation>(new Abs()),
            std::shared_ptr<Operation>(new Add()),
            std::shared_ptr<Operation>(new Int32())
        }));
    }

    return detail::operations;
}

} // namespace ranally
