#include "fern/operation/std/operations.h"
#include "fern/operation/std/abs.h"
#include "fern/operation/std/add.h"
#include "fern/operation/std/int32.h"
#include "fern/operation/std/read.h"
#include "fern/operation/std/write.h"


namespace fern {

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
            std::shared_ptr<Operation>(new Int32()),
            std::shared_ptr<Operation>(new Read()),
            std::shared_ptr<Operation>(new Write())
        }));
    }

    return detail::operations;
}

} // namespace fern
