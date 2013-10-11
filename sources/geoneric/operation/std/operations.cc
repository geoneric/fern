#include "geoneric/operation/std/operations.h"
#include "geoneric/operation/std/abs.h"
#include "geoneric/operation/std/add.h"
#include "geoneric/operation/std/int32.h"
#include "geoneric/operation/std/read.h"
#include "geoneric/operation/std/write.h"


namespace geoneric {

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

} // namespace geoneric
