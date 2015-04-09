// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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
        detail::operations = std::make_shared<Operations>(
            std::initializer_list<std::shared_ptr<Operation>>{
                std::shared_ptr<Operation>(std::make_shared<Abs>()),
                std::shared_ptr<Operation>(std::make_shared<Add>()),
                std::shared_ptr<Operation>(std::make_shared<Int32>()),
                std::shared_ptr<Operation>(std::make_shared<Read>()),
                std::shared_ptr<Operation>(std::make_shared<Write>())
        });
    }

    return detail::operations;
}

} // namespace fern
