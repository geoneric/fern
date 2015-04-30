// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/operation/core/operation.h"


namespace fern {
namespace language {

class Add:
    public Operation
{

public:

                   Add                 ();

                   ~Add                ()=default;

                   Add                 (Add&& other)=delete;

    Add&           operator=           (Add&& other)=delete;

                   Add                 (Add const& other)=delete;

    Add&           operator=           (Add const& other)=delete;

    ExpressionType expression_type     (size_t index,
                                        std::vector<ExpressionType> const&
                                            argument_types) const;

    std::vector<std::shared_ptr<Argument>>
                   execute             (std::vector<std::shared_ptr<Argument>>
                                            const& arguments) const;

private:

};

} // namespace language
} // namespace fern
