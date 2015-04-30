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

class Int32:
    public Operation
{

public:

                   Int32               ();

                   ~Int32              ()=default;

                   Int32               (Int32&& other)=delete;

    Int32&         operator=           (Int32&& other)=delete;

                   Int32               (Int32 const& other)=delete;

    Int32&         operator=           (Int32 const& other)=delete;

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
