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

class Write:
    public Operation
{

public:

                   Write               ();

                   ~Write              ()=default;

                   Write               (Write&& other)=delete;

    Write&         operator=           (Write&& other)=delete;

                   Write               (Write const& other)=delete;

    Write&         operator=           (Write const& other)=delete;

    std::vector<std::shared_ptr<Argument>>
                   execute             (std::vector<std::shared_ptr<Argument>>
                                            const& arguments) const;

private:

};

} // namespace fern
