// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <memory>
#include "fern/core/string.h"


namespace fern {

class Uncertainty;

//! TODO
/*!
   TODO
*/
class UncertML2Parser
{

public:

                   UncertML2Parser     ()=default;

                   ~UncertML2Parser    ()=default;

                   UncertML2Parser     (UncertML2Parser&&)=delete;

    UncertML2Parser& operator=         (UncertML2Parser&&)=delete;

                   UncertML2Parser     (UncertML2Parser const&)=delete;

    UncertML2Parser& operator=         (UncertML2Parser const&)=delete;

    std::shared_ptr<Uncertainty> parse (String const& xml) const;

private:

    std::shared_ptr<Uncertainty> parse (std::istream& stream) const;

};

} // namespace fern
