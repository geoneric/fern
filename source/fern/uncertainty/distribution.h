// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/uncertainty/uncertainty.h"


namespace fern {

//! Abstract base class for classes that model uncertainty using a distribution.
/*!
*/
class Distribution:
    public Uncertainty
{

public:

protected:

                   Distribution        ()=default;

    virtual        ~Distribution       ()=default;

                   Distribution        (Distribution&&)=default;

    Distribution&  operator=           (Distribution&&)=default;

                   Distribution        (Distribution const&)=default;

    Distribution&  operator=           (Distribution const&)=default;

private:

};

} // namespace fern
