// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once


namespace fern {

//! Abstract base class for classes that model some form of uncertainty.
/*!
*/
class Uncertainty
{

public:

protected:

                   Uncertainty         ()=default;

    virtual        ~Uncertainty        ()=default;

                   Uncertainty         (Uncertainty&&)=default;

    Uncertainty&   operator=           (Uncertainty&&)=default;

                   Uncertainty         (Uncertainty const&)=default;

    Uncertainty&   operator=           (Uncertainty const&)=default;

private:

};

} // namespace fern
