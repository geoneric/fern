#pragma once
#include "geoneric/uncertainty/uncertainty.h"


namespace geoneric {

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

} // namespace geoneric
