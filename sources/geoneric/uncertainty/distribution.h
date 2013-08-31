#pragma once
#include "ranally/uncertainty/uncertainty.h"


namespace ranally {

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

} // namespace ranally
