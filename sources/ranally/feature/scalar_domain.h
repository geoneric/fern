#pragma once
#include "ranally/feature/domain.h"


namespace ranally {

//! Domain when spatio-temporal locations are not relevant.
/*!
  \sa        .
*/
class ScalarDomain:
    public Domain
{

public:

                   ScalarDomain        ()=default;

                   ScalarDomain        (ScalarDomain const&)=delete;

    ScalarDomain&  operator=           (ScalarDomain const&)=delete;

                   ScalarDomain        (ScalarDomain&&)=delete;

    ScalarDomain&  operator=           (ScalarDomain&&)=delete;

                   ~ScalarDomain       () noexcept(true)=default;

protected:

private:

};

} // namespace ranally
