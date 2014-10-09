#pragma once
#include "fern/feature/domain.h"


namespace fern {

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

                   ~ScalarDomain       ();

protected:

private:

};

} // namespace fern
