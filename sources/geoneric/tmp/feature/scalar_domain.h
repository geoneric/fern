#pragma once
#include "geoneric/feature/domain.h"


namespace geoneric {

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

} // namespace geoneric
