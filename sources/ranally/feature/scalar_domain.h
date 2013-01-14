#pragma once
#include "ranally/feature/domain.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ScalarDomain:
    public Domain
{

    friend class ScalarDomainTest;

public:

                   ScalarDomain        ()=default;

                   ScalarDomain        (ScalarDomain const&)=delete;

    ScalarDomain&  operator=           (ScalarDomain const&)=delete;

                   ScalarDomain        (ScalarDomain&&)=delete;

    ScalarDomain&  operator=           (ScalarDomain&&)=delete;

                   ~ScalarDomain       () noexcept(true) =default;

protected:

private:

};

} // namespace ranally
