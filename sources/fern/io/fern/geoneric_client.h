#pragma once
#include <cstdlib>
#include "fern/core/string.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class GeonericClient
{

public:

protected:

                   GeonericClient      ();

                   GeonericClient      (GeonericClient const&)=delete;

    GeonericClient& operator=          (GeonericClient const&)=delete;

                   GeonericClient      (GeonericClient&&)=delete;

    GeonericClient& operator=          (GeonericClient&&)=delete;

    virtual        ~GeonericClient     ();

private:

    //! Number of times an instance is created.
    static size_t  _count;

    static String const _driver_name;

    void           register_driver();

    void           deregister_driver();

};

} // namespace fern
