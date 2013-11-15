#pragma once
#include <cstdlib>


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class GDALClient
{

public:

protected:

                   GDALClient          ();

                   GDALClient          (GDALClient const&)=delete;

    GDALClient&    operator=           (GDALClient const&)=delete;

                   GDALClient          (GDALClient&&)=delete;

    GDALClient&    operator=           (GDALClient&&)=delete;

    virtual        ~GDALClient         ();

private:

    //! Number of times an instance is created.
    static size_t  _count;

    void           register_all_drivers();

    void           deregister_all_drivers();

};

} // namespace fern
