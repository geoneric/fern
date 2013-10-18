#pragma once
#include <cstdlib>


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class HDF5Client
{

public:

protected:

                   HDF5Client          ();

                   HDF5Client          (HDF5Client const&)=delete;

    HDF5Client&    operator=           (HDF5Client const&)=delete;

                   HDF5Client          (HDF5Client&&)=delete;

    HDF5Client&    operator=           (HDF5Client&&)=delete;

    virtual        ~HDF5Client         ();

private:

    //! Number of times an instance is created.
    static size_t  _count;

};

} // namespace geoneric
