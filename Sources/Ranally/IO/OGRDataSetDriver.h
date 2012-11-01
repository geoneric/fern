#pragma once
#include "Ranally/IO/DataSetDriver.h"
#include "Ranally/IO/OGRDataSet.h"


class OGRSFDriver;

namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class OGRDataSetDriver:
    public DataSetDriver
{

    friend class OGRDataSetDriverTest;

public:

                   OGRDataSetDriver    (String const& name);

                   ~OGRDataSetDriver   ();

    bool           exists              (String const& name) const;

    OGRDataSet*    create              (String const& name) const;

    void           remove              (String const& name) const;

    OGRDataSet*    open                (String const& name) const;

private:

    OGRSFDriver*     _driver;

};

} // namespace ranally
