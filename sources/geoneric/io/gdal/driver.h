#pragma once
#include <memory>
#include "geoneric/io/gdal/dataset.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Driver
{

public:

    virtual std::shared_ptr<Dataset> open(String const& name)=0;

protected:

                   Driver             ()=default;

                   Driver             (Driver const&)=delete;

    Driver&       operator=           (Driver const&)=delete;

                   Driver             (Driver&&)=delete;

    Driver&       operator=           (Driver&&)=delete;

    virtual        ~Driver            ()=default;

private:

};

} // namespace geoneric
