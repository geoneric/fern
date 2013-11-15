#pragma once
#include "fern/io/core/driver.h"


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class GeonericDriver:
    public Driver
{

public:

                   GeonericDriver      ();

                   GeonericDriver      (GeonericDriver const&)=delete;

    GeonericDriver& operator=          (GeonericDriver const&)=delete;

                   GeonericDriver      (GeonericDriver&&)=delete;

    GeonericDriver& operator=          (GeonericDriver&&)=delete;

                   ~GeonericDriver     ()=default;

    bool           can_open            (String const& name,
                                        OpenMode open_mode);

    // ExpressionType expression_type     (DataName const& data_name);

    std::shared_ptr<Dataset> open      (String const& name,
                                        OpenMode open_mode);

private:

    bool           can_open_for_read   (String const& name);

    bool           can_open_for_overwrite(
                                        String const& name);

    bool           can_open_for_update (String const& name);

};

} // namespace fern
