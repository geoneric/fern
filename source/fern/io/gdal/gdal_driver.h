// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/feature/core/attributes.h"
#include "fern/io/core/driver.h"


class GDALDriver;

namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class GDALDriver:
    public Driver
{

public:

                   GDALDriver          (String const& name);

                   GDALDriver          (::GDALDriver* driver);

                   GDALDriver          (GDALDriver const&)=delete;

    GDALDriver&    operator=           (GDALDriver const&)=delete;

                   GDALDriver          (GDALDriver&&)=delete;

    GDALDriver&    operator=           (GDALDriver&&)=delete;

                   ~GDALDriver         ()=default;

    bool           can_open            (String const& name,
                                        OpenMode open_mode);

    // ExpressionType expression_type     (DataName const& data_name);

    std::shared_ptr<Dataset> open      (String const& name,
                                        OpenMode open_mode);

private:

    ::GDALDriver*  _driver;

    bool           can_open_for_read   (String const& name);

    bool           can_open_for_update (String const& name);

    bool           can_open_for_overwrite(
                                        String const& name);

};

} // namespace fern
