// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/io/ogr/dataset_driver.h"
#include "fern/language/io/ogr/ogr_dataset.h"


class OGRSFDriver;

namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class OGRDatasetDriver:
    public DatasetDriver
{

    friend class OGRDatasetDriverTest;

public:

                   OGRDatasetDriver    (std::string const& name);

                   ~OGRDatasetDriver   ();

    bool           exists              (std::string const& name) const;

    OGRDataset*    create              (std::string const& name) const;

    void           remove              (std::string const& name) const;

    OGRDataset*    open                (std::string const& name) const;

private:

    OGRSFDriver*   _driver;

};

} // namespace fern
