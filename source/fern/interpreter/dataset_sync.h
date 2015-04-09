// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/data_name.h"
#include "fern/interpreter/data_sync.h"


namespace fern {

class Dataset;


//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class DatasetSync:
    public DataSync
{

public:

                   DatasetSync         (DataName const& data_name);

                   DatasetSync         (std::shared_ptr<Dataset> const& dataset,
                                        Path const& path);

                   DatasetSync         (DatasetSync const&)=delete;

    DatasetSync&   operator=           (DatasetSync const&)=delete;

                   DatasetSync         (DatasetSync&&)=delete;

    DatasetSync&   operator=           (DatasetSync&&)=delete;

                   ~DatasetSync        ()=default;

    void           write               (Argument const& argument);

    std::shared_ptr<Dataset> const& dataset() const;

private:

    Path const     _data_path;

    std::shared_ptr<Dataset> _dataset;

};

} // namespace fern
