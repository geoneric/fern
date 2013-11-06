#pragma once
#include "geoneric/core/data_name.h"
#include "geoneric/interpreter/data_sync.h"


namespace geoneric {

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

private:

    Path const     _data_path;

    std::shared_ptr<Dataset> _dataset;

};

} // namespace geoneric
