#pragma once
#include "geoneric/io/core/dataset.h"


namespace H5 {
    class H5File;
}


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class GeonericDataset:
    public Dataset
{

public:

                   GeonericDataset     (String const& name,
                                        OpenMode open_mode);

                   GeonericDataset     (std::shared_ptr<H5::H5File> const& file,
                                        String const& name,
                                        OpenMode open_mode);

                   GeonericDataset     (GeonericDataset const&)=delete;

    GeonericDataset& operator=         (GeonericDataset const&)=delete;

                   GeonericDataset     (GeonericDataset&&)=delete;

    GeonericDataset& operator=         (GeonericDataset&&)=delete;

                   ~GeonericDataset    ();

    size_t         nr_features         () const;

    bool           contains_feature    (String const& name) const;

    bool           contains_attribute  (String const& name) const;

    std::shared_ptr<Feature> read_feature(
                                        String const& name) const;

    std::shared_ptr<Attribute> read_attribute(
                                        String const& name) const;

    void           write_attribute     (Attribute const& attribute,
                                        String const& name) const;

private:

    std::shared_ptr<H5::H5File> _file;

};

} // namespace geoneric
