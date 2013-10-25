#pragma once
#include "geoneric/feature/core/attributes.h"
#include "geoneric/io/core/dataset.h"


namespace H5 {
    class DataSet;
    class Group;
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

    size_t         nr_features         (Path const& path) const;

    size_t         nr_attributes       (Path const& path) const;

    bool           contains_feature    (Path const& path) const;

    bool           contains_attribute  (Path const& path) const;

    std::shared_ptr<Feature> read_feature(
                                        Path const& Path) const;

    std::shared_ptr<Attribute> read_attribute(
                                        Path const& Path) const;

    void           write_attribute     (Attribute const& attribute,
                                        Path const& Path);

    template<
        class T>
    void           write_attribute     (ConstantAttribute<T> const& constant,
                                        Path const& Path);

private:

    std::shared_ptr<H5::H5File> _file;

    std::shared_ptr<H5::Group> group   (Path const& path) const;

    bool           contains_feature    (std::vector<String> const& names) const;

    bool           contains_feature_by_name(
                                        String const& pathname) const;

    bool           contains_attribute_by_name(
                                        String const& pathname) const;

    void           add_feature         (Path const& path);

    void           add_feature         (std::vector<String> const& names);

    template<
        class T>
    std::shared_ptr<Attribute> read_numeric_attribute(
                                        Path const& path,
                                        H5::DataSet const& dataset) const;

    template<
        class T>
    std::shared_ptr<Attribute> read_constant_attribute(
                                        Path const& path,
                                        H5::DataSet const& dataset) const;

};

} // namespace geoneric
