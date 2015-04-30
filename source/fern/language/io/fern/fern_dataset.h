// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/feature/core/attributes.h"
#include "fern/language/io/core/dataset.h"


namespace H5 {
    class DataSet;
    class Group;
    class H5File;
}


namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class FernDataset:
    public Dataset
{

public:

                   FernDataset         (std::string const& name,
                                        OpenMode open_mode);

                   FernDataset         (std::shared_ptr<H5::H5File> const& file,
                                        std::string const& name,
                                        OpenMode open_mode);

                   FernDataset         (FernDataset const&)=delete;

    FernDataset&   operator=           (FernDataset const&)=delete;

                   FernDataset         (FernDataset&&)=delete;

    FernDataset&   operator=           (FernDataset&&)=delete;

                   ~FernDataset        ();

    size_t         nr_features         () const;

    size_t         nr_features         (Path const& path) const;

    std::vector<std::string>
                   feature_names       () const;

    size_t         nr_attributes       (Path const& path) const;

    bool           contains_feature    (Path const& path) const;

    bool           contains_attribute  (Path const& path) const;

    std::shared_ptr<Feature>
                   open_feature        (Path const& path) const;

    std::shared_ptr<Attribute>
                   open_attribute      (Path const& path) const;

    ExpressionType expression_type     (Path const& path) const;

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

    H5::DataSet    dataset             (Path const& path) const;

    ValueType      value_type          (H5::DataSet const& dataset) const;

    std::shared_ptr<H5::Group> group   (Path const& path) const;

    bool           contains_feature    (std::vector<std::string> const&
                                            names) const;

    bool           contains_feature_by_name(
                                        std::string const& pathname) const;

    bool           contains_attribute_by_name(
                                        std::string const& pathname) const;

    void           add_feature         (Path const& path);

    void           add_feature         (std::vector<std::string> const& names);

    template<
        class T>
    ExpressionType expression_type_numeric_attribute(
                                        H5::DataSet const& dataset) const;

    template<
        class T>
    std::shared_ptr<Attribute>
                   open_attribute      (H5::DataSet const& dataset) const;

    template<
        class T>
    std::shared_ptr<Attribute>
                   read_constant_attribute(
                                        H5::DataSet const& dataset) const;

    template<
        class T>
    std::shared_ptr<Attribute>
                   read_attribute      (H5::DataSet const& dataset) const;

};

} // namespace fern
