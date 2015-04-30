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


class GDALDataset;
class GDALDriver;
class GDALRasterBand;

namespace fern {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class GDALDataset:
    public Dataset
{

public:

                   GDALDataset         (std::string const& format,
                                        std::string const& name,
                                        OpenMode open_mode);

                   GDALDataset         (::GDALDriver* driver,
                                        std::string const& name,
                                        OpenMode open_mode);

                   GDALDataset         (::GDALDataset* dataset,
                                        std::string const& name,
                                        OpenMode open_mode);

                   GDALDataset         (GDALDataset const&)=delete;

    GDALDataset&   operator=           (GDALDataset const&)=delete;

                   GDALDataset         (GDALDataset&&)=delete;

    GDALDataset&   operator=           (GDALDataset&&)=delete;

                   ~GDALDataset        ();

    size_t         nr_features         () const;

    std::vector<std::string>
                   feature_names       () const;

    bool           contains_feature    (Path const& path) const;

    bool           contains_attribute  (Path const& path) const;

    ExpressionType expression_type     (Path const& path) const;

    std::shared_ptr<Feature>
                   open_feature        (Path const& path) const;

    std::shared_ptr<Attribute>
                   open_attribute      (Path const& path) const;

    std::shared_ptr<Feature> read_feature(
                                        Path const& path) const;

    std::shared_ptr<Attribute> read_attribute(
                                        Path const& path) const;

    void           write_attribute     (Attribute const& attribute,
                                        Path const& path);

private:

    ::GDALDriver*  _driver;

    ::GDALDataset* _dataset;

    GDALRasterBand*
                   band                (Path const& path) const;

    ValueType      value_type          (GDALRasterBand& band,
                                        Path const& path) const;

    template<
        class T>
    std::shared_ptr<FieldAttribute<T>>
                   open_attribute      (GDALRasterBand& band) const;

    template<
        class T>
    std::shared_ptr<Attribute>
                   read_attribute      (GDALRasterBand& band) const;

    template<
        class T>
    void           write_attribute     (FieldAttribute<T> const& field,
                                        Path const& path);

};

} // namespace fern
