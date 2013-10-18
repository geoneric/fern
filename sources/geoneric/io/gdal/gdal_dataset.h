#pragma once
#include "geoneric/feature/core/attributes.h"
#include "geoneric/io/core/dataset.h"


class GDALDataset;
class GDALDriver;
class GDALRasterBand;

namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class GDALDataset:
    public Dataset
{

public:

                   GDALDataset         (String const& name,
                                        OpenMode open_mode);

                   GDALDataset         (::GDALDriver* driver,
                                        String const& name,
                                        OpenMode open_mode);

                   GDALDataset         (::GDALDataset* dataset,
                                        String const& name,
                                        OpenMode open_mode);

                   GDALDataset         (GDALDataset const&)=delete;

    GDALDataset&   operator=           (GDALDataset const&)=delete;

                   GDALDataset         (GDALDataset&&)=delete;

    GDALDataset&   operator=           (GDALDataset&&)=delete;

                   ~GDALDataset        ();

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

    ::GDALDriver*  _driver;

    ::GDALDataset* _dataset;

    ::GDALDriver*  gdal_driver_for_read(String const& name);

    ::GDALDriver*  gdal_driver_for_update(
                                        String const& name);

    ::GDALDataset* gdal_open_for_read  (String const& name);

    ::GDALDataset* gdal_open_for_update(String const& name);

    template<
        class T>
    std::shared_ptr<Attribute> read_attribute(
                                        GDALRasterBand& band) const;

    template<
        class T>
    void           write_attribute     (FieldAttribute<T> const& field,
                                        String const& name) const;

};

} // namespace geoneric
