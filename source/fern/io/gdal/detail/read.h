// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <gdal_priv.h>
#include "fern/core/io_error.h"
#include "fern/io/gdal/dataset.h"
#include "fern/io/gdal/value_type_traits.h"


namespace fern {
namespace io {
namespace gdal {
namespace detail {

template<
    typename OutputNoDataPolicy,
    typename Destination
>
void               read                (OutputNoDataPolicy&
                                            output_no_data_policy,
                                        DataName const& data_name,
                                        Destination& destination);


template<
    typename OutputNoDataPolicy,
    typename Destination
>
inline void read(
    OutputNoDataPolicy& output_no_data_policy,
    DataName const& data_name,
    Destination& destination)
{
    // Read a raster from the GDAL dataset pointed to by data_name.

    auto dataset = open_dataset(data_name, GA_ReadOnly);

    size_t nr_rows = static_cast<size_t>(dataset->GetRasterYSize());
    size_t nr_cols = static_cast<size_t>(dataset->GetRasterXSize());

    assert(size(destination, 0) == nr_rows);
    assert(size(destination, 1) == nr_cols);

    // TODO Verify the transformation passed in is the same as the one
    //      in the dataset. We don't support subsetting, yet.
    // double transformation[6];
    // /* auto status = */ dataset->GetGeoTransform(transformation);
    // double west{transformation[0]};
    // double north{transformation[3]};
    // double cell_width{transformation[1]};
    // double cell_height{std::abs(transformation[5])};


    auto band = raster_band(dataset, 1);

    using value_type = value_type<Destination>;

    // Verify that the elements in the raster to read have the same value
    // type as the variable passed in.
    GDALDataType gdal_value_type{band->GetRasterDataType()};
    assert(ValueTypeTraits<value_type>::gdal_value_type == gdal_value_type);

    // TODO Exception if not.

    if(band->RasterIO(GF_Read, 0, 0, nr_cols, nr_rows, data(destination),
            nr_cols, nr_rows, ValueTypeTraits<value_type>::gdal_value_type,
            0, 0) != CE_None) {
        // TODO
        assert(false);
    }


    int has_no_data;
    value_type gdal_no_data_value{static_cast<value_type>(
        band->GetNoDataValue(&has_no_data))};

    // If the raster dataset contains a special no-data value,
    // then mark those special values as no-data values.
    if(has_no_data) {
        size_t index_;

        for(size_t i = 0; i < size(destination, 0); ++i) {
            index_ = index(destination, i, 0);

            for(size_t j = 0; j < size(destination, 1); ++j) {
                if(get(destination, index_) == gdal_no_data_value) {
                    output_no_data_policy.mark_as_no_data(index_);
                }

                ++index_;
            }
        }
    }


    // auto mask_flags(band.GetMaskFlags());

    // if(!(mask_flags & GMF_ALL_VALID)) {
    //     assert(!(mask_flags & GMF_ALPHA));
    //     GDALRasterBand* mask_band = band.GetMaskBand();
    //     assert(mask_band->GetRasterDataType() == GDT_Byte);
    //     // The mask band has gdal data type GDT_Byte. A value of zero
    //     // means that the value must be masked.
    //     Array<GDALDataTypeTraits<GDT_Byte>::type, 2> mask(
    //         extents[nr_rows][nr_cols]);

    //     if(mask_band->RasterIO(GF_Read, 0, 0, nr_cols, nr_rows, mask.data(),
    //             nr_cols, nr_rows, GDT_Byte, 0, 0) != CE_None) {
    //         assert(false);
    //     }

    //     auto elements = raster.data();
    //     auto mask_elements = mask.data();

    //     for(size_t i = 0; i < raster.size(); ++i) {
    //         if(mask_elements[i] == 0) {
    //             elements[i] = no_data_value<T>();
    //         }
    //     }
    // }


    // // Set transformation in raster.
    // set_transformation(destination, 0, west);
    // set_transformation(destination, 1, cell_width);
    // set_transformation(destination, 2, north);
    // set_transformation(destination, 3, cell_height);
}

} // namespace detail
} // namespace gdal
} // namespace io
} // namespace fern
