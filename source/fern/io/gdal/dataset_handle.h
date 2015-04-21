// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once


class GDALDataset;


namespace fern {
namespace io {
namespace gdal {

/*!
    @ingroup    fern_io_gdal_group
    @brief      Dataset handle representing a GDAL dataset.

    The handle will close the dataset if it goes out of scope.

    There can be only one handle to a dataset.
*/
class DatasetHandle
{

public:

    explicit       DatasetHandle       (GDALDataset* dataset);

                   DatasetHandle       (DatasetHandle const& other)=delete;

                   DatasetHandle       (DatasetHandle&& other)=default;

                   ~DatasetHandle      ();

    DatasetHandle& operator=           (DatasetHandle const& other)=delete;

    DatasetHandle& operator=           (DatasetHandle&& other)=default;

    GDALDataset*   operator->          ();

private:

    //! GDAL dataset.
    GDALDataset*   _dataset;

};

} // namespace gdal
} // namespace io
} // namespace fern
