// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once


namespace fern {
namespace io {
namespace netcdf {

/*!
    @ingroup    fern_io_netcdf_group
    @brief      Dataset handle representing a NetCDF dataset.

    The handle will close the dataset if it goes out of scope.

    There can be only one handle to a dataset.
*/
class DatasetHandle
{

public:

    explicit       DatasetHandle       (int ncid);

                   DatasetHandle       (DatasetHandle const& other)=delete;

                   DatasetHandle       (DatasetHandle&& other)=default;

                   ~DatasetHandle      ();

    DatasetHandle& operator=           (DatasetHandle const& other)=delete;

    DatasetHandle& operator=           (DatasetHandle&& other)=default;

    int            ncid                () const;

private:

    //! NetCDF dataset id.
    int            _ncid;

};

} // namespace netcdf
} // namespace io
} // namespace fern
