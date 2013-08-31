#pragma once
#include "geoneric/io/dataset_driver.h"
#include "geoneric/io/hdf5_dataset.h"


namespace geoneric {

//! Data set driver for HDF5 data sets.
/*!
  This driver performs I/O on HDF5 data sets.

  \sa        HDF5Dataset
*/
class HDF5DatasetDriver:
    public DatasetDriver
{

    friend class HDF5DatasetDriverTest;

public:

                   HDF5DatasetDriver   ();

                   ~HDF5DatasetDriver  ();

    bool           exists              (String const& name) const;

    HDF5Dataset*   create              (String const& name) const;

    void           remove              (String const& name) const;

    HDF5Dataset*   open                (String const& name) const;

private:

};

} // namespace geoneric
