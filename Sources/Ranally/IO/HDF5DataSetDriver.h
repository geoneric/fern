#pragma once
#include "Ranally/IO/DataSetDriver.h"
#include "Ranally/IO/HDF5DataSet.h"


namespace ranally {

//! Data set driver for HDF5 data sets.
/*!
  This driver performs I/O on HDF5 data sets.

  \sa        HDF5DataSet
*/
class HDF5DataSetDriver:
    public DataSetDriver
{

    friend class HDF5DataSetDriverTest;

public:

                   HDF5DataSetDriver   ();

                   ~HDF5DataSetDriver  ();

    bool           exists              (String const& name) const;

    HDF5DataSet*   create              (String const& name) const;

    void           remove              (String const& name) const;

    HDF5DataSet*   open                (String const& name) const;

private:

};

} // namespace ranally
