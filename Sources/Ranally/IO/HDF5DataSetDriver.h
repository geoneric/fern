#ifndef INCLUDED_RANALLY_IO_HDF5DATASETDRIVER
#define INCLUDED_RANALLY_IO_HDF5DATASETDRIVER

#include "Ranally/IO/DataSetDriver.h"
#include "Ranally/IO/HDF5DataSet.h"



namespace ranally {
namespace io {

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

  bool             exists              (UnicodeString const& name) const;

  HDF5DataSet*     create              (UnicodeString const& name) const;

  void             remove              (UnicodeString const& name) const;

  HDF5DataSet*     open                (UnicodeString const& name) const;

private:

};

} // namespace io
} // namespace ranally

#endif
