#ifndef INCLUDED_RANALLY_IO_HDF5DATASET
#define INCLUDED_RANALLY_IO_HDF5DATASET

#include "Ranally/IO/DataSet.h"



namespace ranally {
namespace io {

//! Class for HDF5 data sets.
/*!
  HDF5 data sets represent data sets stored using the HDF5 library.

  \sa        HDF5DataSetDriver
*/
class HDF5DataSet:
  public DataSet
{

  friend class HDF5DataSetTest;

public:

                   HDF5DataSet         (UnicodeString const& name);

                   ~HDF5DataSet        ();

private:

};

} // namespace io
} // namespace ranally

#endif
