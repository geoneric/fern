#ifndef INCLUDED_RANALLY_IO_HDF5DATASET
#define INCLUDED_RANALLY_IO_HDF5DATASET

#include <boost/scoped_ptr.hpp>
#include "Ranally/IO/DataSet.h"


namespace H5 {
  class H5File;
} // namespace H5

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

                   HDF5DataSet         (UnicodeString const& name,
                                        H5::H5File* file);

                   ~HDF5DataSet        ();

private:

  boost::scoped_ptr<H5::H5File> _file;

};

} // namespace io
} // namespace ranally

#endif
