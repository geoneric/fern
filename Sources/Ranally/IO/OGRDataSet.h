#ifndef INCLUDED_RANALLY_IO_OGRDATASET
#define INCLUDED_RANALLY_IO_OGRDATASET

#include "Ranally/IO/DataSet.h"



class OGRDataSource;

namespace ranally {
namespace io {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class OGRDataSet:
  public DataSet
{

  friend class OGRDataSetTest;

public:

                   OGRDataSet          (UnicodeString const& name,
                                        OGRDataSource* dataSource);

                   ~OGRDataSet         ();

  void             copy                (DataSet const& dataSet);

private:

  OGRDataSource*   _dataSource;

};

} // namespace io
} // namespace ranally

#endif
