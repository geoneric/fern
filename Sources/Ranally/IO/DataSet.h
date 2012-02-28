#ifndef INCLUDED_RANALLY_IO_DATASET
#define INCLUDED_RANALLY_IO_DATASET

#include <unicode/unistr.h>
#include <boost/noncopyable.hpp>



namespace ranally {
namespace io {

//! Abstract base class for data sets.
/*!
  A data set is a format specific instance containing information about the
  data set. For example, it may contain/cache a file pointer that is used when
  the data set is used for I/O. A data set is conceptually similar to a file,
  but may consist of multiple files.

  \sa        .
*/
class DataSet:
  private boost::noncopyable
{

  friend class DataSetTest;

public:

  virtual          ~DataSet            ();

  UnicodeString const& name            () const;

  virtual void     copy                (DataSet const& dataSet)=0;

protected:

                   DataSet             (UnicodeString const& name);

private:

  //! Name of data set.
  UnicodeString    _name;

};

} // namespace io
} // namespace ranally

#endif
