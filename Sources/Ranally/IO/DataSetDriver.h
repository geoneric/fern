#ifndef INCLUDED_RANALLY_IO_DATASETDRIVER
#define INCLUDED_RANALLY_IO_DATASETDRIVER

#include <unicode/unistr.h>
#include <boost/noncopyable.hpp>



namespace ranally {
namespace io {

class DataSet;

//! Abstract base class for data set drivers.
/*!
  Data set drivers perform I/O on data sets.

  \sa        .
*/
class DataSetDriver:
  private boost::noncopyable
{

  friend class DataSetDriverTest;

public:

  virtual          ~DataSetDriver      ();

  virtual bool     exists              (UnicodeString const& name) const=0;

  virtual DataSet* create              (UnicodeString const& name) const=0;

  virtual void     remove              (UnicodeString const& name) const=0;

  virtual DataSet*  open               (UnicodeString const& name) const=0;

protected:

                   DataSetDriver       ();

private:

};

} // namespace io
} // namespace ranally

#endif
