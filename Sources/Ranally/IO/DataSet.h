#ifndef INCLUDED_RANALLY_IO_DATASET
#define INCLUDED_RANALLY_IO_DATASET

#include <unicode/unistr.h>
#include <boost/noncopyable.hpp>



namespace ranally {

class Feature;

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

  //! Return the number of features available in the data set.
  /*!
    \return    Number of features.
  */
  virtual size_t   nrFeatures          () const=0;

  //! Return feature with id \a i.
  /*!
    \return    Feature.
  */
  virtual Feature* feature             (size_t i) const=0;

  //! Add \a feature to the data set.
  /*!
    \param     feature Feature to add to the data set.
  */
  virtual void     addFeature          (Feature const* feature)=0;

  //! Copy all features from \a dataSet.
  /*!
    \param     dataSet Data set to copy.
  */
  virtual void     copy                (DataSet const& dataSet)=0;

protected:

                   DataSet             (UnicodeString const& name);

private:

  //! Name of data set.
  UnicodeString    _name;

  //! Copy \a feature.
  /*!
    \param     feature Feature to copy.
  */
  virtual void     copy                (Feature const& feature)=0;

};

} // namespace io
} // namespace ranally

#endif
