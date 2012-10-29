#pragma once
#include "Ranally/Util/String.h"


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
class DataSet
{

    friend class DataSetTest;

public:

                   DataSet             (DataSet const&)=delete;

    DataSet&       operator=           (DataSet const&)=delete;

    virtual        ~DataSet            ();

    String const&  name                () const;

    //! Return the number of features available in the data set.
    /*!
      \return    Number of features.
    */
    virtual size_t nrFeatures          () const=0;

    //! Return feature with id \a i.
    /*!
      \return    Feature.
    */
    virtual Feature* feature           (size_t i) const=0;

    //! Return feature with name \a name.
    /*!
      \return    Feature.
    */
    virtual Feature* feature           (String const& name) const=0;

    //! Add \a feature to the data set.
    /*!
      \param     feature Feature to add to the data set.
    */
    virtual void   addFeature          (Feature const& feature)=0;

    //! Copy all features from \a dataSet.
    /*!
      \param     dataSet Data set to copy.
    */
    virtual void   copy                (DataSet const& dataSet)=0;

    //! Return whether feature \a name is present.
    /*!
      \param     name Name of feature.
      \return    true or false
    */
    virtual bool   exists              (String const& name) const=0;

    //! Remove feature \a name.
    /*!
      \param     name Name of feature to remove.
      \warning   It is assumed that feature \a name is present in the data set.
      \sa        exists(String const&)
    */
    virtual void   remove              (String const& name)=0;

protected:

                   DataSet             (String const& name);

private:

    //! Name of data set.
    String         _name;

    //! Copy \a feature.
    /*!
      \param     feature Feature to copy.
    */
    virtual void   copy                (Feature const& feature)=0;

};

} // namespace io
} // namespace ranally
