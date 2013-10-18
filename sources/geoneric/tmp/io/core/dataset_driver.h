#pragma once
#include "geoneric/core/string.h"


namespace geoneric {

class Dataset;

//! Abstract base class for data set drivers.
/*!
  Data set drivers perform I/O on data sets.

  \sa        .
*/
class DatasetDriver
{

    friend class DatasetDriverTest;

public:

                   DatasetDriver       (DatasetDriver const&)=delete;

    DatasetDriver& operator=           (DatasetDriver const&)=delete;

    virtual        ~DatasetDriver      ();

    virtual bool   exists              (String const& name) const=0;

    //! Create data set \a name.
    /*!
      \param     name Name of data set to create.
      \return    Pointer to new Dataset instance.
      \exception .
      \warning   Data set \a name will be truncated if it already exists.
    */
    virtual Dataset* create            (String const& name) const=0;

    virtual void   remove              (String const& name) const=0;

    virtual Dataset* open              (String const& name) const=0;

protected:

                   DatasetDriver       ();

private:

};

} // namespace geoneric
