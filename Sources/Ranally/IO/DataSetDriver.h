#pragma once
#include "Ranally/Util/String.h"


namespace ranally {

class DataSet;

//! Abstract base class for data set drivers.
/*!
  Data set drivers perform I/O on data sets.

  \sa        .
*/
class DataSetDriver
{

    friend class DataSetDriverTest;

public:

                   DataSetDriver       (DataSetDriver const&)=delete;

    DataSetDriver& operator=           (DataSetDriver const&)=delete;

    virtual        ~DataSetDriver      ();

    virtual bool   exists              (String const& name) const=0;

    //! Create data set \a name.
    /*!
      \param     name Name of data set to create.
      \return    Pointer to new DataSet instance.
      \exception .
      \warning   Data set \a name will be truncated if it already exists.
    */
    virtual DataSet* create            (String const& name) const=0;

    virtual void   remove              (String const& name) const=0;

    virtual DataSet* open              (String const& name) const=0;

protected:

                   DataSetDriver       ();

private:

};

} // namespace ranally
