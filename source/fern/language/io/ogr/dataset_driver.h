// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <string>


namespace fern {

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

    virtual bool   exists              (std::string const& name) const=0;

    //! Create data set \a name.
    /*!
      \param     name Name of data set to create.
      \return    Pointer to new Dataset instance.
      \exception .
      \warning   Data set \a name will be truncated if it already exists.
    */
    virtual Dataset* create            (std::string const& name) const=0;

    virtual void   remove              (std::string const& name) const=0;

    virtual Dataset* open              (std::string const& name) const=0;

protected:

                   DatasetDriver       ();

private:

};

} // namespace fern
