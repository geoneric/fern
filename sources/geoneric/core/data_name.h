#pragma once
#include "geoneric/core/path.h"
#include "geoneric/core/string.h"


namespace geoneric {


//! Class for names to data.
/*!
  A data name can name a feature or an attribute in a dataset.

  The convention is that the name of the database and of the data in the
  dataset is separated by a colon: <database pathname>:<data pathname>.

  The database name folows the native pathname conventions.

  The data pathname folows the generic pathname conventions as described
  in the Boost.Filesystem documentation. The first name is always the name
  of a feature set. Subsequent names point to data within this feature set.
  Data pathname is an absolute pathname.

  The kind of data the data pathname points to is up to the application.
*/
class DataName
{

public:

                   DataName            (String const& string);

                   DataName            (DataName&&)=default;

    DataName&      operator=           (DataName&&)=default;

                   DataName            (DataName const&)=default;

    DataName&      operator=           (DataName const&)=default;

                   ~DataName           ()=default;

    Path const&    database_pathname   () const;

    Path const&    data_pathname       () const;

private:

    //! Name of dataset.
    Path           _database_pathname;

    //! Name of path in the dataset.
    Path           _data_pathname;

};

} // namespace geoneric
