#pragma once
#include "geoneric/core/string.h"


namespace geoneric {


//! Class for names to data.
/*!
  A data name can name a feature or an attribute in a dataset.

  The convention is that the name of the dataset and of the data in the dataset
  is separated by a colon: <dataset name>:<data pathname>.

  The dataset name folows the native pathname conventions.

  The data pathname folows the generic pathname conventions as described
  in the Boost.Filesystem documentation.

  The kind of data the data name points to is up to the application.
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

    String const&  dataset_name        () const;

    String const&  data_pathname       () const;

private:

    //! Name of dataset.
    String         _dataset_name;

    //! Name of path in the dataset.
    String         _data_pathname;

};

} // namespace geoneric
