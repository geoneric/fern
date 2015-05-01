// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/path.h"


namespace fern {

/*!
    @brief      Class for names to data.
    @sa         Path

    A data name can name a feature or an attribute in a dataset.

    The convention is that the name of the database and of the data in the
    dataset is separated by a colon: \<database pathname\>:\<data pathname\>.

    The database name folows the native pathname conventions.

    The data pathname folows the generic pathname conventions as described
    in the Boost.Filesystem documentation. The first name is always the name
    of a feature set. Subsequent names point to data within this feature set.
    Data pathname is an absolute pathname.

    The kind of data the data pathname points to is up to the application.

    In the future, this class can be extended to support URI's to databases.
*/
class DataName
{

public:

                   DataName            (char const* string);

                   DataName            (std::string const& string);

                   DataName            (DataName&&)=default;

                   DataName            (DataName const&)=default;

                   ~DataName           ()=default;

    DataName&      operator=           (DataName&&)=default;

    DataName&      operator=           (DataName const&)=default;

    Path const&    database_pathname   () const;

    Path const&    data_pathname       () const;

private:

    //! Name of dataset.
    Path           _database_pathname;

    //! Name of path in the dataset.
    Path           _data_pathname;

};

} // namespace fern
