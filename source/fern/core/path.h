// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <vector>
#include <boost/filesystem.hpp>
#include "fern/core/string.h"


namespace fern {

//! Class for representing paths.
/*!
  A path can be a path to a file in the filesystem or to an attribute in a
  database, for example.

  \sa        DataName
*/
class Path:
    private boost::filesystem::path
{

public:

                   Path                ()=default;

                   Path                (String const& string);

                   Path                (char const* string);

                   Path                (Path&&)=default;

    Path&          operator=           (Path&&)=default;

                   Path                (Path const&)=default;

    Path&          operator=           (Path const&)=default;

    // Path&          operator=           (String const& string);

                   ~Path               ()=default;

    bool           operator==          (Path const& path) const;

    Path&          operator/=          (Path const& path);

    String         generic_string      () const;

    String         native_string       () const;

    bool           is_empty            () const;

    bool           is_absolute         () const;

    Path           stem                () const;

    Path           parent_path         () const;

    Path           filename            () const;

    std::vector<String> names          () const;

    Path&          replace_extension   (Path const& extension);

private:

};


Path               operator+           (Path const& lhs,
                                        Path const& rhs);

Path               operator/           (Path const& lhs,
                                        Path const& rhs);

std::ostream&      operator<<          (std::ostream& stream,
                                        Path const& path);

} // namespace fern
