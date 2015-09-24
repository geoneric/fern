// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/core/data_name.h"
#include <regex>
#include "fern/core/string.h"


namespace fern {

/// static UErrorCode status;
// No starting /:
/// // static RegexMatcher matcher("([^:]+)(?::(?!/)(.+)?)?", 0, status);
/// // static RegexMatcher matcher("([^:]+)(?::(.+)?)?", 0, status);
// <database pathname>:<data pathname>
static std::regex regular_expression(R"(([^:]+)(?::(.+)?)?)");


DataName::DataName(
    char const* string)

    : DataName(std::string(string))

{
}


DataName::DataName(
    std::string const& string)
{
    std::string database_pathname, data_pathname;
    std::smatch match;

    if(!std::regex_match(string, match, regular_expression)) {
        assert(false);
        // TODO raise exception.
    }

    assert(match.size() >= 2);
    database_pathname = match[1].str();

    if(match.size() > 2) {
        data_pathname = match[2].str();
    }

    // Remove trailing occurences of path separator.
    // Replace double occurences of path separator by single ones.
    strip_end(data_pathname, "/");

    // Loop, otherwise /// will result in //, instead of /, for example.
    while(contains(data_pathname, "//")) {
        replace(data_pathname, "//", "/");
    }

    if(data_pathname.empty()) {
        data_pathname = "/";
    }
    else if(!starts_with(data_pathname, "/")) {
        data_pathname = "/" + data_pathname;
    }

    _database_pathname = database_pathname;
    _data_pathname = data_pathname;

    assert(starts_with(data_pathname, "/"));
}


/*!
    @brief      Return the pathname of the dataset name.
    @return     Pathname.
    @sa         data_pathname()
*/
Path const& DataName::database_pathname() const
{
    return _database_pathname;
}


/*!
    @brief      Return the pathname of the data in the dataset.
    @return     Pathname.
    @sa         database_pathname()
*/
Path const& DataName::data_pathname() const
{
    return _data_pathname;
}

} // namespace fern
