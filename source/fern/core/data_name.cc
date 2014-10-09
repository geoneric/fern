#include "fern/core/data_name.h"
#include <regex>


namespace fern {

/// static UErrorCode status;
// No starting /:
/// // static RegexMatcher matcher("([^:]+)(?::(?!/)(.+)?)?", 0, status);
/// // static RegexMatcher matcher("([^:]+)(?::(.+)?)?", 0, status);
// <database pathname>:<data pathname>
static std::regex regular_expression(R"(([^:]+)(?::(.+)?)?)");



DataName::DataName(
    char const* string)

    : DataName(String(string))

{
}


DataName::DataName(
    String const& string)

    : _database_pathname(),
      _data_pathname()

{
    String database_pathname, data_pathname;
    std::smatch match;

    if(!std::regex_match(static_cast<std::string const&>(string), match,
            regular_expression)) {
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
    data_pathname.strip_end("/");

    // Loop, otherwise /// will result in //, instead of /, for example.
    while(data_pathname.contains("//")) {
        data_pathname.replace("//", "/");
    }

    if(data_pathname.is_empty()) {
        data_pathname = "/";
    }
    else if(!data_pathname.starts_with("/")) {
        data_pathname = String("/") + data_pathname;
    }

    _database_pathname = database_pathname;
    _data_pathname = data_pathname;

    assert(_data_pathname.is_absolute());
}


//! Return the pathname of the dataset name.
/*!
  \return    Pathname.
  \sa        data_pathname()
*/
Path const& DataName::database_pathname() const
{
    return _database_pathname;
}


//! Return the pathname of the data in the dataset.
/*!
  \return    Pathname.
  \sa        database_pathname()
*/
Path const& DataName::data_pathname() const
{
    return _data_pathname;
}

} // namespace fern
