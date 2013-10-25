#include "geoneric/core/data_name.h"
#include <unicode/regex.h>


namespace geoneric {

static UErrorCode status;
// No starting /:
// static RegexMatcher matcher("([^:]+)(?::(?!/)(.+)?)?", 0, status);
static RegexMatcher matcher("([^:]+)(?::(.+)?)?", 0, status);

DataName::DataName(
    String const& string)

    : _database_pathname(),
      _data_pathname()

{
    String database_pathname, data_pathname;

    // http://userguide.icu-project.org/strings/regexp
    assert(!U_FAILURE(status));

    matcher.reset(string);
    if(!matcher.matches(status)) {
        assert(false);
        // TODO raise exception.
    }

    assert(!U_FAILURE(status));

    database_pathname = matcher.group(1, status);
    assert(!U_FAILURE(status));

    if(matcher.groupCount() > 1) {
        data_pathname = matcher.group(2, status);
        assert(!U_FAILURE(status));
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
        data_pathname = "/" + data_pathname;
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

} // namespace geoneric
