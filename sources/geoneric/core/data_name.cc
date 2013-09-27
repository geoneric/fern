#include "geoneric/core/data_name.h"
#include <unicode/regex.h>


namespace geoneric {

static UErrorCode status;
static RegexMatcher matcher("([^:]+)(?::(.+)?)?", 0, status);

DataName::DataName(
    String const& string)

    : _dataset_name(),
      _data_pathname()

{
    // http://userguide.icu-project.org/strings/regexp
    assert(!U_FAILURE(status));

    matcher.reset(string);
    if(!matcher.matches(status)) {
        assert(false);
        // TODO raise exception.
    }

    assert(!U_FAILURE(status));

    _dataset_name = matcher.group(1, status);
    assert(!U_FAILURE(status));

    if(matcher.groupCount() > 1) {
        _data_pathname = matcher.group(2, status);
        assert(!U_FAILURE(status));
    }

    if(_data_pathname.is_empty()) {
        _data_pathname = "/";
    }
}


String const& DataName::dataset_name() const
{
    return _dataset_name;
}


String const& DataName::data_pathname() const
{
    return _data_pathname;
}

} // namespace geoneric
