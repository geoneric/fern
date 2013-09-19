#pragma once
#include <boost/filesystem.hpp>
#include "geoneric/core/string.h"


namespace geoneric {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Path:
    private boost::filesystem::path
{

public:

                   Path                ()=default;

                   Path                (boost::filesystem::path const& path);

                   Path                (String const& string);

                   Path                (Path&&)=default;

    Path&          operator=           (Path&&)=default;

                   Path                (Path const&)=default;

    Path&          operator=           (Path const&)=default;

                   ~Path               ()=default;

    bool           operator==          (Path const& path) const;

    Path           stem                () const;

private:

};

} // namespace geoneric
