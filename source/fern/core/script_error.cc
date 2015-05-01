// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/core/script_error.h"
#include <cassert>


namespace fern {

ScriptError::ScriptError(
    MessageId message_id,
    std::string const& source_name,
    long line_nr,
    long col_nr)

    : Exception(message_id),
      _source_name(source_name),
      _line_nr(line_nr),
      _col_nr(col_nr)

{
    assert(!_source_name.empty());
}


std::string ScriptError::source_name() const
{
    return _source_name;
}


long ScriptError::line_nr() const
{
    return _line_nr;
}


long ScriptError::col_nr() const
{
    return _col_nr;
}

} // namespace fern
