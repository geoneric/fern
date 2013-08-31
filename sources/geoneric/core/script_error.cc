#include "geoneric/core/script_error.h"


namespace geoneric {

ScriptError::ScriptError(
    MessageId message_id,
    String const& source_name,
    long line_nr,
    long col_nr)

    : Exception(message_id),
      _source_name(source_name),
      _line_nr(line_nr),
      _col_nr(col_nr)

{
    assert(!_source_name.is_empty());
}


String ScriptError::source_name() const
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

} // namespace geoneric
