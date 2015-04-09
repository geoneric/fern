// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/core/validate_error.h"


namespace fern {

ValidateError::ValidateError(
    String const& source_name,
    long line_nr,
    long col_nr,
    String const& message)

    : ScriptError(MessageId::ERROR_VALIDATING, source_name, line_nr,
          col_nr),
      _message(message)

{
}


String ValidateError::message() const
{
    String message_;

    message_ = boost::format(Exception::message().encode_in_utf8())
        % source_name().encode_in_utf8()
        % line_nr()
        % col_nr()
        % _message.encode_in_utf8()
        ;

    return message_;
}

} // namespace fern
