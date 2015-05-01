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
    std::string const& source_name,
    long line_nr,
    long col_nr,
    std::string const& message)

    : ScriptError(MessageId::ERROR_VALIDATING, source_name, line_nr,
          col_nr),
      _message(message)

{
}


std::string ValidateError::message() const
{
    std::string message_;

    message_ = (boost::format(Exception::message())
        % source_name()
        % line_nr()
        % col_nr()
        % _message
        ).str();

    return message_;
}

} // namespace fern
