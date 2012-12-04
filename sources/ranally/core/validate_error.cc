#include "ranally/core/validate_error.h"


namespace ranally {

ValidateError::ValidateError(
    String const& source_name,
    long line_nr,
    long col_nr,
    String const& message)

    : ScriptError(MessageId::ERROR_VALIDATING_FILE, source_name, line_nr,
          col_nr),
      _message(message)

{
}


ValidateError::~ValidateError() noexcept(true) =default;


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

} // namespace ranally
