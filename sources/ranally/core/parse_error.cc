#include "ranally/core/parse_error.h"


namespace ranally {

ParseError::ParseError(
    long line_nr,
    long col_nr,
    String statement,
    String const& message)

    : Exception(MessageId::ERROR_PARSING),
      _line_nr(line_nr),
      _col_nr(col_nr),
      _statement(statement),
      _message(message)

{
}


ParseError::ParseError(
    String const& filename,
    long line_nr,
    long col_nr,
    String statement,
    String const& message)

    : Exception(MessageId::ERROR_PARSING_FILE),
      _filename(filename),
      _line_nr(line_nr),
      _col_nr(col_nr),
      _statement(statement),
      _message(message)

{
}


ParseError::~ParseError() noexcept(true) =default;


String ParseError::message() const
{
    String message_;

    if(_filename.is_empty()) {
        message_ = boost::format(Exception::message().encode_in_utf8())
            % _message.encode_in_utf8()
            % _line_nr
            % _col_nr
            % _statement
            ;
    }
    else {
        message_ = boost::format(Exception::message().encode_in_utf8())
            % _filename.encode_in_utf8()
            % _message.encode_in_utf8()
            % _line_nr
            % _col_nr
            % _statement
            ;
    }

    return message_;
}

} // namespace ranally
