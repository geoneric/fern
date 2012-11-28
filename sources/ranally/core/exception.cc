#include "ranally/core/exception.h"
// #include "ranally/core/messages.h"


namespace ranally {

Messages Exception::_messages;

Messages const& Exception::messages()
{
    return _messages;
}


Exception::Exception()

    : _message_id(MessageId::UNKNOWN_ERROR)

{
}


Exception::Exception(
    MessageId message_id)

    : _message_id(message_id)

{
}


String Exception::message() const
{
    return _messages[_message_id];
}


ParseError::ParseError(
    String const& message)

    : Exception(MessageId::ERROR_PARSING),
      _message(message)

{
}


ParseError::ParseError(
    String const& filename,
    String const& message)

    : Exception(MessageId::ERROR_PARSING_FILE),
      _filename(filename),
      _message(message)

{
}


ParseError::~ParseError() noexcept(true) =default;


String ParseError::message() const
{
    String message_;

    if(_filename.is_empty()) {
        message_ = boost::format(Exception::message().encode_in_utf8())
            % _message.encode_in_utf8();
    }
    else {
        message_ = boost::format(Exception::message().encode_in_utf8())
            % _filename.encode_in_utf8()
            % _message.encode_in_utf8();
    }

    return message_;
}

} // namespace ranally
