#include "ranally/core/exception.h"


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

} // namespace ranally
