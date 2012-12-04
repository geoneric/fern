#include "ranally/core/messages.h"


namespace ranally {

Messages::Messages()
{
    // TODO Use this format:  <source name>:<line>:<col>:<message>
    insert(std::make_pair(MessageId::UNKNOWN_ERROR,
        "Sorry, unknown error"));

    insert(std::make_pair(MessageId::IO_ERROR,
        // source: message
        "IO error handling %1%: %2%"));

    insert(std::make_pair(MessageId::ERROR_PARSING,
        // source:line:col: message
        "Error parsing %1%:%2%:%3%: %4%"));
    insert(std::make_pair(MessageId::ERROR_PARSING_STATEMENT,
        // source:line:col:statement: message
        "Error parsing %1%:%2%:%3%:%4%: %5%"));

    insert(std::make_pair(MessageId::UNSUPPORTED_EXPRESSION,
        // expression
        "Unsupported expression: %1%"));

    insert(std::make_pair(MessageId::UNDEFINED_IDENTIFIER,
        // identifier
        "Undefined identifier: %1%"));

    insert(std::make_pair(MessageId::ERROR_VALIDATING,
        // source:line:col: message
        "%1%:%2%:%3%: %4%"));
}


String const& Messages::operator[](
    MessageId message_id) const
{
    const_iterator it = find(message_id);
    // If this fails, the message_id hasn't been added to the collection. See
    // above.
    assert(it != end());

    if(it == end()) {
        // Pick a default error message. Just don't crash.
        it = find(MessageId::UNKNOWN_ERROR);
    }
    assert(it != end());

    return (*it).second;
}

} // namespace ranally
