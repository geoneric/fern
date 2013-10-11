#include "geoneric/core/messages.h"


namespace geoneric {

Messages::Messages()
{
    // TODO Use this format:  <source name>:<line>:<col>:<message>
    insert(std::make_pair(MessageId::UNKNOWN_ERROR,
        "Sorry, unknown error"));

    insert(std::make_pair(MessageId::IO_ERROR,
        // source: message
        "IO error handling %1%: %2%"));
    insert(std::make_pair(MessageId::DOES_NOT_EXIST,
        "Does not exist"));
    insert(std::make_pair(MessageId::CANNOT_BE_READ,
        "Cannot be read"));
    insert(std::make_pair(MessageId::CANNOT_BE_WRITTEN,
        "Cannot be written"));
    insert(std::make_pair(MessageId::CANNOT_BE_CREATED,
        "Cannot be created"));
    insert(std::make_pair(MessageId::DOES_NOT_CONTAIN_FEATURE,
        // feature
        "Does not contain feature: %1%"));
    insert(std::make_pair(MessageId::DOES_NOT_CONTAIN_ATTRIBUTE,
        // attribute
        "Does not contain attribute: %1%"));
    insert(std::make_pair(MessageId::UNSUPPORTED_VALUE_TYPE,
        // attribute: value type
        "Unsupported value type: %1%: %2%"));
    insert(std::make_pair(MessageId::NO_SUCH_DRIVER,
        // driver
        "No such driver: %1%"));

    insert(std::make_pair(MessageId::ERROR_PARSING,
        // source:line:col: message
        "Error parsing %1%:%2%:%3%: %4%"));
    insert(std::make_pair(MessageId::ERROR_PARSING_STATEMENT,
        // source:line:col: statement: message
        "Error parsing %1%:%2%:%3%:%4%: %5%"));

    // insert(std::make_pair(MessageId::UNSUPPORTED_EXPRESSION,
    //     // expression
    //     "Unsupported expression: %1%"));

    insert(std::make_pair(MessageId::UNSUPPORTED_LANGUAGE_CONSTRUCT,
        // construct
        "Unsupported language construct: %1%"));

    insert(std::make_pair(MessageId::UNDEFINED_IDENTIFIER,
        // identifier
        "Undefined identifier: %1%"));
    insert(std::make_pair(MessageId::UNDEFINED_OPERATION,
        // operation
        "Undefined operation: %1%"));
    insert(std::make_pair(MessageId::WRONG_NUMBER_OF_ARGUMENTS,
        // operation: required_number: provided_number
        "Wrong number of arguments for operation: %1%: "
        "%2% required, but %3% provided"));
    insert(std::make_pair(MessageId::WRONG_TYPE_OF_ARGUMENT,
        // argument: operation: required_type: provided_type
        "Wrong type of argument %1% provided for operation: %2%: "
        "%3% required, but %4% provided"));

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

} // namespace geoneric
