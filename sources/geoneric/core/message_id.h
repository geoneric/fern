#pragma once


namespace geoneric {

enum class MessageId {

    UNKNOWN_ERROR,

    IO_ERROR,
    DOES_NOT_EXIST,
    CANNOT_BE_READ,
    CANNOT_BE_WRITTEN,
    CANNOT_BE_CREATED,
    DOES_NOT_CONTAIN_FEATURE,
    DOES_NOT_CONTAIN_ATTRIBUTE,
    UNSUPPORTED_VALUE_TYPE,
    NO_SUCH_DRIVER,

    ERROR_PARSING,
    ERROR_PARSING_STATEMENT,

    // UNSUPPORTED_EXPRESSION,
    UNSUPPORTED_LANGUAGE_CONSTRUCT,
    UNDEFINED_IDENTIFIER,
    UNDEFINED_OPERATION,
    WRONG_NUMBER_OF_ARGUMENTS,
    WRONG_TYPE_OF_ARGUMENT,

    ERROR_VALIDATING

};

}
