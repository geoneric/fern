// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once


namespace fern {

enum class MessageId {

    UNKNOWN_ERROR,

    IO_ERROR,
    DOES_NOT_EXIST,
    CANNOT_BE_READ,
    CANNOT_BE_WRITTEN,
    CANNOT_BE_CREATED,

    // NetCDF
    DOES_NOT_CONFORM_TO_CONVENTION,
    DOES_NOT_CONTAIN_VARIABLE,
    VARIABLE_IS_NOT_A_SCALAR,
    // /NetCDF

    DOES_NOT_CONTAIN_FEATURE,
    DOES_NOT_CONTAIN_ATTRIBUTE,
    DOES_NOT_CONTAIN_DATA,
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
