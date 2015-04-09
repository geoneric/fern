// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <exception>
#include <boost/exception/all.hpp>
#include "fern/core/message_id.h"
#include "fern/core/messages.h"
#include "fern/core/string.h"


namespace fern {

// High level exception classes. The classes in this namespace end up in the
// user's code.

class Exception:
    public virtual std::exception
{

public:

                   Exception           ()=delete;

                   Exception           (MessageId message_id);

                   Exception           (Exception const&)=default;

    Exception&     operator=           (Exception const&)=default;

                   Exception           (Exception&&)=default;

    Exception&     operator=           (Exception&&)=default;

    virtual        ~Exception          ()=default;

    virtual String message             () const;

    static Messages const& messages    ();

private:

    static Messages _messages;

    MessageId      _message_id;

};


namespace detail {

// Low level exception handling code. None of these should end up in the user's
// code. This is all private to fern's code.

// Exception information that is added to the low level exception instances
// used in the core of the code.
// using ExceptionMessageId = boost::error_info<struct tag_message_id, MessageId>;
using ExceptionSourceName = boost::error_info<struct tag_source_name, String>;
using ExceptionMessage = boost::error_info<struct tag_message, String>;
using ExceptionConstruct = boost::error_info<struct tag_expression_kind,
    String>;
using ExceptionStatement = boost::error_info<struct tag_statement, String>;
using ExceptionIdentifier = boost::error_info<struct tag_identifier, String>;
using ExceptionFunction = boost::error_info<struct tag_function_name, String>;
using ExceptionRequiredNrArguments = boost::error_info<
    struct tag_required_nr_arguments, size_t>;
using ExceptionProvidedNrArguments = boost::error_info<
    struct tag_provided_nr_arguments, size_t>;
using ExceptionLineNr = boost::error_info<struct tag_line_nr, long>;
using ExceptionColNr = boost::error_info<struct tag_col_nr, long>;
using ExceptionArgumentId = boost::error_info<struct tag_argument_id, size_t>;
using ExceptionRequiredArgumentTypes = boost::error_info<
    struct tag_required_argument_type, String>;
using ExceptionProvidedArgumentTypes = boost::error_info<
    struct tag_provided_argument_type, String>;

// Low leven exception classes.
struct Exception:
    public virtual std::exception, public virtual boost::exception
{ };

// Exception types.
struct ParseError: public virtual Exception { };
struct UnsupportedLanguageConstruct: public virtual Exception { };

struct ValidateError: public virtual Exception { };
struct UndefinedIdentifier: public virtual ValidateError { };
struct UndefinedOperation: public virtual ValidateError { };
struct WrongNumberOfArguments: public virtual ValidateError { };
struct WrongTypeOfArgument: public virtual ValidateError { };

struct IOError: public virtual Exception { };
struct FileOpenError: public virtual IOError { };

// struct ExecuteError: public virtual Exception { };
// struct RangeError: public virtual Exception { };
// struct DomainError: public virtual Exception { };

} // namespace detail
} // namespace fern
