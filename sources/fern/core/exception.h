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
// typedef boost::error_info<struct tag_message_id, MessageId> ExceptionMessageId;
typedef boost::error_info<struct tag_source_name, String> ExceptionSourceName;
typedef boost::error_info<struct tag_message, String> ExceptionMessage;
typedef boost::error_info<struct tag_expression_kind, String>
    ExceptionConstruct;
typedef boost::error_info<struct tag_statement, String> ExceptionStatement;
typedef boost::error_info<struct tag_identifier, String> ExceptionIdentifier;
typedef boost::error_info<struct tag_function_name, String>
    ExceptionFunction;
typedef boost::error_info<struct tag_required_nr_arguments, size_t>
    ExceptionRequiredNrArguments;
typedef boost::error_info<struct tag_provided_nr_arguments, size_t>
    ExceptionProvidedNrArguments;
typedef boost::error_info<struct tag_line_nr, long> ExceptionLineNr;
typedef boost::error_info<struct tag_col_nr, long> ExceptionColNr;
typedef boost::error_info<struct tag_argument_id, size_t> ExceptionArgumentId;
typedef boost::error_info<struct tag_required_argument_type, String>
    ExceptionRequiredArgumentTypes;
typedef boost::error_info<struct tag_provided_argument_type, String>
    ExceptionProvidedArgumentTypes;

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
