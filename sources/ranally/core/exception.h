#pragma once
#include <exception>
#include <boost/exception/all.hpp>
#include "ranally/core/message_id.h"
#include "ranally/core/messages.h"
#include "ranally/core/string.h"


namespace ranally {

// High level exception classes. The classes in this namespace end up in the
// user's code.

class Exception:
    public virtual std::exception
{
public:

                   Exception           ();

                   Exception           (MessageId message_id);

                   Exception           (Exception const&)=default;

    Exception&     operator=           (Exception const&)=default;

                   Exception           (Exception&&)=default;

    Exception&     operator=           (Exception&&)=default;

    virtual        ~Exception          () noexcept(true) =default;

    virtual String message             () const;

protected:

    static Messages const& messages    ();

private:

    static Messages _messages;

    MessageId      _message_id;

};


namespace detail {

// Low level exception handling code. None of these should end up in the user's
// code. This is all private to ranally's code.

// Exception information that is added to the low level exception instances
// used in the core of the code.
// typedef boost::error_info<struct tag_message_id, MessageId> ExceptionMessageId;
typedef boost::error_info<struct tag_filename, String> ExceptionFilename;
typedef boost::error_info<struct tag_message, String> ExceptionMessage;
typedef boost::error_info<struct tag_expression_kind, String>
    ExceptionExpressionKind;
typedef boost::error_info<struct tag_statement, String> ExceptionStatement;
typedef boost::error_info<struct tag_line_nr, long> ExceptionLineNr;
typedef boost::error_info<struct tag_col_nr, long> ExceptionColNr;

// Low leven exception classes.
struct Exception:
    public virtual std::exception, public virtual boost::exception
{ };

// Exception types.
struct ParseError: public virtual Exception { };
struct UnsupportedExpressionError: public virtual Exception { };
struct IOError: public virtual Exception { };
struct FileOpenError: public virtual IOError { };
// struct ValidateError: public virtual Exception { };
// struct ExecuteError: public virtual Exception { };
// struct RangeError: public virtual Exception { };
// struct DomainError: public virtual Exception { };

} // namespace detail
} // namespace ranally
