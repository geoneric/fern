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


class ParseError:
    public Exception
{
public:

                   ParseError          (String const& message);

                   ParseError          (String const& filename,
                                        String const& message);

                   ParseError          (ParseError const&)=default;

    ParseError&    operator=           (ParseError const&)=default;

                   ParseError          (ParseError&&)=default;

    ParseError&    operator=           (ParseError&&)=default;

                   ~ParseError         () noexcept(true);

    String         message             () const;

private:

    String         _filename;

    String         _message;

};


namespace detail {

// Low level exception handling code. None of these should end up in the user's
// code. This is all private to ranally's code.

// Exception information that is added to the low level exception instances
// used in the core of the code.
// typedef boost::error_info<struct tag_message_id, MessageId> ExceptionMessageId;
typedef boost::error_info<struct tag_filename, String> ExceptionFilename;
typedef boost::error_info<struct tag_filename, String> ExceptionMessage;
typedef boost::error_info<struct tag_expression_kind, String>
    ExceptionExpressionKind;

// Low leven exception classes.
struct Exception:
    public virtual std::exception, public virtual boost::exception
{ };

// Exception types.
struct ParseError: public virtual Exception { };
struct UnsupportedExpressionError: public virtual Exception { };
struct FileOpenError: public virtual Exception { };
// struct ValidateError: public virtual Exception { };
// struct ExecuteError: public virtual Exception { };
// struct RangeError: public virtual Exception { };
// struct DomainError: public virtual Exception { };

} // namespace detail
} // namespace ranally
