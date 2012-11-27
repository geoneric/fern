#pragma once
#include <exception>
#include <boost/exception/all.hpp>
// #include "ranally/core/message_id.h"
#include "ranally/core/string.h"


namespace ranally {

// Exception information.
// typedef boost::error_info<struct tag_message_id, MessageId> ExceptionMessageId;
typedef boost::error_info<struct tag_filename, String> ExceptionFilename;
typedef boost::error_info<struct tag_filename, String> ExceptionErrorMessage;
typedef boost::error_info<struct tag_expression_kind, String>
    ExceptionExpressionKind;

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

} // namespace ranally
