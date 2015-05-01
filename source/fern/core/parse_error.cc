// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/core/parse_error.h"
#include <cassert>


namespace fern {

ParseError::ParseError(
    std::string const& source_name,
    long line_nr,
    long col_nr,
    std::string const& message)

    : ScriptError(MessageId::ERROR_PARSING, source_name, line_nr, col_nr),
      _message(message)

{
}


ParseError::ParseError(
    std::string const& source_name,
    long line_nr,
    long col_nr,
    std::string statement,
    std::string const& message)

    : ScriptError(MessageId::ERROR_PARSING_STATEMENT, source_name, line_nr,
        col_nr),
      _statement(statement),
      _message(message)

{
    assert(!_statement.empty());
}


std::string ParseError::message() const
{
    std::string message_;

    if(_statement.empty()) {
        message_ = (boost::format(Exception::message())
            % source_name()
            % line_nr()
            % col_nr()
            % _message
            ).str();
    }
    else {
        message_ = (boost::format(Exception::message())
            % source_name()
            % line_nr()
            % col_nr()
            % _statement
            % _message
            ).str();
    }

    return message_;
}

} // namespace fern
