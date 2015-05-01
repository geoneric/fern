// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/script_error.h"


namespace fern {

class ValidateError:
    public ScriptError
{
public:

                   ValidateError       (std::string const& source_name,
                                        long line_nr,
                                        long col_nr,
                                        std::string const& message);

                   ValidateError       (ValidateError const&)=default;

    ValidateError& operator=           (ValidateError const&)=default;

                   ValidateError       (ValidateError&&)=default;

    ValidateError& operator=           (ValidateError&&)=default;

                   ~ValidateError      ()=default;

    std::string    message             () const;

private:

    std::string    _message;

};

} // namespace fern
