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

class ParseError:
    public ScriptError
{
public:

                   ParseError          (String const& source_name,
                                        long line_nr,
                                        long col_nr,
                                        String const& message);

                   ParseError          (String const& source_name,
                                        long line_nr,
                                        long col_nr,
                                        String statement,
                                        String const& message);

                   ParseError          (ParseError const&)=default;

    ParseError&    operator=           (ParseError const&)=default;

                   ParseError          (ParseError&&)=default;

    ParseError&    operator=           (ParseError&&)=default;

                   ~ParseError         ()=default;

    String         message             () const;

private:

    String         _statement;

    String         _message;

};

} // namespace fern
