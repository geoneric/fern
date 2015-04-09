// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/exception.h"


namespace fern {

class ScriptError:
    public Exception
{
public:

                   ScriptError         (ScriptError const&)=default;

    ScriptError&   operator=           (ScriptError const&)=default;

                   ScriptError         (ScriptError&&)=default;

    ScriptError&   operator=           (ScriptError&&)=default;

                   ~ScriptError        ()=default;

protected:

                   ScriptError         (MessageId message_id,
                                        String const& source_name,
                                        long line_nr,
                                        long col_nr);

    String         source_name         () const;

    long           line_nr             () const;

    long           col_nr              () const;

private:

    String         _source_name;

    long           _line_nr;

    long           _col_nr;

};

} // namespace fern
