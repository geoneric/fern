// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/language/command/command.h"


namespace fern {
namespace language {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ImportCommand:
    public Command
{

    friend class ImportCommandTest;

public:

                   ImportCommand       (int argc,
                                        char** argv);

                   ~ImportCommand      ()=default;

                   ImportCommand       (ImportCommand&&)=delete;

    ImportCommand& operator=           (ImportCommand&&)=delete;

                   ImportCommand       (ImportCommand const&)=delete;

    ImportCommand& operator=           (ImportCommand const&)=delete;

    int            execute             () const;

private:

};

} // namespace language
} // namespace fern
