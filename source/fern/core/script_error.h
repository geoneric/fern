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
