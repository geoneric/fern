#pragma once
#include "ranally/core/exception.h"


namespace ranally {

class ParseError:
    public Exception
{
public:

                   ParseError          (long line_nr,
                                        long col_nr,
                                        String statement,
                                        String const& message);

                   ParseError          (String const& filename,
                                        long line_nr,
                                        long col_nr,
                                        String statement,
                                        String const& message);

                   ParseError          (ParseError const&)=default;

    ParseError&    operator=           (ParseError const&)=default;

                   ParseError          (ParseError&&)=default;

    ParseError&    operator=           (ParseError&&)=default;

                   ~ParseError         () noexcept(true);

    String         message             () const;

private:

    String         _filename;

    long           _line_nr;

    long           _col_nr;

    String         _statement;

    String         _message;

};

} // namespace ranally
