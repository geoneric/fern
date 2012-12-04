#pragma once
#include "ranally/core/script_error.h"


namespace ranally {

class ValidateError:
    public ScriptError
{
public:

                   ValidateError       (String const& source_name,
                                        long line_nr,
                                        long col_nr,
                                        String const& message);

                   ValidateError       (ValidateError const&)=default;

    ValidateError& operator=           (ValidateError const&)=default;

                   ValidateError       (ValidateError&&)=default;

    ValidateError& operator=           (ValidateError&&)=default;

                   ~ValidateError      () noexcept(true);

    String         message             () const;

private:

    String         _message;

};

} // namespace ranally
