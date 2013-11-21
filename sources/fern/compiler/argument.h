#pragma once
#include "fern/core/string.h"


namespace fern {

class Argument
{

public:

                   Argument            (String const& name);

  String const&    name                () const;

private:

  String const     _name;

};

} // namespace fern
