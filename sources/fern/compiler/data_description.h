#pragma once
#include "fern/core/string.h"


namespace fern {

class DataDescription
{

public:

                   DataDescription     (String const& name);

  std::string      name                () const;

  std::string      description         () const;

private:

  String const     _name;

  String           _description;

};

} // namespace fern
