#pragma once
#include "geoneric/core/string.h"


namespace geoneric {

class DataName
{

public:

                   DataName            (String const& string);

                   DataName            (DataName&&)=default;

    DataName&      operator=           (DataName&&)=default;

                   DataName            (DataName const&)=default;

    DataName&      operator=           (DataName const&)=default;

                   ~DataName           ()=default;

    String const&  dataset_name        () const;

    String const&  data_pathname       () const;

private:

    String         _dataset_name;

    String         _data_pathname;

};

} // namespace geoneric
