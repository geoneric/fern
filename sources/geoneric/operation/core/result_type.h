#pragma once
#include "geoneric/core/data_types.h"
#include "geoneric/core/value_types.h"


namespace geoneric {

class ResultType
{

public:

                   ResultType          ()=default;

                   ResultType          (DataTypes const& data_types,
                                        ValueTypes const& value_types);

                   ResultType          (ResultType&&)=default;

    ResultType&    operator=           (ResultType&&)=default;

                   ResultType          (ResultType const&)=default;

    ResultType&    operator=           (ResultType const&)=default;

                   ~ResultType         ()=default;

    DataTypes      data_type           () const;

    ValueTypes     value_type          () const;

    bool           defined             () const;

    bool           fixed               () const;

    bool           is_satisfied_by     (ResultType const& result_type) const;

private:

    DataTypes      _data_types;

    ValueTypes     _value_types;

};


bool               operator==          (ResultType const& lhs,
                                        ResultType const& rhs);

bool               operator!=          (ResultType const& lhs,
                                        ResultType const& rhs);

std::ostream&      operator<<          (std::ostream& stream,
                                        ResultType const& result_type);

} // namespace geoneric
