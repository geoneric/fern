#pragma once
#include "fern/operation/core/operation.h"


namespace fern {

class Read:
    public Operation
{

public:

                   Read                ();

                   ~Read               ()=default;

                   Read                (Read&& other)=delete;

    Read&          operator=           (Read&& other)=delete;

                   Read                (Read const& other)=delete;

    Read&          operator=           (Read const& other)=delete;

    ExpressionType expression_type     (size_t index,
                   std::vector<ExpressionType> const& argument_types) const;

    std::vector<std::shared_ptr<Argument>>
                   execute             (std::vector<std::shared_ptr<Argument>>
                                            const& arguments) const;

private:

};

} // namespace fern
