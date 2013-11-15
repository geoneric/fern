#pragma once
#include "fern/operation/core/operation.h"


namespace fern {

class Abs:
    public Operation
{

public:

                   Abs                 ();

                   ~Abs                ()=default;

                   Abs                 (Abs&& other)=delete;

    Abs&           operator=           (Abs&& other)=delete;

                   Abs                 (Abs const& other)=delete;

    Abs&           operator=           (Abs const& other)=delete;

    ExpressionType expression_type     (size_t index,
                                        std::vector<ExpressionType> const&
                                            argument_types) const;

    std::vector<std::shared_ptr<Argument>>
                   execute             (std::vector<std::shared_ptr<Argument>>
                                            const& arguments) const;

private:

};

} // namespace fern
