#pragma once
#include "fern/operation/core/operation.h"


namespace fern {

class Int32:
    public Operation
{

public:

                   Int32               ();

                   ~Int32              ()=default;

                   Int32               (Int32&& other)=delete;

    Int32&         operator=           (Int32&& other)=delete;

                   Int32               (Int32 const& other)=delete;

    Int32&         operator=           (Int32 const& other)=delete;

    ExpressionType expression_type     (size_t index,
                                        std::vector<ExpressionType> const&
                                            argument_types) const;

    std::vector<std::shared_ptr<Argument>>
                   execute             (std::vector<std::shared_ptr<Argument>>
                                            const& arguments) const;

private:

};

} // namespace fern
