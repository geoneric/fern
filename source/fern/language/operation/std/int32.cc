// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/operation/std/int32.h"
#include <cassert>
#include "fern/language/operation/core/attribute_argument.h"
#include "fern/language/operation/core/expression_type_calculation.h"


namespace fern {
namespace language {

Int32::Int32()

    : Operation("int32",
          "Cast the argument to int32 and return the result.",
          {
              Parameter("Input argument",
                  "Argument to cast.",
                  DataTypes::CONSTANT | DataTypes::STATIC_FIELD,
                  ValueTypes::NUMBER)
          },
          {
              Result("Result value",
                  "Input argument as an int32.",
                  ExpressionType(DataTypes::CONSTANT | DataTypes::STATIC_FIELD,
                      ValueTypes::INT32))
          }
      )

{
}


ExpressionType Int32::expression_type(
    size_t index,
    std::vector<ExpressionType> const& argument_types) const
{
    return standard_expression_type(*this, index, argument_types);
}


std::vector<std::shared_ptr<Argument>> Int32::execute(
#ifndef NDEBUG
    std::vector<std::shared_ptr<Argument>> const& arguments
#else
    std::vector<std::shared_ptr<Argument>> const& /* arguments */
#endif
    ) const
{
    assert(arguments.size() == 1);
    assert(arguments[0]->argument_type() == ArgumentType::AT_ATTRIBUTE);

    assert(false);

    return std::vector<std::shared_ptr<Argument>>();
}

} // namespace language
} // namespace fern
