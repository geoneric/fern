// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/language/operation/std/add.h"
#include <cassert>
#include "fern/language/operation/core/attribute_argument.h"
#include "fern/language/operation/core/expression_type_calculation.h"


namespace fern {
namespace language {

Add::Add()

    : Operation("add",
          "Add the two argument values and return the result.",
          {
              Parameter("First input argument",
                  "First argument to add.",
                  DataTypes::CONSTANT | DataTypes::STATIC_FIELD,
                  ValueTypes::NUMBER),
              Parameter("Second input argument",
                  "Second argument to add.",
                  DataTypes::CONSTANT | DataTypes::STATIC_FIELD,
                  ValueTypes::NUMBER)
          },
          {
              Result("Result value",
                  "Input arguments, added together.",
                  ExpressionType(DataTypes::CONSTANT | DataTypes::STATIC_FIELD,
                      ValueTypes::NUMBER))
          }
      )

{
}


ExpressionType Add::expression_type(
    size_t index,
    std::vector<ExpressionType> const& argument_types) const
{
    return standard_expression_type(*this, index, argument_types);
}


std::vector<std::shared_ptr<Argument>> Add::execute(
    std::vector<std::shared_ptr<Argument>> const& arguments) const
{
    assert(arguments.size() == 2);
    assert(arguments[0]->argument_type() == ArgumentType::AT_ATTRIBUTE);
    assert(arguments[1]->argument_type() == ArgumentType::AT_ATTRIBUTE);

    AttributeArgument const& attribute_argument(
        *std::dynamic_pointer_cast<AttributeArgument>(arguments[0]));
    assert(attribute_argument.data_type() == DataType::DT_CONSTANT);
    assert(attribute_argument.value_type() == ValueType::VT_INT64);

    return std::vector<std::shared_ptr<Argument>>();
}

} // namespace language
} // namespace fern
