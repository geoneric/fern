#include "fern/operation/std/int32.h"
#include "fern/operation/core/attribute_argument.h"
#include "fern/operation/core/expression_type_calculation.h"


namespace fern {

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
    std::vector<std::shared_ptr<Argument>> const& arguments) const
{
    assert(arguments.size() == 1);
    assert(arguments[0]->argument_type() == ArgumentType::AT_ATTRIBUTE);

    assert(false);
}

} // namespace fern
