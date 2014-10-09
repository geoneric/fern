#ifndef INCLUDED_AGGREGATEOPERATION
#include "AggregateOperation.h"
#define INCLUDED_AGGREGATEOPERATION
#endif

#ifndef INCLUDED_BINARYOPERATION
#include "BinaryOperation.h"
#define INCLUDED_BINARYOPERATION
#endif

#ifndef INCLUDED_UNARYOPERATION
#include "UnaryOperation.h"
#define INCLUDED_UNARYOPERATION
#endif

#ifndef INCLUDED_MATHOPERATIONS
#include "MathOperations.h"
#define INCLUDED_MATHOPERATIONS
#endif



/// Wat requirements die ik nog bedenk:
/// - er is ook een category functions die het mask nodig hebben zoals IsNull en SetNull.
/// - Tevens wil het team voor een category functions een 'ignore no data' optie, voor b.v. min, max, mean, etc. Valt bij ons zo'n beetje in NaryOperation klasse denk ik
//
//
//
//
//
/// - Nary valt denk ik weer in subklassen uit elkaar:
/// - Pairwise computation, Min, Max, Sum
///         1 v/d input pixel blocks can be reused store the pair wise result  and that result pixel block can be returned. Pixel type wise the operation is done as:
///         Result = input[0]
///         For (i=1; I < nr; ++i)
///            Result op= input[i] with op in {min, max, sum}
/// - Collector (ik roep maar wat) style computation, visit all inputs and update a collector object for each cell. Then the collector object can compute a result:
///   Mean, Std, collect sum and sum of squares
///   Variety, Minority, Majority, count classes in collector.
/// 
/// Heb alle functies beschreven in arcpy/Generator/implementationTraits.py



//! Macro to create a class definition for a unary operator.
/*!
  \param     name Name of the operator.
  \param     symbol Symbol of operator to use to calculate the result.

  The resulting class has template arguments for the type of the operands
  and the result type. The default type of the result is the same as the type
  of the operands.

  Then \a name passed in is used to name the class: [name]Operation.

  For some operators there is a unary and a binary variant. As a guideline,
  prepend the names of these operators with the arity. The name than becomes
  [arity][name]Operation, eg: UnaryPlus and BinaryPlus.
*/
#define UNARY_OPERATOR(name, symbol)                                           \
template<typename Argument, typename Result=Argument>                          \
struct name##Operation                                                         \
{                                                                              \
  inline void operator()(                                                      \
         Result& result,                                                       \
         Argument const& argument)                                             \
  {                                                                            \
    result = symbol argument;                                                  \
  }                                                                            \
};

//! Macro to create a class definition for a unary function.
/*!
  \param     name Name of the operator.
  \param     function Name of function to use to calculate the result.

  The resulting class has template arguments for the type of the operands
  and the result type. The default type of the result is the same as the type
  of the operands.

  Then \a name passed in is used to name the class: [name]Operation.

  For some operators there is a unary and a binary variant. As a guideline,
  prepend the names of these operators with the arity. The name than becomes
  [arity][name]Operation, eg: UnaryPlus and BinaryPlus.
*/
#define UNARY_FUNCTION(name, function)                                         \
template<typename Argument, typename Result=Argument>                          \
struct name##Operation                                                         \
{                                                                              \
  inline void operator()(                                                      \
         Result& result,                                                       \
         Argument const& argument)                                             \
  {                                                                            \
    result = function<Argument, Result>(argument);                             \
  }                                                                            \
};

//! Macro to create a class definition for a binary operator.
/*!
  \param     name Name of the operator.
  \param     symbol Symbol of operator to use to calculate the result.

  The resulting class has template arguments for the type of the operands
  and the result type. The default type of the result is the same as the type
  of the operands.

  Then \a name passed in is used to name the class: [name]Operation.
*/
#define BINARY_OPERATOR(name, symbol)                                          \
template<typename Argument, typename Result=Argument>                          \
struct name##Operation                                                         \
{                                                                              \
  inline void operator()(                                                      \
         std::vector<bool>::reference result,                                  \
         Argument const& argument1,                                            \
         Argument const& argument2)                                            \
  {                                                                            \
    result = argument1 symbol argument2;                                       \
  }                                                                            \
                                                                               \
  inline void operator()(                                                      \
         Result& result,                                                       \
         Argument const& argument1,                                            \
         Argument const& argument2)                                            \
  {                                                                            \
    result = argument1 symbol argument2;                                       \
  }                                                                            \
};

#define BINARY_FUNCTION(name, function)                                        \
template<typename Argument, typename Result=Argument>                          \
struct name##Operation                                                         \
{                                                                              \
  inline void operator()(                                                      \
         Result& result,                                                       \
         Argument const& argument1,                                            \
         Argument const& argument2)                                            \
  {                                                                            \
    result = function<Argument, Result>(argument1, argument2);                 \
  }                                                                            \
};

#define AGGREGATOR_FUNCTION(name, function)                                    \
template<typename Argument, typename Result=Argument>                          \
struct name##Operation: public function<Argument, Result>                      \
{                                                                              \
};

//! Macro to create a class definition for an operation.
/*!
  \param     arity Arity of the operation.
  \param     name Name of the operation.

  The resulting class has template arguments for the type of the operands
  and the result type. The default type of the result is the same as the type
  of the operands.

  The \a arity passed in is used to select a base class: [arity]Operation.

  The \a name passed in is used to name the class: [name].
*/
#define CONCRETE_OPERATION(arity, name)                                        \
template<typename Argument, typename Result=Argument,                          \
         class NoDataHandler=IgnoreNoData<Result>,                             \
         class ValueDomainHandler=DontCheckValueDomain<Argument>,              \
         class MaskHandler=DontMask>                                           \
struct name: public arity##Operation<Argument, Result, name##Operation,        \
         NoDataHandler,                                                        \
         ValueDomainHandler,                                                   \
         MaskHandler>                                                          \
{                                                                              \
};



// TODO group operation according to arity and combinations of argument and
// TODO result types. Test an operation from each group. Then complete list.

// TODO Document for each operation the relevant types the template can be
// TODO instantiated by. Enforce this by design/assertions.
UNARY_OPERATOR(UnaryMinus, -)
UNARY_OPERATOR(UnaryPlus, +)

UNARY_FUNCTION(SquareRoot, ::sqrt)
UNARY_FUNCTION(Abs, ::abs)
UNARY_FUNCTION(IsNull, ::isNull)
UNARY_FUNCTION(SetNull, ::setNull)

BINARY_OPERATOR(BinaryPlus, +)
BINARY_OPERATOR(BinaryMinus, -)
BINARY_OPERATOR(Multiply, *)
BINARY_OPERATOR(Divide, /)

BINARY_OPERATOR(GreaterThan, >)
BINARY_OPERATOR(GreaterEqual, >=) // TODO overload this for float types
BINARY_OPERATOR(LessThan, <)
BINARY_OPERATOR(LessEqual, <=)    // TODO overload this for float types
BINARY_OPERATOR(Equals, ==)       // TODO overload this for float types
BINARY_OPERATOR(NotEquals, !=)    // TODO overload this for float types

CONCRETE_OPERATION(Unary, UnaryMinus)
CONCRETE_OPERATION(Unary, UnaryPlus)
CONCRETE_OPERATION(Unary, SquareRoot)
CONCRETE_OPERATION(Unary, Abs)
CONCRETE_OPERATION(Unary, IsNull)
CONCRETE_OPERATION(Unary, SetNull)

CONCRETE_OPERATION(Binary, BinaryPlus)
CONCRETE_OPERATION(Binary, BinaryMinus)
CONCRETE_OPERATION(Binary, Multiply)
CONCRETE_OPERATION(Binary, Divide)
CONCRETE_OPERATION(Binary, GreaterThan)
CONCRETE_OPERATION(Binary, GreaterEqual)
CONCRETE_OPERATION(Binary, LessThan)
CONCRETE_OPERATION(Binary, LessEqual)
CONCRETE_OPERATION(Binary, Equals)
CONCRETE_OPERATION(Binary, NotEquals)

AGGREGATOR_FUNCTION(Max, algorithms::Max)
AGGREGATOR_FUNCTION(Mean, algorithms::Mean)

CONCRETE_OPERATION(Aggregate, Max)
CONCRETE_OPERATION(Aggregate, Mean)

