/**
  Module with implementation of Plus local function and related policies.

  License: Use freely for any purpose.
*/
module ranally.operations.local.algorithms.binary.plus;

import ranally.operations.local.policies.domain;



/**
  Plus template.

  Authors: Kor de Jong, kor@jemig.eu
  Params:
         T = Value type, must be numeric.
*/
class Plus(T)
{
  alias T Argument;
  alias T Result;

  /// Domain policy.
  alias DummyDomainPolicy!(T) DomainPolicy;

  unittest {
    assert(Plus!(int).DomainPolicy.inDomain( 2,  2));
    assert(Plus!(int).DomainPolicy.inDomain( 0,  0));
    assert(Plus!(int).DomainPolicy.inDomain(-2, -2));

    assert(Plus!(double).DomainPolicy.inDomain( 2.0,  2.0));
    assert(Plus!(double).DomainPolicy.inDomain( 0.0,  0.0));
    assert(Plus!(double).DomainPolicy.inDomain(-2.0, -2.0));
  }

  class RangePolicy
  {
    @safe
    static pure nothrow bool inRange(
         T argument1,
         T argument2,
         T result) nothrow
    {
      assert(__traits(isArithmetic, T));
      assert(__traits(isIntegral, T) || __traits(isFloating, T));

      static if(__traits(isFloating, T)) {
        // Returns true, even if result is infinity.
        return true;
      }
      else {
        static if(__traits(isUnsigned, T)) {
          // Return true if the value is not wrapped.
          return !(result < argument1);
        }
        else {
          if(argument1 < cast(T)0 && argument2 < cast(T)0) {
            return argument1 + argument2 < cast(T)0;
          }
          else if(argument1 > cast(T)0 && argument2 > cast(T)0) {
            return argument1 + argument2 >= cast(T)0;
          }
          else {
            return true;
          }
        }
      }
    }

    unittest {
      assert( Plus!(int).RangePolicy.inRange(int.max,  0, int.max +  0));
      assert(!Plus!(int).RangePolicy.inRange(int.max,  1, int.max +  1));
      assert( Plus!(int).RangePolicy.inRange(int.min,  0, int.min +  0));
      assert(!Plus!(int).RangePolicy.inRange(int.min, -1, int.min + -1));

      assert( Plus!(double).RangePolicy.inRange(double.max,  0.0,
         double.max + 0.0));
      assert( Plus!(double).RangePolicy.inRange(double.max, double.max,
         double.max + double.max));

      assert( Plus!(uint).RangePolicy.inRange(uint.max, cast(uint)0,
         uint.max + cast(uint)0));
         // algorithm(uint.max, cast(uint)0)));
      assert(!Plus!(uint).RangePolicy.inRange(uint.max, cast(uint)1,
         uint.max + cast(uint)1));
         // algorithm(uint.max, cast(uint)1)));
      // assert( Plus!(uint).RangePolicy.inRange(int.max, 0, int.max + 1));
      // assert( Plus!(uint).RangePolicy.inRange(0, 0, 0 + 0));
    }
  }

  /**
    Returns the sum of the two arguments.

    Authors: Kor de Jong, kor@jemig.eu
    Returns: Result of adding argument1 to argument2.
    Params:
         argument1 = First argument.
         argument2 = Second argument.
  */
  @safe
  static pure nothrow T algorithm(
         T argument1,
         T argument2)
  {
    assert(__traits(isArithmetic, T));

    return argument1 + argument2;
  }

  unittest
  {
    assert(Plus!(int).algorithm(2, 2) == 4);
    assert(Plus!(double).algorithm(2.5, 2.5) == 5.0);
  }
}


