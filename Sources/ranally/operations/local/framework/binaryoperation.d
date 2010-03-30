module ra.BinaryOperation;

import ra.ClipMaskPolicies;
import ra.NoDataPolicies;
import ra.Plus;



class BinaryOperation(
         LocalOperation,
         ClipMaskPolicy=IgnoreClipMask,
         NoDataPolicy=IgnoreNoData):
              public LocalOperation
              // public ClipMaskPolicy,
              // public NoDataPolicy
{
  alias LocalOperation.Argument Argument;
  alias LocalOperation.Result Result;
  alias LocalOperation.DomainPolicy DomainPolicy;
  alias LocalOperation.RangePolicy RangePolicy;

  static Result opCall(
         Argument argument1,
         Argument argument2)
  {
    return LocalOperation.algorithm(argument1, argument2);
  }
}

unittest {
  ra.BinaryOperation.BinaryOperation!(ra.Plus.Plus!(int)) operation;
  assert(operation(2, 3) == 5);
}

