module ranally.operations.local.framework.binaryoperation;

import ranally.operations.local.policies.clipmask;
import ranally.operations.local.policies.nodata;



class BinaryOperation(
         LocalOperation,
         ClipMaskPolicy=IgnoreClipMask,
         NoDataPolicy=IgnoreNoData):
              public LocalOperation
{
  // alias LocalOperation.Argument Argument;
  // alias LocalOperation.Result Result;
  // alias LocalOperation.DomainPolicy DomainPolicy;
  // alias LocalOperation.RangePolicy RangePolicy;

private:

  ClipMaskPolicy   _clipMaskPolicy;

  NoDataPolicy     _noDataPolicy;

public:

  static void opCall(
         ref Result result,
         Argument argument1,
         Argument argument2)
  {
    // if(!_clipMaskPolicy.mask(cast(size_t)0)) {
    //   if(!DomainPolicy.inDomain(argument1, argument2)) {
    //     _noDataPolicy.setNoData(cast(size_t)0);
    //   }
    //   else {
        result = algorithm(argument1, argument2);

    //     if(!RangePolicy.inRange(argument1, argument2, result)) {
    //       _noDataPolicy.setNoData(cast(size_t)0);
    //     }
    //   }
    // }
  }
}

