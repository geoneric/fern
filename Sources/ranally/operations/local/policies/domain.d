/**
  Module with implementation of some general domain policies.

  License: Use freely for any purpose.
*/

module ranally.operations.local.policies.domain;



/**
  Dummy domain policy.

  This policy accepts all argument values.

  Authors: Kor de Jong, kor@jemig.eu
*/
class DummyDomainPolicy(T)
{
  /**
    No-op, always returns true.

    This function assumes the algorithm accepts every value.

    Authors: Kor de Jong, kor@jemig.eu
    Returns: true
  */
  static bool inDomain(
       T /* argument1 */,
       T /* argument2 */)
  {
    return true;
  }
}

