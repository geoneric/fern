module ranally.operations.local.policies.nodata;



class IgnoreNoData
{
  static bool isNoData(
         size_t /* index */)
  {
    return false;
  }
}

