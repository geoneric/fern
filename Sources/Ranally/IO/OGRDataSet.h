#ifndef INCLUDED_RANALLY_IO_OGRDATASET
#define INCLUDED_RANALLY_IO_OGRDATASET

#include "Ranally/IO/DataSet.h"



class OGRDataSource;

namespace ranally {
namespace io {

class OGRFeatureLayer;

//! Class representing an OGR data set with feature geometries and attributes.
/*!
  The OGRDataSet class is comparable with the OGRDataSource class as defined
  in OGR's API. An OGRLayer as used in OGR is what we call a Feature in
  Ranally.

  \sa        .
*/
class OGRDataSet:
  public DataSet
{

  friend class OGRDataSetTest;

public:

                   OGRDataSet          (UnicodeString const& name,
                                        OGRDataSource* dataSource);

                   ~OGRDataSet         ();

  size_t           nrFeatures          () const;

  Feature*         feature             (size_t i) const;

  Feature*         feature             (UnicodeString const& name) const;

  void             addFeature          (Feature const& feature);

  void             copy                (DataSet const& dataSet);

  bool             exists              (UnicodeString const& name) const;

  void             remove              (UnicodeString const& name);

private:

  OGRDataSource*   _dataSource;

  void             copy                (Feature const& feature);

  Feature*         feature             (OGRFeatureLayer const& layer) const;

  template<class Feature>
  void             add                 (Feature const& feature);

};

} // namespace io
} // namespace ranally

#endif