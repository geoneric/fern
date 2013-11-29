#pragma once
#include "fern/command/command.h"
#include "fern/io/io_client.h"


namespace fern {

class DataName;
class Dataset;
class Path;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class DescribeCommand:
    private IOClient,
    public Command
{

    friend class DescribeCommandTest;

public:

                   DescribeCommand     (int argc,
                                        char** argv);

                   DescribeCommand     (DescribeCommand&&)=delete;

    DescribeCommand& operator=         (DescribeCommand&&)=delete;

                   DescribeCommand     (DescribeCommand const&)=delete;

    DescribeCommand& operator=         (DescribeCommand const&)=delete;

                   ~DescribeCommand    ()=default;

    int            execute             () const;

private:

    void           describe            (DataName const& data_name) const;

    void           describe_feature    (Dataset const& dataset,
                                        Path const& path) const;

    void           describe_attribute  (Dataset const& dataset,
                                        Path const& path) const;

};

} // namespace fern
