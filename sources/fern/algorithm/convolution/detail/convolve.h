#pragma once
#include <cassert>
#include <cmath>
#include <cstddef>
#include <utility>
#include <type_traits>
#include "fern/algorithm/convolution/kernel_traits.h"
#include "fern/core/assert.h"
#include "fern/core/argument_traits.h"
#include "fern/core/base_class.h"


namespace fern {
namespace convolve {
namespace detail {

template<
    class Result>
struct OutOfRangePolicy
{

    FERN_STATIC_ASSERT(std::is_floating_point, Result)

    inline bool within_range(
        Result const& result) const
    {
        return std::isfinite(result);
    }

};


/// namespace dispatch {
/// 
/// // template<
/// //     class SourceImage,
/// //     class Kernel,
/// //     class DestinationImage,
/// //     class NormalizePolicy,
/// //     class OutOfImagePolicy,
/// //     template<class> class OutOfRangePolicy,
/// //     class InputNoDataPolicy,
/// //     class OutputNoDataPolicy>
/// //     // class SourceImageCollectionCategory,
/// //     // class KernelCollectionCategory,
/// //     // class DestinationCollectionCategory
/// // >
/// // class Convolve
/// // {
/// // };
/// 
/// 
/// 
/// 
/// 
/// 
/// 
/// 
/// 
/// 
/// template<class SourceImage, class Kernel, class DestinationImage,
///     class NormalizePolicy,
///     template<class> class OutOfRangePolicy,
///     class InputNoDataPolicy,
///     class OutputNoDataPolicy,
///     class SourceImageCollectionCategory,
///     class KernelCollectionCategory,
///     class DestinationCollectionCategory
/// >
/// class Convolve
/// {
/// };
/// 
/// 
/// template<class SourceImage, class Kernel, class DestinationImage,
///     class NormalizePolicy,
///     template<class> class OutOfRangePolicy,
///     class InputNoDataPolicy,
///     class OutputNoDataPolicy
/// >
/// class Convolve<SourceImage, Kernel, DestinationImage,
///         NormalizePolicy,
///         OutOfRangePolicy,
///         InputNoDataPolicy,
///         OutputNoDataPolicy,
///         array_2d_tag,
///         array_2d_tag,
///         array_2d_tag>:
/// 
///     public OutOfRangePolicy<value_type<DestinationImage>>,
///     public InputNoDataPolicy,
///     public OutputNoDataPolicy
/// 
/// {
/// 
/// public:
/// 
///     Convolve()
///         : InputNoDataPolicy(),
///           OutputNoDataPolicy()
///     {
///     }
/// 
///     Convolve(
///         InputNoDataPolicy&& input_no_data_policy,
///         OutputNoDataPolicy&& output_no_data_policy)
///         : InputNoDataPolicy(std::forward<InputNoDataPolicy>(
///               input_no_data_policy)),
///           OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
///               output_no_data_policy))
///     {
///     }
/// 
///     inline void calculate(
///         SourceImage const& source,
///         Kernel const& kernel,
///         DestinationImage& destination)
///     {
///         assert(size(source) == size(destination));
///         assert(size(kernel) > 0);
///         assert(width(kernel) % 2u == 1u);
///         assert(height(kernel) % 2u == 1u);
///         assert(radius(kernel) > 0u);
/// 
///         // Apply kernel on the various sections of the source image.
///         // We divide the source image in 9 section, which differ with
///         // respect to the number of source cells that fall within the
///         // kernel neighborhood:
///         // - 4 Corners: missing cells on two sides.
///         //     - north-west corner
///         //     - north-east corner
///         //     - south-west corner
///         //     - south-east corner
///         // - 4 Sides, between the corners: missing cells on one side.
///         //     - north side
///         //     - west side
///         //     - east side
///         //     - south side
///         // - Center: no missing cells.
///         // For each section of the source image, we call a seperate
///         // function that will handle the section. This code knows exactly
///         // were the missing cells are located. This save us lots of checks
///         // that aren't necessary. For example, when processing the west
///         // side of the source image, the logic can assume that the north,
///         // east and south side of the kernel are located within the source
///         // image. Only at the west side of the kernel some cells are missing.
/// 
///         // Corners.
///         convolve_north_west_corner(source, kernel, destination);
///         convolve_north_east_corner(source, kernel, destination);
///         convolve_south_west_corner(source, kernel, destination);
///         convolve_south_east_corner(source, kernel, destination);
/// 
///         // Sides.
///         convolve_north_side(source, kernel, destination);
///         convolve_west_side(source, kernel, destination);
///         convolve_east_side(source, kernel, destination);
///         convolve_south_side(source, kernel, destination);
/// 
///         // Inner part.
///         convolve_inner_part(source, kernel, destination);
///     }
/// 
/// private:
/// 
///     //! Convolve \a source with \a kernel and write the result to \a destination.
///     /*!
///       \tparam    NormalizePolicy Policy used to handle the normalization of the
///                  convolution result by the kernel weights.
///       \tparam    SourceImage Type of source image.
///       \tparam    Kernel Type of convolution kernel.
///       \tparam    DestinationImage Type of destination image.
///       \param     source Source image.
///       \param     kernel Convolution kernel.
///       \param     destination Destination image.
///       \param     row_source Index of row to calculate value for.
///       \param     col_source Index of column to calculate value for.
///       \param     first_row_source Index of first row in source image to use.
///       \param     first_col_source Index of first column in source image to use.
///       \param     first_row_kernel Index of first row in kernel to use.
///       \param     first_col_kernel Index of first column in kernel to use.
///       \param     nr_rows_kernel Number of rows in kernel to use.
///       \param     nr_cols_kernel Number of columns in kernel to use.
/// 
///       This function calculates a value for a single cell in \a destination. This
///       cell is positioned at \a row_source, \a col_source. The new value is
///       calculated based on the values in \a source and the \a kernel weights.
/// 
///       When the kernel is positioned on the cell to calculate a value for, some
///       cells in the \a source may fall outside of the kernel. This means that we
///       may not have values for all cells in the kernel. The caller can configure
///       this by passing \a first_row_source, \a first_col_source,
///       \a first_row_kernel, \a first_col_kernel, \a nr_rows_kernel,
///       \a nr_cols_kernel appropriately. These arguments configure a section of
///       the source and kernel for which the \a source has values to use
///       during the calculation.
///     */
///     inline void convolve(
///         SourceImage const& source,
///         Kernel const& kernel,
///         DestinationImage& destination,
///         size_t row_source,
///         size_t col_source,
///         size_t first_row_source,
///         size_t first_col_source,
///         size_t first_row_kernel,
///         size_t first_col_kernel,
///         size_t nr_rows_kernel,
///         size_t nr_cols_kernel)
///     {
///         // std::cout <<
///         //     "--------------------------------------------------\n" <<
///         //     "row_source      : " << row_source       << "\n" <<
///         //     "col_source      : " << col_source       << "\n" <<
///         //     "first_row_source: " << first_row_source << "\n" <<
///         //     "first_col_source: " << first_col_source << "\n" <<
///         //     "first_row_kernel: " << first_row_kernel << "\n" <<
///         //     "first_col_kernel: " << first_col_kernel << "\n" <<
///         //     "nr_rows_kernel  : " << nr_rows_kernel   << "\n" <<
///         //     "nr_cols_kernel  : " << nr_cols_kernel   << "\n"
///         //     ;
/// 
///         // Verify dimensions of source and destination images are equal
///         assert(size(source, 0) == size(destination, 0));
///         assert(size(source, 1) == size(destination, 1));
/// 
///         assert(nr_rows_kernel > 0);
///         assert(nr_cols_kernel > 0);
/// 
///         // Verify kernel section is positioned within source image.
///         assert(first_row_source < size(source, 0));
///         assert(first_col_source < size(source, 1));
///         assert(first_row_source + nr_rows_kernel <= size(source, 0));
///         assert(first_col_source + nr_cols_kernel <= size(source, 1));
/// 
///         // Verify kernel section is positioned within kernel.
///         assert(first_row_kernel < height(kernel));
///         assert(first_col_kernel < width(kernel));
///         assert(first_row_kernel + nr_rows_kernel <= height(kernel));
///         assert(first_col_kernel + nr_cols_kernel <= width(kernel));
/// 
///         // Verify cell to calculate for is positioned within kernel section of
///         // source image.
///         assert(row_source >= first_row_source);
///         assert(row_source < first_row_source + nr_rows_kernel);
///         assert(col_source >= first_col_source);
///         assert(col_source < first_col_source + nr_cols_kernel);
/// 
///         typename ArgumentTraits<SourceImage>::value_type value{0};
///         typename ArgumentTraits<Kernel>::value_type count{0};
///         typename ArgumentTraits<SourceImage>::value_type alt_value{0};
/// 
///         using INDP = InputNoDataPolicy;
///         using ONDP = OutputNoDataPolicy;
///         using OORP = OutOfRangePolicy<value_type<DestinationImage>>;
/// 
///         bool value_seen{false};
/// 
///         if(INDP::is_no_data(row_source, col_source)) {
///             ONDP::mark_as_no_data(row_source, col_source);
///         }
///         else {
///             // Iterate over the kernel, or a piece thereof.
///             for(size_t row = 0; row < nr_rows_kernel; ++row) {
///                 for(size_t col = 0; col < nr_cols_kernel; ++col) {
///                     if(INDP::is_no_data(first_row_source + row,
///                             first_col_source + col)) {
///                         if(UseInCaseOfNoDataPolicy::value(
///                                 static_cast<INDP const&>(*this),
///                                 source, size(source, 0), size(source, 1),
///                                 first_row_source + row, first_col_source + col,
///                                 alt_value)) {
///                             value += alt_value *
///                                 get(kernel, first_row_kernel + row,
///                                     first_col_kernel + col);
///                             count += get(kernel, first_row_kernel + row,
///                                 first_col_kernel + col);
///                             value_seen = true;
///                         }
///                     }
///                     else {
///                         value +=
///                             get(kernel, first_row_kernel + row,
///                                 first_col_kernel + col) *
///                             get(source, first_row_source + row,
///                                 first_col_source + col);
///                         count +=
///                             get(kernel, first_row_kernel + row,
///                                 first_col_kernel + col);
///                         value_seen = true;
///                     }
///                 }
///             }
///         }
/// 
///         // The result can go out of range when the convolution results in an
///         // infinite value. Normalizing such a value makes no sense.
/// 
///         if(!value_seen || !OORP::within_range(value)) {
///             ONDP::mark_as_no_data(row_source, col_source);
///         }
///         else {
///             // get(destination, row_source, col_source) = value / count;
///             get(destination, row_source, col_source) =
///                 NormalizePolicy::normalize(value, count);
///         }
///     }
/// 
/// 
///     inline void convolve_north_west_corner(
///         SourceImage const& source,
///         Kernel const& kernel,
///         DestinationImage& destination)
///     {
///         size_t const radius_{radius(kernel)};
///         size_t const first_row_source{0};
///         size_t const first_col_source{0};
///         size_t first_row_kernel{radius_};
///         size_t first_col_kernel;
///         size_t nr_rows_kernel{radius_ + 1};
///         size_t nr_cols_kernel;
/// 
///         // Loop over all cells that are situated in the north west corner
///         // and for which some cells are outside of the kernel.
///         for(size_t row_source = 0; row_source < radius_; ++row_source) {
/// 
///             first_col_kernel = radius_;
///             nr_cols_kernel = radius_ + 1;
/// 
///             for(size_t col_source = 0; col_source < radius_; ++col_source) {
/// 
///                 convolve(source, kernel, destination,
///                     row_source, col_source,
///                     first_row_source, first_col_source,
///                     first_row_kernel, first_col_kernel,
///                     nr_rows_kernel, nr_cols_kernel);
/// 
///                 --first_col_kernel;
///                 ++nr_cols_kernel;
///             }
/// 
///             --first_row_kernel;
///             ++nr_rows_kernel;
///         }
///     }
/// 
/// 
///     inline void convolve_north_east_corner(
///         SourceImage const& source,
///         Kernel const& kernel,
///         DestinationImage& destination)
///     {
///         size_t const radius_{radius(kernel)};
///         size_t const nr_cols_source{size(source, 1)};
///         size_t const first_row_source{0};
///         size_t first_col_source;
///         size_t first_row_kernel{radius_};
///         size_t const first_col_kernel{0};
///         size_t nr_rows_kernel{radius_ + 1};
///         size_t nr_cols_kernel;
/// 
///         // Loop over all cells that are situated in the north east corner
///         // and for which some cells are outside of the kernel.
///         for(size_t row_source = 0; row_source < radius_; ++row_source) {
/// 
///             first_col_source = nr_cols_source - radius_ - radius_;
///             nr_cols_kernel = radius_ + radius_;
/// 
///             for(size_t col_source = nr_cols_source - radius_;
///                     col_source < nr_cols_source; ++col_source) {
/// 
///                 convolve(source, kernel, destination,
///                     row_source, col_source,
///                     first_row_source, first_col_source,
///                     first_row_kernel, first_col_kernel,
///                     nr_rows_kernel, nr_cols_kernel);
/// 
///                 ++first_col_source;
///                 --nr_cols_kernel;
///             }
/// 
///             --first_row_kernel;
///             ++nr_rows_kernel;
///         }
///     }
/// 
/// 
///     inline void convolve_south_west_corner(
///         SourceImage const& source,
///         Kernel const& kernel,
///         DestinationImage& destination)
///     {
///         size_t const radius_{radius(kernel)};
///         size_t const nr_rows_source{size(source, 0)};
///         size_t first_row_source{nr_rows_source - radius_ - radius_};
///         size_t const first_col_source{0};
///         size_t const first_row_kernel{0};
///         size_t first_col_kernel;
///         size_t nr_rows_kernel{radius_ + radius_};
///         size_t nr_cols_kernel;
/// 
///         // Loop over all cells that are situated in the south west corner
///         // and for which some cells are outside of the kernel.
///         for(size_t row_source = nr_rows_source - radius_;
///                 row_source < nr_rows_source; ++row_source) {
/// 
///             first_col_kernel = radius_;
///             nr_cols_kernel = radius_ + 1;
/// 
///             for(size_t col_source = 0; col_source < radius_; ++col_source) {
/// 
///                 convolve(source, kernel, destination,
///                     row_source, col_source,
///                     first_row_source, first_col_source,
///                     first_row_kernel, first_col_kernel,
///                     nr_rows_kernel, nr_cols_kernel);
/// 
///                 --first_col_kernel;
///                 ++nr_cols_kernel;
///             }
/// 
///             ++first_row_source;
///             --nr_rows_kernel;
///         }
///     }
/// 
/// 
///     inline void convolve_south_east_corner(
///         SourceImage const& source,
///         Kernel const& kernel,
///         DestinationImage& destination)
///     {
///         size_t const radius_{radius(kernel)};
///         size_t const nr_rows_source{size(source, 0)};
///         size_t const nr_cols_source{size(source, 1)};
///         size_t first_row_source{nr_rows_source - radius_ - radius_};
///         size_t first_col_source;
///         size_t const first_row_kernel{0};
///         size_t const first_col_kernel{0};
///         size_t nr_rows_kernel{radius_ + radius_};
///         size_t nr_cols_kernel;
/// 
///         // Loop over all cells that are situated in the south east corner
///         // and for which some cells are outside of the kernel.
///         for(size_t row_source = nr_rows_source - radius_;
///                 row_source < nr_rows_source; ++row_source) {
/// 
///             first_col_source = nr_cols_source - radius_ - radius_;
///             nr_cols_kernel = radius_ + radius_;
/// 
///             for(size_t col_source = nr_cols_source - radius_;
///                     col_source < nr_cols_source; ++col_source) {
/// 
///                 convolve(source, kernel, destination,
///                     row_source, col_source,
///                     first_row_source, first_col_source,
///                     first_row_kernel, first_col_kernel,
///                     nr_rows_kernel, nr_cols_kernel);
/// 
///                 ++first_col_source;
///                 --nr_cols_kernel;
///             }
/// 
///             ++first_row_source;
///             --nr_rows_kernel;
///         }
///     }
/// 
/// 
///     inline void convolve_north_side(
///         SourceImage const& source,
///         Kernel const& kernel,
///         DestinationImage& destination)
///     {
///         size_t const radius_{radius(kernel)};
///         size_t const nr_cols_source{size(source, 1)};
///         size_t const first_row_source{0};
///         size_t first_col_source;
///         size_t first_row_kernel{radius_};
///         size_t const first_col_kernel{0};
///         size_t nr_rows_kernel{radius_ + 1};
///         size_t const nr_cols_kernel{width(kernel)};
/// 
///         // Loop over all cells that are situated in the north side and for
///         // which some cells are outside of the kernel.
///         for(size_t row_source = 0; row_source < radius_; ++row_source) {
/// 
///             first_col_source = 0;
/// 
///             for(size_t col_source = radius_; col_source <
///                     nr_cols_source - radius_; ++col_source) {
/// 
///                 convolve(source, kernel, destination,
///                     row_source, col_source,
///                     first_row_source, first_col_source,
///                     first_row_kernel, first_col_kernel,
///                     nr_rows_kernel, nr_cols_kernel);
/// 
///                 ++first_col_source;
///             }
/// 
///             --first_row_kernel;
///             ++nr_rows_kernel;
///         }
///     }
/// 
/// 
///     inline void convolve_west_side(
///         SourceImage const& source,
///         Kernel const& kernel,
///         DestinationImage& destination)
///     {
///         size_t const radius_{radius(kernel)};
///         size_t const nr_rows_source{size(source, 0)};
///         size_t first_row_source{0};
///         size_t const first_col_source{0};
///         size_t const first_row_kernel{0};
///         size_t first_col_kernel;
///         size_t const nr_rows_kernel{height(kernel)};
///         size_t nr_cols_kernel;
/// 
///         // Loop over all cells that are situated in the west side and for
///         // which some cells are outside of the kernel.
///         for(size_t row_source = radius_; row_source < nr_rows_source - radius_;
///                 ++row_source) {
/// 
///             first_col_kernel = radius_;
///             nr_cols_kernel = radius_ + 1;
/// 
///             for(size_t col_source = 0; col_source < radius_; ++col_source) {
/// 
///                 convolve(source, kernel, destination,
///                     row_source, col_source,
///                     first_row_source, first_col_source,
///                     first_row_kernel, first_col_kernel,
///                     nr_rows_kernel, nr_cols_kernel);
/// 
///                 --first_col_kernel;
///                 ++nr_cols_kernel;
///             }
/// 
///             ++first_row_source;
///         }
///     }
/// 
/// 
///     inline void convolve_east_side(
///         SourceImage const& source,
///         Kernel const& kernel,
///         DestinationImage& destination)
///     {
///         size_t const radius_{radius(kernel)};
///         size_t const nr_rows_source{size(source, 0)};
///         size_t const nr_cols_source{size(source, 1)};
///         size_t first_row_source{0};
///         size_t first_col_source;
///         size_t const first_row_kernel{0};
///         size_t const first_col_kernel{0};
///         size_t const nr_rows_kernel{height(kernel)};
///         size_t nr_cols_kernel;
/// 
///         // Loop over all cells that are situated in the east side and for
///         // which some cells are outside of the kernel.
///         for(size_t row_source = radius_; row_source < nr_rows_source - radius_;
///                 ++row_source) {
/// 
///             first_col_source = nr_cols_source - radius_ - radius_;
///             nr_cols_kernel = radius_ + radius_;
/// 
///             for(size_t col_source = nr_cols_source - radius_;
///                     col_source < nr_cols_source; ++col_source) {
/// 
///                 convolve(source, kernel, destination,
///                     row_source, col_source,
///                     first_row_source, first_col_source,
///                     first_row_kernel, first_col_kernel,
///                     nr_rows_kernel, nr_cols_kernel);
/// 
///                 ++first_col_source;
///                 --nr_cols_kernel;
///             }
/// 
///             ++first_row_source;
///         }
///     }
/// 
/// 
///     inline void convolve_south_side(
///         SourceImage const& source,
///         Kernel const& kernel,
///         DestinationImage& destination)
///     {
///         size_t const radius_{radius(kernel)};
///         size_t const nr_cols_source{size(source, 1)};
///         size_t const nr_rows_source{size(source, 0)};
///         size_t first_row_source{nr_rows_source - radius_ - radius_};
///         size_t first_col_source;
///         size_t const first_row_kernel{0};
///         size_t const first_col_kernel{0};
///         size_t nr_rows_kernel{radius_ + radius_};
///         size_t const nr_cols_kernel{width(kernel)};
/// 
///         // Loop over all cells that are situated in the south side and for
///         // which some cells are outside of the kernel.
///         for(size_t row_source = nr_rows_source - radius_;
///                 row_source < nr_rows_source; ++row_source) {
/// 
///             first_col_source = 0;
/// 
///             for(size_t col_source = radius_; col_source < nr_cols_source -
///                     radius_; ++col_source) {
/// 
///                 convolve(source, kernel, destination,
///                     row_source, col_source,
///                     first_row_source, first_col_source,
///                     first_row_kernel, first_col_kernel,
///                     nr_rows_kernel, nr_cols_kernel);
/// 
///                 ++first_col_source;
///             }
/// 
///             ++first_row_source;
///             --nr_rows_kernel;
///         }
///     }
/// 
/// 
///     inline void convolve_inner_part(
///         SourceImage const& source,
///         Kernel const& kernel,
///         DestinationImage& destination)
///     {
///         size_t const radius_{radius(kernel)};
///         size_t const nr_rows_source{size(source, 0)};
///         size_t const nr_cols_source{size(source, 1)};
///         size_t first_row_source{0};
///         size_t first_col_source;
///         size_t const first_row_kernel{0};
///         size_t const first_col_kernel{0};
///         size_t const nr_rows_kernel{height(kernel)};
///         size_t const nr_cols_kernel{width(kernel)};
/// 
///         // Loop over all cells that are situated in the inner part. The kernel
///         // does not extent outside of the source image.
///         for(size_t row_source = radius_; row_source < nr_rows_source - radius_;
///                 ++row_source) {
/// 
///             first_col_source = 0;
/// 
///             for(size_t col_source = radius_; col_source < nr_cols_source -
///                     radius_; ++col_source) {
/// 
///                 convolve(source, kernel, destination,
///                     row_source, col_source,
///                     first_row_source, first_col_source,
///                     first_row_kernel, first_col_kernel,
///                     nr_rows_kernel, nr_cols_kernel);
/// 
///                 ++first_col_source;
///             }
/// 
///             ++first_row_source;
///         }
///     }
/// };
/// 
/// } // namespace dispatch


/// inline void convolve(
///     SourceImage const& source,
///     Kernel const& kernel,
///     DestinationImage& destination,
///     size_t row_source,
///     size_t col_source,
///     size_t first_row_source,
///     size_t first_col_source,
///     size_t first_row_kernel,
///     size_t first_col_kernel,
///     size_t nr_rows_kernel,
///     size_t nr_cols_kernel)
/// {
///     // std::cout <<
///     //     "--------------------------------------------------\n" <<
///     //     "row_source      : " << row_source       << "\n" <<
///     //     "col_source      : " << col_source       << "\n" <<
///     //     "first_row_source: " << first_row_source << "\n" <<
///     //     "first_col_source: " << first_col_source << "\n" <<
///     //     "first_row_kernel: " << first_row_kernel << "\n" <<
///     //     "first_col_kernel: " << first_col_kernel << "\n" <<
///     //     "nr_rows_kernel  : " << nr_rows_kernel   << "\n" <<
///     //     "nr_cols_kernel  : " << nr_cols_kernel   << "\n"
///     //     ;
/// 
///     // Verify dimensions of source and destination images are equal
///     assert(size(source, 0) == size(destination, 0));
///     assert(size(source, 1) == size(destination, 1));
/// 
///     assert(nr_rows_kernel > 0);
///     assert(nr_cols_kernel > 0);
/// 
///     // Verify kernel section is positioned within source image.
///     assert(first_row_source < size(source, 0));
///     assert(first_col_source < size(source, 1));
///     assert(first_row_source + nr_rows_kernel <= size(source, 0));
///     assert(first_col_source + nr_cols_kernel <= size(source, 1));
/// 
///     // Verify kernel section is positioned within kernel.
///     assert(first_row_kernel < height(kernel));
///     assert(first_col_kernel < width(kernel));
///     assert(first_row_kernel + nr_rows_kernel <= height(kernel));
///     assert(first_col_kernel + nr_cols_kernel <= width(kernel));
/// 
///     // Verify cell to calculate for is positioned within kernel section of
///     // source image.
///     assert(row_source >= first_row_source);
///     assert(row_source < first_row_source + nr_rows_kernel);
///     assert(col_source >= first_col_source);
///     assert(col_source < first_col_source + nr_cols_kernel);
/// 
///     typename ArgumentTraits<SourceImage>::value_type value{0};
///     typename ArgumentTraits<Kernel>::value_type count{0};
///     typename ArgumentTraits<SourceImage>::value_type alt_value{0};
/// 
///     using INDP = InputNoDataPolicy;
///     using ONDP = OutputNoDataPolicy;
///     using OORP = OutOfRangePolicy<value_type<DestinationImage>>;
/// 
///     bool value_seen{false};
/// 
///     if(INDP::is_no_data(row_source, col_source)) {
///         ONDP::mark_as_no_data(row_source, col_source);
///     }
///     else {
///         // Iterate over the kernel, or a piece thereof.
///         for(size_t row = 0; row < nr_rows_kernel; ++row) {
///             for(size_t col = 0; col < nr_cols_kernel; ++col) {
/// 
///                 if(INDP::is_no_data(first_row_source + row,
///                         first_col_source + col)) {
///                     if(UseInCaseOfNoDataPolicy::value(
///                             static_cast<INDP const&>(*this),
///                             source, size(source, 0), size(source, 1),
///                             first_row_source + row, first_col_source + col,
///                             alt_value)) {
///                         value += alt_value *
///                             get(kernel, first_row_kernel + row,
///                                 first_col_kernel + col);
///                         count += get(kernel, first_row_kernel + row,
///                             first_col_kernel + col);
///                         value_seen = true;
///                     }
///                 }
///                 else {
///                     value +=
///                         get(kernel, first_row_kernel + row,
///                             first_col_kernel + col) *
///                         get(source, first_row_source + row,
///                             first_col_source + col);
///                     count +=
///                         get(kernel, first_row_kernel + row,
///                             first_col_kernel + col);
///                     value_seen = true;
///                 }
/// 
///             }
///         }
///     }
/// 
///     // The result can go out of range when the convolution results in an
///     // infinite value. Normalizing such a value makes no sense.
/// 
///     if(!value_seen || !OORP::within_range(value)) {
///         ONDP::mark_as_no_data(row_source, col_source);
///     }
///     else {
///         // get(destination, row_source, col_source) = value / count;
///         get(destination, row_source, col_source) =
///             NormalizePolicy::normalize(value, count);
///     }
/// }



namespace dispatch {

template<
    bool weigh_values
>
struct ConvolveNorthWestCorner
{
};


template<
    bool weigh_values
>
struct ConvolveNorthEastCorner
{
};


template<>
struct ConvolveNorthWestCorner<true>
{

    template<
        class NormalizePolicy,
        class OutOfImagePolicy,
        template<class> class OutOfRangePolicy,
        class InputNoDataPolicy,
        class OutputNoDataPolicy,
        class SourceImage,
        class Kernel,
        class DestinationImage>
    static void apply(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
        SourceImage const& source,
        Kernel const& kernel,
        DestinationImage& destination)
    {
        size_t const radius_{radius(kernel)};
        size_t const first_row_source{0};
        size_t const first_col_source{0};
        size_t first_row_kernel{radius_};
        size_t first_col_kernel;
        size_t nr_rows_kernel{radius_ + 1};
        size_t nr_cols_kernel;
        size_t nr_rows_outside_of_image{radius_};
        size_t nr_cols_outside_of_image;

        value_type<SourceImage> sum_of_values;
        value_type<Kernel> sum_of_weights;
        value_type<SourceImage> out_of_image_value;

        using OORP = OutOfRangePolicy<value_type<DestinationImage>>;
        using NP = NormalizePolicy;

        bool value_seen;

        // Loop over all cells that are situated in the north west corner
        // and for which some cells are outside of the kernel.
        for(size_t row_source = 0; row_source < radius_; ++row_source) {

            first_col_kernel = radius_;
            nr_cols_kernel = radius_ + 1;
            nr_cols_outside_of_image = radius_;

            for(size_t col_source = 0; col_source < radius_; ++col_source) {

// std::cout << "\n";
// std::cout << "src              : " << row_source << " "
//                                    << col_source << std::endl;
/// std::cout << "first_src        : " << first_row_source << " "
///                                    << first_col_source << std::endl;
/// std::cout << "first_krnl       : " << first_row_kernel << " "
///                                    << first_col_kernel << std::endl;
/// std::cout << "nr_rows_krnl     : " << nr_rows_kernel << std::endl;
/// std::cout << "nr_cols_krnl     : " << nr_cols_kernel << std::endl;
/// 
/// std::cout << "nr_rws_out_of_img: " << nr_rows_outside_of_image << std::endl;
/// std::cout << "nr_cls_out_of_img: " << nr_cols_outside_of_image << std::endl;

                sum_of_values = 0;
                sum_of_weights = 0;
                value_seen = false;

                if(input_no_data_policy.is_no_data(row_source, col_source)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    // We are now positioned on a cell within the image.
                    // When we center the kernel on this cell, part
                    // of the kernel falls outside of the kernel. The
                    // OutOfImagePolicy knows how to calculate values
                    // for these cells.  It returns the value to use or a
                    // signal that no such value exists. This happens when
                    // the policy doesn't want to consider out of image
                    // cells, or when it has to base a new value upon only
                    // no-data values.

                    // Handle cells positioned in the kernel, but outside of
                    // the source image.
                    for(size_t out_of_image_kernel_row = 0;
                            out_of_image_kernel_row < size(kernel, 0);
                            ++out_of_image_kernel_row) {
                        for(size_t out_of_image_kernel_col = 0;
                                out_of_image_kernel_col < size(kernel, 1);
                                    ++out_of_image_kernel_col) {
                            if(out_of_image_kernel_row < first_row_kernel ||
                                out_of_image_kernel_col < first_col_kernel) {

                                if(OutOfImagePolicy::value_north_west(
                                        input_no_data_policy,
                                        source,
                                        out_of_image_kernel_row,
                                        out_of_image_kernel_col,
                                        first_row_kernel, first_col_kernel,
                                        nr_rows_kernel, nr_cols_kernel,
                                        first_row_source, first_col_source,
                                        out_of_image_value)) {
                                    sum_of_values += out_of_image_value *
                                        get(kernel,
                                            out_of_image_kernel_row,
                                            out_of_image_kernel_col);
                                    sum_of_weights += get(kernel,
                                        out_of_image_kernel_row,
                                        out_of_image_kernel_col);
                                    value_seen = true;
                                }
                            }
                        }
                    }

                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    for(size_t row = 0; row < nr_rows_kernel; ++row) {
                        for(size_t col = 0; col < nr_cols_kernel; ++col) {
                            if(!input_no_data_policy.is_no_data(
                                    first_row_source + row,
                                    first_col_source + col)) {
                                sum_of_values +=
                                    get(kernel, first_row_kernel + row,
                                        first_col_kernel + col) *
                                    get(source, first_row_source + row,
                                        first_col_source + col);
                                sum_of_weights +=
                                    get(kernel, first_row_kernel + row,
                                        first_col_kernel + col);
                                value_seen = true;
                            }
                        }
                    }
                }


                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    get(destination, row_source, col_source) =
                        NP::normalize(sum_of_values, sum_of_weights);
                }

                --first_col_kernel;
                ++nr_cols_kernel;
                --nr_cols_outside_of_image;
            }

            --first_row_kernel;
            ++nr_rows_kernel;
            --nr_rows_outside_of_image;
        }
    }

};


template<>
struct ConvolveNorthEastCorner<true>
{

    template<
        class NormalizePolicy,
        class OutOfImagePolicy,
        template<class> class OutOfRangePolicy,
        class InputNoDataPolicy,
        class OutputNoDataPolicy,
        class SourceImage,
        class Kernel,
        class DestinationImage>
    static void apply(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy,
        SourceImage const& source,
        Kernel const& kernel,
        DestinationImage& destination)
    {
        size_t const radius_{radius(kernel)};
        size_t const nr_cols_source{size(source, 1)};
        size_t const first_row_source{0};
        size_t first_col_source;
        size_t first_row_kernel{radius_};
        size_t const first_col_kernel{0};
        size_t nr_rows_kernel{radius_ + 1};
        size_t nr_cols_kernel;
        size_t nr_rows_outside_of_image{radius_};
        size_t nr_cols_outside_of_image;

        value_type<SourceImage> sum_of_values;
        value_type<Kernel> sum_of_weights;
        value_type<SourceImage> out_of_image_value;

        using OORP = OutOfRangePolicy<value_type<DestinationImage>>;
        using NP = NormalizePolicy;

        bool value_seen;

        // Loop over all cells that are situated in the north east corner
        // and for which some cells are outside of the kernel.
        for(size_t row_source = 0; row_source < radius_; ++row_source) {

            first_col_source = nr_cols_source - radius_ - radius_;
            nr_cols_kernel = radius_ + radius_;
            nr_cols_outside_of_image = 1;

            for(size_t col_source = nr_cols_source - radius_;
                    col_source < nr_cols_source; ++col_source) {

// std::cout << "\n";
// std::cout << "src              : " << row_source << " "
//                                    << col_source << std::endl;
// std::cout << "first_src        : " << first_row_source << " "
//                                    << first_col_source << std::endl;
// std::cout << "first_krnl       : " << first_row_kernel << " "
//                                    << first_col_kernel << std::endl;
// std::cout << "nr_rows_krnl     : " << nr_rows_kernel << std::endl;
// std::cout << "nr_cols_krnl     : " << nr_cols_kernel << std::endl;
// 
// std::cout << "nr_rws_out_of_img: " << nr_rows_outside_of_image << std::endl;
// std::cout << "nr_cls_out_of_img: " << nr_cols_outside_of_image << std::endl;

                sum_of_values = 0;
                sum_of_weights = 0;
                value_seen = false;

                if(input_no_data_policy.is_no_data(row_source, col_source)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    // Handle cells positioned in the kernel, but outside of
                    // the source image.
                    for(size_t out_of_image_kernel_row = 0;
                            out_of_image_kernel_row < size(kernel, 0);
                            ++out_of_image_kernel_row) {
                        for(size_t out_of_image_kernel_col = 0;
                                out_of_image_kernel_col < size(kernel, 1);
                                    ++out_of_image_kernel_col) {
                            if(out_of_image_kernel_row < first_row_kernel ||
                                out_of_image_kernel_col >= nr_cols_kernel) {

                                if(OutOfImagePolicy::value_north_east(
                                        input_no_data_policy,
                                        source,
                                        out_of_image_kernel_row,
                                        out_of_image_kernel_col,
                                        first_row_kernel, first_col_kernel,
                                        nr_rows_kernel, nr_cols_kernel,
                                        first_row_source, first_col_source,
                                        out_of_image_value)) {
                                    sum_of_values += out_of_image_value *
                                        get(kernel,
                                            out_of_image_kernel_row,
                                            out_of_image_kernel_col);
                                    sum_of_weights += get(kernel,
                                        out_of_image_kernel_row,
                                        out_of_image_kernel_col);
                                    value_seen = true;
                                }
                            }
                        }
                    }

                    // Handle cells positioned in the kernel, and in the source
                    // image.
                    for(size_t row = 0; row < nr_rows_kernel; ++row) {
                        for(size_t col = 0; col < nr_cols_kernel; ++col) {
                            if(!input_no_data_policy.is_no_data(
                                    first_row_source + row,
                                    first_col_source + col)) {
                                sum_of_values +=
                                    get(kernel, first_row_kernel + row,
                                        first_col_kernel + col) *
                                    get(source, first_row_source + row,
                                        first_col_source + col);
                                sum_of_weights +=
                                    get(kernel, first_row_kernel + row,
                                        first_col_kernel + col);
                                value_seen = true;
                            }
                        }
                    }
                }


                // The result can go out of range when the convolution
                // results in an infinite value. Normalizing such a
                // value makes no sense.

                // TODO OutOfRangePolicy must handle integral results too.
                if(!value_seen || !OORP::within_range(sum_of_values)) {
                    output_no_data_policy.mark_as_no_data(row_source,
                        col_source);
                }
                else {
                    get(destination, row_source, col_source) =
                        NP::normalize(sum_of_values, sum_of_weights);
                }

                ++first_col_source;
                --nr_cols_kernel;
                ++nr_cols_outside_of_image;
            }

            --first_row_kernel;
            ++nr_rows_kernel;
            --nr_rows_outside_of_image;
        }
    }

};

} // namespace dispatch





// template<
//     class NormalizePolicy,
//     class OutOfImagePolicy,
//     template<class> class OutOfRangePolicy,
//     class InputNoDataPolicy,
//     class OutputNoDataPolicy,
//     class SourceImage,
//     class Kernel,
//     class DestinationImage
// >
// void convolve_north_west_corner(
//     InputNoDataPolicy&& input_no_data_policy,
//     OutputNoDataPolicy&& output_no_data_policy,
//     SourceImage const& source,
//     Kernel const& kernel,
//     DestinationImage& destination)
// {
//     size_t const radius_{radius(kernel)};
//     size_t const first_row_source{0};
//     size_t const first_col_source{0};
//     size_t first_row_kernel{radius_};
//     size_t first_col_kernel;
//     size_t nr_rows_kernel{radius_ + 1};
//     size_t nr_cols_kernel;
//     size_t nr_rows_outside_of_image{radius_};
//     size_t nr_cols_outside_of_image;
// 
//     value_type<SourceImage> value{0};
//     value_type<SourceImage> out_of_image_value;
// 
//     // Loop over all cells that are situated in the north west corner
//     // and for which some cells are outside of the kernel.
//     for(size_t row_source = 0; row_source < radius_; ++row_source) {
// 
//         first_col_kernel = radius_;
//         nr_cols_kernel = radius_ + 1;
//         nr_cols_outside_of_image = radius_;
// 
//         for(size_t col_source = 0; col_source < radius_; ++col_source) {
// 
//             std::cout << "\n";
//             std::cout << "source        : " << row_source << " " << col_source << std::endl;
//             std::cout << "first_source  : " << first_row_source << " " << first_col_source << std::endl;
//             std::cout << "first_kernel  : " << first_row_kernel << " " << first_col_kernel << std::endl;
//             std::cout << "nr_rows_kernel: " << nr_rows_kernel << std::endl;
//             std::cout << "nr_cols_kernel: " << nr_rows_kernel << std::endl;
// 
//             std::cout << "nr_rows_outside_of_image: " << nr_rows_outside_of_image << std::endl;
//             std::cout << "nr_cols_outside_of_image: " << nr_cols_outside_of_image << std::endl;
// 
// 
//             // We are now positioned on a cell within the image. When we
//             // center the kernel on this cell, part of the kernel falls
//             // outside of the kernel. The OutOfImagePolicy knows how to
//             // calculate values for these cells. We iterate over these
//             // positions and ask the OutOfImagePolicy to come up with values,
//             // if any.
//             // We pass it
//             // - the current position
//             // - the extent of the source image that falls within the kernel
//             // - the distance between the current position and the position
//             //   of the cell outside of the source image
//             // It returns the value to use or a signal that no such value
//             // exists. This happens when the policy doesn't want to consider
//             // out of image cells, or when it has to base a new value upon
//             // no-data values.
// 
// 
//             for(size_t out_of_image_row = 0; out_of_image_row <
//                     nr_rows_outside_of_image; ++out_of_image_row) {
//                 for(size_t out_of_image_col = 0; out_of_image_col <
//                         nr_cols_outside_of_image; ++out_of_image_col) {
//                     std::cout << out_of_image_row << " " << out_of_image_col << std::endl;
// 
//                     if(OutOfImagePolicy::value(
//                             input_no_data_policy,
//                             source,
//                             nr_rows_outside_of_image, nr_cols_outside_of_image,
//                             row_source, col_source,
//                             first_row_source, first_col_source,
//                             first_row_kernel, first_col_kernel,
//                             nr_rows_kernel, nr_cols_kernel,
//                             out_of_image_value)) {
// 
//     // Kernels
//     // - Cells may contain weights.
//     //   - Values are multiplied by the weights.
//     //   - Values are normalized using the NormalizePolicy.
//     // - Cells may be turned on or off.
//     //   - Values are just summed. This saves many multiplications.
//     //   - Values are normalized using the NormalizePolicy.
// 
//                         aggregate_value<KernelTraits<Kernel>::weigh_values>(
//                             out_of_image_value, get(kernel, first_row_kernel + row,
//                                 first_col_kernel + col);
//                         value += out_of_image *
//                             get(kernel,
//                                 first_row_kernel + row,
//                                 first_col_kernel + col);
//                         count += get(kernel,
//                             first_row_kernel + row,
//                             first_col_kernel + col);
//                         value_seen = true;
//                     }
//                 }
//             }
// 
// 
// 
// 
// 
//             // TODO hier verder. Determine values for the out of image cells.
//             //      Use the policy to calculate new values. In case a new
//             //      value is calculated, weigh it (or not! dispatch).
//             //      Then handle the values within the image and calculate a
//             //      result.
// 
//             // Handle part outside of the image.
// 
// 
// 
//             // Handle part within the image.
// 
// 
//             // dispatch::sum(source, kernel, sum_of_values, sum_оf_weights,
//             //     row_source, col_source,
//             //     first_row_source, first_col_source,
//             //     first_row_kernel, first_col_kernel,
//             //     nr_rows_kernel, nr_cols_kernel);
// 
//             --first_col_kernel;
//             ++nr_cols_kernel;
//             --nr_cols_outside_of_image;
//         }
// 
//         --first_row_kernel;
//         ++nr_rows_kernel;
//         --nr_rows_outside_of_image;
//     }
// 
// 
// }


template<
    class NormalizePolicy,
    class OutOfImagePolicy,
    template<class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class SourceImage,
    class Kernel,
    class DestinationImage
>
void convolve(
    InputNoDataPolicy&& input_no_data_policy,
    OutputNoDataPolicy&& output_no_data_policy,
    ExecutionPolicy&& /* execution_policy */,
    SourceImage const& source,
    Kernel const& kernel,
    DestinationImage& destination)
{



    // dispatch::Convolve<
    //     SourceImage,
    //     Kernel,
    //     DestinationImage,
    //     NormalizePolicy,
    //     // OutOfImagePolicy,
    //     OutOfRangePolicy,
    //     InputNoDataPolicy,
    //     OutputNoDataPolicy,
    //     // OutOfImagePolicyTraits<OutOfImagePolicy>::skip,
    //     base_class<argument_category<SourceImage>, array_2d_tag>,
    //     base_class<argument_category<Kernel>, array_2d_tag>,
    //     base_class<argument_category<DestinationImage>, array_2d_tag>
    // > algorithm(
    //     std::forward<InputNoDataPolicy>(input_no_data_policy),
    //     std::forward<OutputNoDataPolicy>(output_no_data_policy));
    // algorithm.calculate(source, kernel, destination);

    // class NormalizePolicy,
    // class OutOfImagePolicy,
    // template<class> class OutOfRangePolicy,
    // class ExecutionPolicy,
    // class InputNoDataPolicy,
    // class OutputNoDataPolicy,
    // class SourceImage,
    // class Kernel,
    // class DestinationImage

    // How to handle out of image values.
    // - Create a masked temp raster with the size of the kernel.
    // - Copy source values, 'invent' out of image values.
    // - Mmm, how to handle no-data.
    // - Maybe just implement all logic in each of these functions and
    //   explicitly handle the out of image values.

    // Corners.
    dispatch::ConvolveNorthWestCorner<KernelTraits<Kernel>::weigh_values>::
        template apply<NormalizePolicy,
            OutOfImagePolicy, OutOfRangePolicy>(
                std::forward<InputNoDataPolicy>(input_no_data_policy),
                std::forward<OutputNoDataPolicy>(output_no_data_policy),
                source, kernel, destination);
    dispatch::ConvolveNorthEastCorner<KernelTraits<Kernel>::weigh_values>::
        template apply<NormalizePolicy,
            OutOfImagePolicy, OutOfRangePolicy>(
                std::forward<InputNoDataPolicy>(input_no_data_policy),
                std::forward<OutputNoDataPolicy>(output_no_data_policy),
                source, kernel, destination);


    // convolve_north_west_corner<NormalizePolicy,
    //     OutOfImagePolicy, OutOfRangePolicy>(input_no_data_policy,
    //         output_no_data_policy, source, kernel, destination);

    /// convolve_north_east_corner(source, kernel, destination);
    /// convolve_south_west_corner(source, kernel, destination);
    /// convolve_south_east_corner(source, kernel, destination);

    /// // Sides.
    /// convolve_north_side(source, kernel, destination);
    /// convolve_west_side(source, kernel, destination);
    /// convolve_east_side(source, kernel, destination);
    /// convolve_south_side(source, kernel, destination);

    /// // Inner part.
    /// convolve_inner_part(source, kernel, destination);





    // OutOfImagePolicy:
    // - All convolutions handle out of image values in some way. They may
    //   be skipped, or they may be 'invented', eg: focal average.
    //   There is probably no reason to split algorithms because of this.

    // UseInCaseOfNoData:
    // - All convolutions handle no-data. They may be skipped, or they may be
    //   'invented', eg: focal average. If these values are not skipped, the
    //   process of inventing new values may be expensive. In that case, a
    //   buffer should be used to put all values in before operating the
    //   kernel. This should be optional. The user may know that the input
    //   doesn't contain no-data (check no-data policy!) or only contains
    //   very few no-data value, in which case buffering may cost more than
    //   it saves.

    // Kernels
    // - Cells may contain weights.
    //   - Values are multiplied by the weights.
    //   - Values are normalized using the NormalizePolicy.
    // - Cells may be turned on or off.
    //   - Values are just summed. This saves many multiplications.
    //   - Values are normalized using the NormalizePolicy.
    //
    // ----------------------------------------------------------------------
    // KernelTraits<Kernel>::weigh_values triggers a split in the algorithms.
    // ----------------------------------------------------------------------

    // NormalizePolicy:
    // - What to do with the aggregated value that fall within the kernel?
    //   - Divide by the sum of weights.
    //   - Don't divide by the sum of weights.

    // ExecutionPolicy:
    // - SequentialExecution
    // - ParallelExecution
    //
    // ----------------------------------------
    // This triggers a split in the algorithms.
    // ----------------------------------------

    // It is important to have the iteration logic in one place.
    // - Add support for handling out of image border.
    // - When calling the main convolve function, it should dispatch, based
    //   on weigh_values.

    // In case the user wants to use a buffer with preprocessed, or no-data
    // replaced values, there must be a way to fill the buffer with values.
    // This is probably the responsibility of the BufferPolicy. Create
    // overloads with and without this policy. The buffer policy must fill
    // the buffer before we pass it to convolve instead of the source image.

    // // The buffer policy uses:
    // class NormalizePolicy,
    // class OutOfImagePolicy,
    // template<class> class OutOfRangePolicy,
    // class InputNoDataPolicy,

    // // The overload with buffer policy, doesn't use:
    // class ExecutionPolicy, // Handled at a higher level.
    // class OutputNoDataPolicy, // The buffer represents input values.
}

} // namespace detail
} // namespace convolve
} // namespace fern
