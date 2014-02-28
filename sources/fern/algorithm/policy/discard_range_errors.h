#pragma once


namespace fern {

// template<
//     class A1,
//     class A2>
class DiscardRangeErrors {

public:

    template<
        class A1,
        class A2,
        class R>
    static bool    within_range        (A1 const& argument1,
                                        A2 const& argument2,
                                        R const& result);

protected:

                   DiscardRangeErrors  ()=default;

                   DiscardRangeErrors  (DiscardRangeErrors&&)=delete;

    DiscardRangeErrors&
                   operator=           (DiscardRangeErrors&&)=delete;

                   DiscardRangeErrors  (DiscardRangeErrors const&)=delete;

    DiscardRangeErrors&
                   operator=           (DiscardRangeErrors const&)=delete;

                   ~DiscardRangeErrors ()=default;

private:

};


// template<
//     class A1,
//     class A2>
template<
    class A1,
    class A2,
    class R>
inline bool DiscardRangeErrors/* <A1, A2> */::within_range(
    A1 const& /* argument1 */,
    A2 const& /* argument2 */,
    R const& /* result */)
{
    return true;
}

} // namespace fern
