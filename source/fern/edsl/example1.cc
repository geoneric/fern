#include <boost/proto/proto.hpp>

using boost::proto::_;
using boost::proto::or_;
using boost::proto::plus;
using boost::proto::minus;
using boost::proto::terminal;
using boost::proto::divides;
using boost::proto::multiplies;


struct epa:
    or_<
        terminal<_>,
        plus<epa, epa>,
        minus<epa, epa>,
        divides<epa, epa>,
        multiplies<epa, epa>
    >
{};


int main()
{
    boost::proto::terminal<int>::type x{2};

    // auto u = x * x + x - 3 / ~x;
    // boost::proto::display_expr(u);

    std::cout << std::boolalpha
        << boost::proto::matches<decltype(x/x), epa>::value
        << std::endl;

    return 0;
}
