#include <pybind11/pybind11.h>
#include <iostream>
#include <thread>
#include <sstream>
#include <memory>

class ExampleCBlocking
{
public:
    ExampleCBlocking(std::string tag, int count)
        : tag_(std::move(tag)), count_(count) {}

    static std::shared_ptr<ExampleCBlocking> create(const std::string &tag, int count)
    {
        return std::make_shared<ExampleCBlocking>(tag, count);
    }

    ~ExampleCBlocking()
    {
        std::cout << "~ExampleCBlocking" << std::endl;
    }

    void run_blocking_delegate()
    {
        std::cout << "running cpp example load sleep " << count_
                  << " func: tag = " << tag_ << std::endl;
        for (int i = 0; i < count_; ++i)
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        std::cout << "cpp example load sleep " << count_
                  << " func: tag = " << tag_ << " complete!" << std::endl;
        std::ostringstream oss;
        oss << "RESULT=[tag=" << tag_ << ",count=" << count_ << "]";
        result_ = oss.str();
    }

    pybind11::capsule get_blocking_delegate()
    {
        auto closure = new std::function<void()>([this]
                                                 {
            std::cout << "run closure_fn" << std::endl;
            run_blocking_delegate(); });

        return pybind11::capsule(
            closure,
            "closure_fn",
            [](void *p)
            {
                std::cout << "delete closure_fn" << std::endl;
                delete static_cast<std::function<void()> *>(p);
            });
    }

    std::string retrieve_result() const
    {
        return result_;
    }

private:
    std::string tag_;
    int count_;
    std::string result_;
};

PYBIND11_MODULE(simple_load, m)
{
    pybind11::class_<ExampleCBlocking, std::shared_ptr<ExampleCBlocking>>(m, "ExampleCBlocking")
        .def(pybind11::init<>(&ExampleCBlocking::create))
        .def("run_blocking_delegate", &ExampleCBlocking::run_blocking_delegate)
        .def("get_blocking_delegate", &ExampleCBlocking::get_blocking_delegate)
        .def("retrieve_result", &ExampleCBlocking::retrieve_result);
}