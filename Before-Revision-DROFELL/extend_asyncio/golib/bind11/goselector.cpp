#include <pybind11/pybind11.h>
#include <iostream>
#include <thread>
extern "C"
{
#include "libgo.h" // Go 生成的头文件
}

pybind11::bytes memory_view_to_bytes(pybind11::memoryview mv)
{
    pybind11::buffer buf = mv;
    pybind11::buffer_info info = buf.request();
    // 创建 bytes 对象，传入数据指针和长度
    pybind11::bytes b(reinterpret_cast<const char *>(info.ptr), info.size * info.itemsize);
    return b;
}

class PinnedBuffer
{
public:
    PinnedBuffer(int exp, GoUintptr ptr)
    {
        this->exp = exp;
        this->ptr = ptr;
    }
    PinnedBuffer() {}
    ~PinnedBuffer()
    {
        this->release();
    }
    void release()
    {
        if (this->ptr != 0)
        {
            FreePB(this->exp, this->ptr);
            this->ptr = 0;
        }
    }
    static std::shared_ptr<PinnedBuffer> create(int exp, GoUintptr ptr)
    {
        return std::make_shared<PinnedBuffer>(exp, ptr);
    }

private:
    int exp;
    GoUintptr ptr;
};

class ZMCFrames
{
public:
    ZMCFrames(SelectorTakeZMCFrames_return go_data)
    {
        if (go_data.r1 != 0)
        {
            this->pinned_buffer = PinnedBuffer::create(go_data.r5, go_data.r6);
            for (int i = 0; i < go_data.r1; ++i)
            {
                char *frame = go_data.r0[i];
                int len = go_data.r2[i];
                if (frame && len > 0)
                {
                    this->frames.append(pybind11::memoryview::from_memory(frame, len));
                }
                else
                {
                    this->frames.append(pybind11::memoryview(pybind11::bytes())); // empty frame
                }
            }
            FreeZMCFramesHead(
                go_data.r3, (GoUintptr)(go_data.r0),
                go_data.r4, (GoUintptr)(go_data.r2));
            this->recv_time = double(go_data.r7) / 1E9;
        }
    }

    static std::shared_ptr<ZMCFrames> create(SelectorTakeZMCFrames_return go_data)
    {
        return std::make_shared<ZMCFrames>(go_data);
    }
    std::string String()
    {
        std::string result = "[";
        bool first = true;
        for (auto item : frames)
        {
            if (!first)
            {
                result += ","; // 添加分隔符
            }
            pybind11::memoryview mv_item = item.cast<pybind11::memoryview>();
            result += "b\"";
            result += pybind11::str(memory_view_to_bytes(mv_item));
            result += "\"";
            first = false;
        }
        return result + "]";
    }
    int Len()
    {
        if (this->pinned_buffer == nullptr)
        {
            return 0;
        }
        else
        {
            return this->frames.size();
        }
    }

    pybind11::object get_item(pybind11::handle key)
    {
        if (pybind11::isinstance<pybind11::int_>(key))
        {
            ssize_t index = key.cast<ssize_t>();
            return frames[index];
        }
        else if (pybind11::isinstance<pybind11::slice>(key))
        {
            pybind11::slice slice = key.cast<pybind11::slice>();
            ssize_t start, stop, step, slicelength;
            if (!slice.compute(this->frames.size(), &start, &stop, &step, &slicelength))
                throw pybind11::error_already_set();

            pybind11::list result;
            for (ssize_t i = 0; i < slicelength; ++i)
            {
                result.append(frames[start + i * step]);
            }
            return result;
        }
        else
        {
            throw pybind11::type_error("ZMCFrames indices must be integers or slices");
        }
    }

    pybind11::list get_frames() const
    {
        return frames;
    }

    double get_recv_time()
    {
        return recv_time;
    }

    void release()
    {
        this->pinned_buffer->release();
    }

private:
    std::shared_ptr<PinnedBuffer> pinned_buffer;
    pybind11::list frames;
    double recv_time;
};

struct PyFrameKeepAliveHolder
{
    std::vector<char *> ptrs;
    std::vector<int> lens;
    std::vector<pybind11::buffer> owners;
};

class ConnHandler
{
public:
    ConnHandler(GoUintptr handle)
    {
        this->handle = handle;
    }
    static std::shared_ptr<ConnHandler> create(GoUintptr handle)
    {
        return std::make_shared<ConnHandler>(handle);
    }
    ~ConnHandler()
    {
        this->close();
    }

    // void send(pybind11::list frames)
    // {
    //     int frames_count = frames.size();
    //     pybind11::capsule capsule(
    //         new PyFrameKeepAliveHolder(),
    //         [](void *ptr)
    //         {
    //             delete static_cast<PyFrameKeepAliveHolder *>(ptr);
    //         });

    //     auto holder = static_cast<PyFrameKeepAliveHolder *>(capsule.get_pointer());
    //     auto py_keep_alive_id = this->keep_alive_fn.call(capsule);
    //     int keep_alive_id = py_keep_alive_id.cast<int>();
    //     // auto holder = std::make_shared<PyFrameKeepAliveHolder>();
    //     holder->ptrs.reserve(frames_count);
    //     holder->lens.reserve(frames_count);
    //     holder->owners.reserve(frames_count);

    //     for (auto item : frames)
    //     {
    //         pybind11::bytes buf = pybind11::reinterpret_borrow<pybind11::bytes>(item);
    //         std::string s = pybind11::str(buf);
    //         auto buf_size = pybind11::len(buf);
    //         char *data = PyBytes_AsString(buf.ptr());
    //         holder->owners.push_back(buf);
    //         holder->ptrs.push_back(data);
    //         holder->lens.push_back(static_cast<int>(buf_size));
    //     }
    //     HandlerZMCSend(handle, keep_alive_id, holder->ptrs.data(), frames_count, holder->lens.data());
    // }

    void send(pybind11::list frames, int keep_alive_id)
    {
        int frames_count = frames.size();
        // auto holder = std::make_shared<PyFrameKeepAliveHolder>();
        std::vector<char *> ptrs;
        std::vector<int> lens;
        ptrs.reserve(frames_count);
        lens.reserve(frames_count);

        for (auto item : frames)
        {
            int data_len = 0;
            char *data = nullptr;
            if (pybind11::isinstance<pybind11::bytes>(item))
            {
                pybind11::bytes buf = pybind11::reinterpret_borrow<pybind11::bytes>(item);
                data_len = static_cast<int>(pybind11::len(buf));
                data = PyBytes_AsString(buf.ptr());
            }
            else
            {
                pybind11::memoryview mv = pybind11::reinterpret_borrow<pybind11::memoryview>(item);
                data_len = static_cast<int>(pybind11::len(mv));
                data = (char *)PyMemoryView_GET_BUFFER(mv.ptr())->buf;
            }
            ptrs.push_back(data);
            lens.push_back(data_len);
        }
        HandlerZMCSend(handle, keep_alive_id, ptrs.data(), frames_count, lens.data());
    }

    void close()
    {
        if (this->closed)
        {
            return;
        }
        this->closed = true;
        DestoryHandler(this->handle);
        this->handle = 0;
    }
    GoUintptr get_handle() const
    {
        return handle;
    }

private:
    bool closed;
    GoUintptr handle;
};

class ConnectResult
{
public:
    ConnectResult(SelectorTakeConnect_return ret)
    {
        if (ret.r1 != 0)
        {
            this->err_str = std::string(ret.r1);
            std::free(ret.r1);
            throw std::runtime_error(err_str);
        }
        else
        {
            this->ptr = ret.r0;
        }
    }
    std::shared_ptr<ConnHandler> make_handler()
    {
        return ConnHandler::create(this->ptr);
    }
    GoUintptr get_handler_ptr() const
    {
        return ptr;
    }

private:
    GoUintptr ptr;
    std::string err_str;
};

// extern "C" pybind11::capsule make_capsule_from_pyfunc(pybind11::function f, bool with_gil);
extern "C" uintptr_t register_closure(std::function<void()> fn);
extern "C" uintptr_t register_poll_closure(std::function<bool()> fn);

class Selector
{
public:
    Selector(bool busy_wait)
    {
        handle = CreateSelector(busy_wait);
        if (handle == 0)
        {
            throw std::runtime_error("CreateSelector failed");
        }
    }

    static std::shared_ptr<Selector> create(bool busy_wait)
    {
        return std::make_shared<Selector>(busy_wait);
    }

    ~Selector()
    {
        if (handle != 0)
        {
            DestorySelector(handle);
            handle = 0;
        }
    }

    int select(double waitUntil)
    {
        pybind11::gil_scoped_release release;
        return SelectorSelect(handle, waitUntil);
    }

    void interrupt()
    {
        SelectorInterrupt(handle);
    }

    int ready_reciver_id()
    {
        return SelectorReadyReciverID(handle);
    }

    void take_none()
    {
        return SelectorTakeNone(handle);
    }

    bool take_false()
    {
        return SelectorTakeFalse(handle);
    }

    int take_int()
    {
        return SelectorTakeInt(handle);
    }

    int take_int64()
    {
        return SelectorTakeInt64(handle);
    }

    std::string take_str()
    {
        char *cstr = SelectorTakeString(handle);
        if (!cstr)
        {
            throw std::runtime_error("Void String");
        }
        std::string result(cstr);
        std::free(cstr);
        return result;
    }

    pybind11::bytes take_bytes()
    {
        SelectorTakeBytes_return ret = SelectorTakeBytes(handle);
        if (!ret.r0 || ret.r1 <= 0)
            return pybind11::bytes();
        pybind11::bytes result(ret.r0, ret.r1);
        std::free(ret.r0);
        return result;
    }

    pybind11::list take_frames()
    {
        SelectorTakeFrames_return ret = SelectorTakeFrames(handle);
        pybind11::list out;
        for (int i = 0; i < ret.r1; ++i)
        {
            char *frame = ret.r0[i];
            int len = ret.r2[i];
            if (frame && len > 0)
            {
                out.append(pybind11::bytes(frame, len));
                std::free(frame);
            }
            else
            {
                out.append(pybind11::bytes()); // empty frame
            }
        }
        std::free(ret.r2);
        return out;
    }

    void debug_event_after(double period, int reciver_id, char *event)
    {
        SelectorDebugEventAfter(handle, period, reciver_id, event);
    }

    void debug_callback_after(double period, int reciver_id, char *event)
    {
        SelectorDebugEventAfter(handle, period, reciver_id, event);
    }

    void tcp_connect(int reciver_id, char *address)
    {
        SelectorTCPConnect(handle, reciver_id, address);
    }

    void tcp_server(int reciver_id, char *address)
    {
        SelectorTCPServe(handle, reciver_id, address);
    }

    ConnectResult take_connect(int reciver_id)
    {
        auto ret = SelectorTakeConnect(handle, reciver_id);
        return ConnectResult(ret);
    }

    std::shared_ptr<ConnHandler> pipe_reader(size_t read_fd, int recvier_id)
    {
        GoUintptr ptr = SelectorPipeReader(handle, read_fd, recvier_id);
        return ConnHandler::create(ptr);
    }

    std::shared_ptr<ConnHandler> pipe_writer(size_t write_fd)
    {
        GoUintptr ptr = SelectorPipeWriter(handle, write_fd);
        return ConnHandler::create(ptr);
    }

    std::shared_ptr<ZMCFrames> take_zmc_frames()
    {
        SelectorTakeZMCFrames_return ret = SelectorTakeZMCFrames(handle);
        return ZMCFrames::create(ret);
    }

    void delegate_c_blocking_closure(pybind11::capsule blocking_fn, int recvier_id)
    {
        if (std::string(blocking_fn.name()) != "closure_fn")
            throw std::runtime_error("invalid capsule name");
        auto fn_ptr = static_cast<std::function<void()> *>(blocking_fn.get_pointer());
        auto id = register_closure(*fn_ptr);
        SelectorDelagateCBlockingEnclosure(handle, id, recvier_id);
    }

    void delegate_c_poll_closure(pybind11::capsule poll_fn, int recvier_id)
    {
        if (std::string(poll_fn.name()) != "closure_fn")
            throw std::runtime_error("invalid capsule name");
        auto fn_ptr = static_cast<std::function<bool()> *>(poll_fn.get_pointer());
        auto id = register_poll_closure(*fn_ptr);
        SelectorDelagatePollEnclosure(handle, id, recvier_id);
    }

    // pybind11::capsule delegate_py_blocking_closure(pybind11::function pyfunc, int recvier_id, bool with_gil)
    // {
    //     pybind11::capsule cap = make_capsule_from_pyfunc(pyfunc, with_gil);
    //     auto fn_ptr = static_cast<std::function<void()> *>(cap.get_pointer());
    //     auto id = register_closure(*fn_ptr);
    //     SelectorDelagateCBlockingEnclosure(handle, id, recvier_id);
    //     return cap;
    // }

    GoUintptr get_handle() const
    {
        return handle;
    }

private:
    GoUintptr handle;
};

// pybind11::capsule make_debug_cpp_blocking_fn()
// {
//     int x = 123;
//     auto fn = new std::function<void()>([x]()
//                                         {
//         printf("running cpp sleep 5 func: x = %d\n", x);
//         std::this_thread::sleep_for(std::chrono::seconds(5));
//         printf("cpp sleep 5 func: x = %d complete!\n", x);; });

//     return pybind11::capsule(
//         fn,
//         "closure_fn",
//         [](void *p)
//         {
//             delete static_cast<std::function<void()> *>(p);
//         });
// }

// void test_run_cpp_blocking_closure(pybind11::capsule cap)
// {
//     if (std::string(cap.name()) != "closure_fn")
//         throw std::runtime_error("invalid capsule name");
//     auto fn_ptr = static_cast<std::function<void()> *>(cap.get_pointer());
//     auto id = register_closure(*fn_ptr);
//     CallRun(id);
// }

// pybind11::capsule test_run_py_blocking_closure(pybind11::function pyfunc, bool with_gil)
// {
//     pybind11::capsule cap = make_capsule_from_pyfunc(pyfunc, with_gil);
//     auto fn_ptr = static_cast<std::function<void()> *>(cap.get_pointer());
//     auto id = register_closure(*fn_ptr);
//     CallRun(id);
//     return cap;
// }

void work_load_simulate(double seconds, char *name)
{
    pybind11::gil_scoped_release release;
    WorkLoadSimulate(seconds, name);
}

void config_debug_pin_buffer_alloc_limit(int limit)
{
    ConfigDebugPinBufferAllocLimit(limit);
}

void config_debug_print_pin_buffer_create(int print)
{
    ConfigDebugPrintPinBufferCreate(print);
}

void config_debug_print_pin_buffer_borrow_give_back(int print)
{
    ConfigDebugPrintPinBufferBorrowGiveBack(print);
}

void config_debug_print_wait_time(int print)
{
    ConfigDebugPrintWaitTime(print);
}

void config_high_resolution_timer(int enable)
{
    ConfigHighResolutionTimer(enable);
}

void config_selector_chan_size(int size)
{
    ConfigSelectorChanSize(size);
}

void config_conn_handler_send_chan_size(int size)
{
    ConfigConnHandlerSendChanSize(size);
}

void config_async_zmc_send(int enable)
{
    ConfigAsyncZMCSend(enable);
}

PYBIND11_MODULE(goselector, m)
{
    m.def("work_load_simulate", &work_load_simulate);
    m.def("config_debug_pin_buffer_alloc_limit", &config_debug_pin_buffer_alloc_limit);
    m.def("config_debug_print_pin_buffer_create", &config_debug_print_pin_buffer_create);
    m.def("config_debug_print_pin_buffer_borrow_give_back", &config_debug_print_pin_buffer_borrow_give_back);
    m.def("config_debug_print_wait_time", &config_debug_print_wait_time);
    m.def("config_high_resolution_timer", &config_high_resolution_timer);
    m.def("config_selector_chan_size", &config_selector_chan_size);
    m.def("config_conn_handler_send_chan_size", &config_conn_handler_send_chan_size);
    m.def("config_async_zmc_send", &config_async_zmc_send);
    // m.def("test_run_cpp_blocking_closure", &test_run_cpp_blocking_closure);
    // m.def("test_run_py_blocking_closure", &test_run_py_blocking_closure);
    // m.def("test_get_blocking_func", &test_get_blocking_func);
    pybind11::class_<Selector, std::shared_ptr<Selector>>(m, "Selector")
        .def(pybind11::init<>(&Selector::create))
        .def("select", &Selector::select, pybind11::arg("waitUntil"))
        .def("interrupt", &Selector::interrupt)
        .def("ready_reciver_id", &Selector::ready_reciver_id)
        .def("take_none", &Selector::take_none)
        .def("take_false", &Selector::take_false)
        .def("take_int", &Selector::take_int)
        .def("take_int64", &Selector::take_int64)
        .def("take_str", &Selector::take_str)
        .def("take_bytes", &Selector::take_bytes)
        .def("take_frames", &Selector::take_frames)
        .def("take_zmc_frames", &Selector::take_zmc_frames)
        .def("debug_event_after", &Selector::debug_event_after)
        .def("debug_callback_after", &Selector::debug_callback_after)
        .def("tcp_connect", &Selector::tcp_connect)
        .def("tcp_server", &Selector::tcp_server)
        .def("pipe_reader", &Selector::pipe_reader)
        .def("pipe_writer", &Selector::pipe_writer)
        .def("take_connect", &Selector::take_connect)
        .def("delegate_c_blocking_closure", &Selector::delegate_c_blocking_closure)
        .def("delegate_c_poll_closure", &Selector::delegate_c_poll_closure)
        // .def("delegate_py_blocking_closure", &Selector::delegate_py_blocking_closure)
        .def_property_readonly("handle", &Selector::get_handle, "Raw GoUintptr handle");
    ;
    pybind11::class_<PinnedBuffer, std::shared_ptr<PinnedBuffer>>(m, "PinnedBuffer")
        .def(pybind11::init<>(&PinnedBuffer::create));
    pybind11::class_<ConnHandler, std::shared_ptr<ConnHandler>>(m, "ConnHandler")
        .def(pybind11::init<>(&ConnHandler::create))
        .def("send", &ConnHandler::send)
        .def("close", &ConnHandler::close);
    pybind11::class_<PyFrameKeepAliveHolder>(m, "PyFrameKeepAliveHolder");
    pybind11::class_<ConnectResult>(m, "ConnectResult")
        .def("make_handler", &ConnectResult::make_handler)
        .def_property_readonly("handler_ptr", &ConnectResult::get_handler_ptr, "Handler Ptr");
    pybind11::class_<ZMCFrames, std::shared_ptr<ZMCFrames>>(m, "ZMCFrames")
        .def("release", &ZMCFrames::release)
        .def("__str__", &ZMCFrames::String)
        .def("__len__", &ZMCFrames::Len)
        .def("__getitem__", &ZMCFrames::get_item)
        .def_property_readonly("recv_time", &ZMCFrames::get_recv_time, "Recv Time (in Golang)")
        .def_property_readonly("frames", &ZMCFrames::get_frames, "Frames Memory View");
}
