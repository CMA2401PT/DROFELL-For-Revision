#include <pybind11/pybind11.h>
#include <iostream>
#include <map>
#include <mutex>
#include <functional>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <chrono>
#include <sstream>
extern "C"
{
#include "libgo.h" // Go 生成的头文件
}
// extern "C" void blocking_func()
// {
//     printf("Hello from my_func!, sleep 3\n");
//     sleep(3);
//     printf("3s slepted");
// }

// void test_get_blocking_func()
// {
//     using run_fn_t = void (*)();
//     run_fn_t fn = &blocking_func;
//     uintptr_t func_ptr = reinterpret_cast<uintptr_t>(fn);
//     CallRun(func_ptr);
// }

// struct ClosureEntry
// {
//     std::function<void()> fn;
//     bool auto_delete;
//     std::string debug_tag;
// };

static std::mutex g_blocking_mu;
static std::map<uintptr_t, std::function<void()>> g_blocking_registry;
static uintptr_t g_blocking_next_id = 1;

static std::mutex g_poll_mu;
static std::map<uintptr_t, std::function<bool()>> g_poll_registry;
static uintptr_t g_poll_next_id = 1;

// extern "C" uintptr_t register_closure_with_flags(
//     std::function<void()> fn,
//     bool auto_delete,
//     const char *tag)
// {
//     std::lock_guard<std::mutex> lk(g_blocking_mu);
//     uintptr_t id = g_blocking_next_id++;
//     g_blocking_registry[id] = ClosureEntry{
//         .fn = std::move(fn),
//         .auto_delete = auto_delete,
//         .debug_tag = tag ? tag : ""};
//     // printf(">>> Registered closure id=%lu, tag='%s', auto_delete=%d\n",
//     //        id, g_blocking_registry[id].debug_tag.c_str(), g_blocking_registry[id].auto_delete);
//     return id;
// }

extern "C" uintptr_t register_closure(std::function<void()> fn)
{
    std::lock_guard<std::mutex> lk(g_blocking_mu);
    uintptr_t id = g_blocking_next_id++;
    g_blocking_registry[id] = std::move(fn);
    return id;
    // return register_closure_with_flags(std::move(fn), false, nullptr);
}

// extern "C" void unregister_closure(uintptr_t id)
// {
//     std::lock_guard<std::mutex> lk(g_blocking_mu);
//     size_t erased = g_blocking_registry.erase(id);
//     if (erased == 0)
//     {
//         throw std::runtime_error((std::ostringstream() << "closure id=" << id << " not found").str());
//     }
//     // else
//     // {
//     //     printf(">>> unregister: Closure id=%lu released\n", id);
//     // }
// }

extern "C" void call_registered_closure(uintptr_t id)
{
    std::function<void()> fn;
    {
        std::lock_guard<std::mutex> lk(g_blocking_mu);
        auto it = g_blocking_registry.find(id);
        if (it == g_blocking_registry.end())
        {
            throw std::runtime_error((std::ostringstream() << "closure id=" << id << " not found").str());
            return;
        }
        fn = it->second;
        g_blocking_registry.erase(it);
    }
    // printf(">>> call: Running closure id=%lu, tag='%s'\n", id, entry.debug_tag.c_str());
    fn();
}

extern "C" uintptr_t register_poll_closure(std::function<bool()> fn)
{
    std::lock_guard<std::mutex> lk(g_poll_mu);
    uintptr_t id = g_poll_next_id++;
    g_poll_registry[id] = std::move(fn);
    return id;
    // return register_closure_with_flags(std::move(fn), false, nullptr);
}

extern "C" bool call_registered_poll_closure(uintptr_t id)
{
    std::function<bool()> fn;
    std::map<uintptr_t, std::function<bool()>>::iterator it;
    {
        std::lock_guard<std::mutex> lk(g_poll_mu);
        it = g_poll_registry.find(id);
        if (it == g_poll_registry.end())
        {
            throw std::runtime_error((std::ostringstream() << "closure id=" << id << " not found").str());
            return false;
        }
        fn = it->second;
    }
    // printf(">>> call: Running closure id=%lu, tag='%s'\n", id, entry.debug_tag.c_str());
    bool ok = fn();
    if (ok)
    {
        std::lock_guard<std::mutex> lk(g_poll_mu);
        g_poll_registry.erase(it);
    }
    return ok;
}

// extern "C" pybind11::capsule make_capsule_from_pyfunc(pybind11::function f, bool with_gil)
// {
//     // 创建新的 std::function<void()> 并捕获 Python 函数
//     auto fn = with_gil ? new std::function<void()>([f]()
//                                                    {pybind11::gil_scoped_acquire gil;f(); })
//                        : new std::function<void()>([f]()
//                                                    { f(); });

//     // 封装成 capsule，并绑定销毁逻辑
//     return pybind11::capsule(
//         fn,
//         "closure_fn", // 类型名标记，可用于类型验证
//         [](void *p)
//         {
//             delete static_cast<std::function<void()> *>(p);
//         });
// }
