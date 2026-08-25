package main

/*
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"math"
	"sync"
	"unsafe"
)

type PinnedBytesPool []sync.Pool

func newPinnedBytesPool(maxExp int) PinnedBytesPool {
	count := 0
	mallocPinnedBytes := func(exp int) PinnedBytes {
		ptr := C.malloc(C.size_t(1 << exp))
		if ptr == nil {
			panic("C.malloc failed")
		}

		if DEBUG_PRINT_PB_CREATE {
			fmt.Println("make pb:", count)
		}
		count += 1
		if DEBUG_PB_ALLOC_LIMIT > 0 && count > DEBUG_PB_ALLOC_LIMIT {
			panic(fmt.Errorf("too many allocations, memory leak?"))
		}
		return PinnedBytes{Exp: exp, Ptr: uintptr(ptr)}
	}

	bp := make([]sync.Pool, maxExp+1)
	for i := 0; i <= maxExp; i++ {
		e := i
		bp[i] = sync.Pool{
			New: func(ex int) func() interface{} {
				return func() interface{} {
					return mallocPinnedBytes(ex)
				}
			}(e),
		}
	}
	return bp
}

// index returns the index in the pool slice for a given size.
func (PinnedBytesPool) index(size int) (exp int) {
	return int(math.Ceil(math.Log2(float64(size))))
}

func (bp PinnedBytesPool) Get(size int) PinnedBytes {
	idx := bp.index(size)
	pb := bp[idx].Get().(PinnedBytes)
	return pb
}

func (bp PinnedBytesPool) Put(pb PinnedBytes) {
	bp[pb.Exp].Put(pb)
}

type PinnedBytes struct {
	Exp int
	Ptr uintptr
}

func (pb PinnedBytes) SliceFromFull() []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(pb.Ptr)), 1<<pb.Exp)
}

func (pb PinnedBytes) SliceFromPart(startOffset int, size int) []byte {
	if (startOffset + size) > (1 << pb.Exp) {
		panic(fmt.Errorf("cannot make slice of %v from %v", (startOffset + size), (1 << pb.Exp)))
	}
	ptr := unsafe.Pointer(pb.Ptr + uintptr(startOffset))
	return unsafe.Slice((*byte)(ptr), size)
}

func (pb PinnedBytes) Free() {
	C.free(unsafe.Pointer(pb.Ptr))
}

var pinnedBytesPool = newPinnedBytesPool(30)

// TODO REMOVE DEBUG CHECK
var takens = sync.Map{}

func GetPinnedBytes(size int) PinnedBytes {
	pb := pinnedBytesPool.Get(size)
	if DEBUG_PRINT_PB_BORROW_GIVE_BACK {
		fmt.Printf("get pb: %v of 1<<%v\n", pb.Ptr, pb.Exp)
		takens.Store(pb.Ptr, pb.Exp)
	}
	return pb
}

func PutPinnedBytes(pb PinnedBytes) {
	if DEBUG_PRINT_PB_BORROW_GIVE_BACK {
		if s, ok := takens.LoadAndDelete(pb.Ptr); !ok || s.(int) != pb.Exp {
			panic(fmt.Errorf("unknown put pb: %v of 1<<%v", pb.Ptr, pb.Exp))
		}
		fmt.Printf("put pb: %v of 1<<%v\n", pb.Ptr, pb.Exp)
		// delete(takens, pb.Ptr)
	}
	pinnedBytesPool.Put(pb)
}

func PutPinnedBytesFromValue(Exp int, Ptr uintptr) {
	PutPinnedBytes(PinnedBytes{Exp: Exp, Ptr: Ptr})
}
