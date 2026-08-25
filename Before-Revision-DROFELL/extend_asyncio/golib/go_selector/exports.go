package main

// #include <stdlib.h>
import "C"
import (
	"context"
	"fmt"
	"time"
	"unsafe"
)

func errToCString(err error) (errP *C.char) {
	if err != nil {
		errP = C.CString(err.Error())
	}
	return errP
}

var selectorCount = 0

func getSelector(ptr uintptr) *Selector {
	return (*Selector)(unsafe.Pointer(ptr))
}

//export ConfigDebugPinBufferAllocLimit
func ConfigDebugPinBufferAllocLimit(limit C.int) {
	DEBUG_PB_ALLOC_LIMIT = int(limit)
}

//export ConfigDebugPrintPinBufferCreate
func ConfigDebugPrintPinBufferCreate(print bool) {
	DEBUG_PRINT_PB_CREATE = bool(print)
}

//export ConfigDebugPrintPinBufferBorrowGiveBack
func ConfigDebugPrintPinBufferBorrowGiveBack(print bool) {
	DEBUG_PRINT_PB_BORROW_GIVE_BACK = bool(print)
}

//export ConfigDebugPrintWaitTime
func ConfigDebugPrintWaitTime(print bool) {
	DEBUG_PRINT_WAIT_TIME = bool(print)
}

//export ConfigHighResolutionTimer
func ConfigHighResolutionTimer(enable bool) {
	HIGH_RESOLUTION_TIMER = bool(enable)
}

//export ConfigSelectorChanSize
func ConfigSelectorChanSize(size int) {
	SELECTOR_CHAN_SIZE = int(size)
}

//export ConfigConnHandlerSendChanSize
func ConfigConnHandlerSendChanSize(size C.int) {
	CONN_HANDER_SEND_CHAN_SIZE = int(size)
}

//export ConfigAsyncZMCSend
func ConfigAsyncZMCSend(enable bool) {
	ASYNC_ZMC_SEND = bool(enable)
}

//export CreateSelector
func CreateSelector(busyWait bool) uintptr {
	fmt.Printf("Create selector: %v\n", selectorCount)
	selectorCount += 1
	ctx, cancel := context.WithCancel(context.Background())
	sel := &Selector{
		EventChan: make(chan Event, SELECTOR_CHAN_SIZE),
		ctx:       ctx,
		cancel:    cancel,
		busyWait:  busyWait,
	}
	return addPin(unsafe.Pointer(sel))
}

//export DestorySelector
func DestorySelector(ptr uintptr) {
	sel := getSelector(ptr)
	sel.cancel()
	removePin(ptr)
}

//export SelectorSelect
func SelectorSelect(selector uintptr, waitUntil C.double) C.int {
	return getSelector(selector).Select(waitUntil)
}

//export SelectorInterrupt
func SelectorInterrupt(selector uintptr) {
	getSelector(selector).Interrupt()
}

//export SelectorReadyReciverID
func SelectorReadyReciverID(selector uintptr) C.int {
	return getSelector(selector).ReadyReciverID()
}

//export SelectorTakeNone
func SelectorTakeNone(selector uintptr) {
	getSelector(selector).TakeNone()
}

//export SelectorTakeFalse
func SelectorTakeFalse(selector uintptr) bool {
	getSelector(selector).TakeNone()
	return false
}

//export SelectorTakeString
func SelectorTakeString(selector uintptr) *C.char {
	return getSelector(selector).TakeString()
}

//export SelectorTakeInt
func SelectorTakeInt(selector uintptr) C.int {
	return getSelector(selector).TakeInt()
}

//export SelectorTakeInt64
func SelectorTakeInt64(selector uintptr) C.longlong {
	return getSelector(selector).TakeInt64()
}

//export SelectorTakeBytes
func SelectorTakeBytes(selector uintptr) (out *C.char, outLen C.int) {
	return getSelector(selector).TakeBytes()
}

//export SelectorTakeFrames
func SelectorTakeFrames(selector uintptr) (outFrames **C.char, outFramesCount C.int, outFramesLen *C.int) {
	return getSelector(selector).TakeFrames()
}

// //export FreeFrames
// func FreeFrames(inFrames **C.char, inFramesCount C.int, inFramesLen *C.int) {
// 	// usafeInFramesLen := unsafe.Slice(inFramesLen, int32(inFramesCount))
// 	usafeInFrames := unsafe.Slice(inFrames, int32(inFramesCount))
// 	for frameI := range inFramesCount {
// 		p := usafeInFrames[frameI]
// 		C.free(unsafe.Pointer(p))
// 	}
// 	C.free(unsafe.Pointer(inFrames))
// 	C.free(unsafe.Pointer(inFramesLen))
// }

//export SelectorTakeZMCFrames
func SelectorTakeZMCFrames(selector uintptr) (outFrames **C.char, outFramesCount C.int, outFramesLen *C.int,
	FramesLenPBExp C.int, FramesPBExp C.int, FramePBExp C.int, FramePBPtr uintptr, recvTime C.longlong,
) {
	return getSelector(selector).TakeZMCFrames()
}

//export FreePB
func FreePB(exp C.int, ptr uintptr) {
	PutPinnedBytesFromValue(int(exp), ptr)
}

//export FreeZMCFramesHead
func FreeZMCFramesHead(fsExp C.int, fsPbPtr uintptr, flExp C.int, flPtr uintptr) {
	PutPinnedBytesFromValue(int(fsExp), fsPbPtr)
	PutPinnedBytesFromValue(int(flExp), flPtr)
}

//export SelectorTCPConnect
func SelectorTCPConnect(selector uintptr, reciverID C.int, data *C.char) {
	getSelector(selector).TCPConnect(C.GoString(data), ReciverID(reciverID))
}

//export SelectorTCPServe
func SelectorTCPServe(selector uintptr, reciverID C.int, data *C.char) {
	getSelector(selector).TCPServe(C.GoString(data), ReciverID(reciverID))
}

//export SelectorPipeReader
func SelectorPipeReader(selector, readFd uintptr, reciverID C.int) (ptr uintptr) {
	return getSelector(selector).PipeReader(readFd, ReciverID(reciverID))
}

//export SelectorPipeWriter
func SelectorPipeWriter(selector, writeFd uintptr) (ptr uintptr) {
	return getSelector(selector).PipeWriter(writeFd)
}

//export SelectorTakeConnect
func SelectorTakeConnect(selector uintptr, reciverID C.int) (ptr uintptr, ep *C.char) {
	return getSelector(selector).TakeConnect(ReciverID(reciverID))
}

//export HandlerZMCSend
func HandlerZMCSend(handler uintptr, keepAliveObjID C.int, inFrames **C.char, inFramesCount C.int, inFramesLen *C.int) {
	getHandler(handler).ZMCSend(inFrames, keepAliveObjID, inFramesCount, inFramesLen)
}

//export SelectorDelagateCBlockingEnclosure
func SelectorDelagateCBlockingEnclosure(handler uintptr, id uintptr, reciverID C.int) {
	getSelector(handler).DelagateCBlockingEnclosure(id, ReciverID(reciverID))
}

//export SelectorDelagatePollEnclosure
func SelectorDelagatePollEnclosure(handler uintptr, id uintptr, reciverID C.int) {
	getSelector(handler).DelagateCPollEnclosure(id, ReciverID(reciverID))
}

//export DestoryHandler
func DestoryHandler(handler uintptr) {
	getHandler(handler).Close()
}

// //export HighResolutionSleep
// func HighResolutionSleep(seconds C.double) {
// 	nowTime := time.Now()
// 	ns := int64(float64(seconds) * 1e9)
// 	l, high := highResoltionTimer.WaitHighRes(nowTime, time.Duration(ns))
// 	<-l
// 	for high() {
// 		BusyWait()
// 	}
// }

//export WorkLoadSimulate
func WorkLoadSimulate(seconds C.double, name *C.char) {
	goName := C.GoString(name)
	duration := time.Duration(float64(seconds) * float64(time.Second))
	Work(duration, goName)
}
