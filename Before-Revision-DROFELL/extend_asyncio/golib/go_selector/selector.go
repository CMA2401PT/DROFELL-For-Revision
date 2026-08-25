package main

// #include <stdlib.h>
import "C"
import (
	"context"
	"fmt"
	"runtime"
	"time"
	"unsafe"
)

type ReciverID int32

const RECIVER_ID_INTERRUPT = 1
const RECIVER_ID_FREE_PYOBJ = 2

type Event struct {
	Data      any
	ReciverID ReciverID
}

type Selector struct {
	EventChan             chan Event
	readFirst             Event
	ctx                   context.Context
	cancel                context.CancelFunc
	busyWait              bool
	delegatePollBlokcings []func() (bool, ReciverID)
}

func (s *Selector) pollAll() (bool, ReciverID) {
	kept := s.delegatePollBlokcings[:0]
	for i, f := range s.delegatePollBlokcings {
		ok, reciverID := f()
		if ok {
			s.delegatePollBlokcings = append(kept, s.delegatePollBlokcings[i+1:]...)
			// fmt.Println("ok, ", len(s.delegatePollBlokcings))
			return true, reciverID
		}
		// fmt.Println("kept")
		kept = append(kept, f)
	}
	s.delegatePollBlokcings = kept
	return false, 0
}

func secondToDuration(seconds C.double) time.Duration {
	return time.Duration(float64(time.Second) * float64(seconds))
}

func secondToTime(seconds C.double) time.Time {
	fs := float64(seconds)
	sec := int64(fs)
	nsec := int64((fs - float64(sec)) * 1e9) // 将小数部分转为纳秒
	return time.Unix(sec, nsec)
}

func (s *Selector) Select(waitUntil C.double) C.int {
	until := secondToTime(waitUntil)
	if DEBUG_PRINT_WAIT_TIME {
		fmt.Printf("select with timeout: %v\n", time.Until(until))
	}
	if s.readFirst.ReciverID != 0 {
		panic("should take first")
	}
	// infinity wait
	if waitUntil < 0 {
		if s.busyWait {
			for {
				if ok, reciverID := s.pollAll(); ok {
					s.readFirst = Event{Data: nil, ReciverID: reciverID}
					return C.int(reciverID)
				}
				select {
				case s.readFirst = <-s.EventChan:
					return s.ReadyReciverID()
				default:
					runtime.Gosched()
				}
			}
		} else {
			s.readFirst = <-s.EventChan
			return s.ReadyReciverID()
		}
	}
	if waitUntil == 0 || time.Now().After(until) {
		return s.ReadyReciverID()
	}
	if s.busyWait {
		if ok, reciverID := s.pollAll(); ok {
			s.readFirst = Event{Data: nil, ReciverID: reciverID}
			return C.int(reciverID)
		}
		for time.Now().Before(until) {
			select {
			case s.readFirst = <-s.EventChan:
				return s.ReadyReciverID()
			default:
				runtime.Gosched()
			}
		}
		return s.ReadyReciverID()
	} else if HIGH_RESOLUTION_TIMER {
		start := time.Now()
		duration := until.Sub(start)
		low, high := highResoltionTimer.WaitHighRes(start, duration)
		select {
		case <-low:
		case s.readFirst = <-s.EventChan:
			return s.ReadyReciverID()
		}
		for high() {
			select {
			case s.readFirst = <-s.EventChan:
				return s.ReadyReciverID()
			default:
				runtime.Gosched()
			}
		}
		return s.ReadyReciverID()
	} else {
		start := time.Now()
		duration := until.Sub(start)
		t := time.NewTimer(duration)
		select {
		case <-t.C:
		case s.readFirst = <-s.EventChan:
		}
		return s.ReadyReciverID()
	}

}

func (s *Selector) Interrupt() {
	s.SendToPython(RECIVER_ID_INTERRUPT, nil)
}

func (s *Selector) ReadyReciverID() C.int {
	if s.readFirst.ReciverID == 0 {
		// fetch data
		select {
		case s.readFirst = <-s.EventChan:
		default:
		}
	}
	return C.int(s.readFirst.ReciverID)
}

func (s *Selector) SendToPython(reciverID ReciverID, data any) (err error) {
	ev := Event{
		Data:      data,
		ReciverID: reciverID,
	}
	if err := s.ctx.Err(); err != nil {
		return err
	}
	s.EventChan <- ev
	return nil
}

func (s *Selector) NotifyReleasePyObj(objKeepAliveID int32) {
	s.SendToPython(RECIVER_ID_FREE_PYOBJ, objKeepAliveID)
}

func (s *Selector) TakeNone() {
	s.readFirst.ReciverID = 0
	s.readFirst.Data = nil
}

func (s *Selector) TakeString() *C.char {
	sd := s.readFirst.Data.(string)
	s.readFirst.ReciverID = 0
	s.readFirst.Data = nil
	return C.CString(sd)
}

func (s *Selector) TakeInt() C.int {
	sd := s.readFirst.Data.(int32)
	s.readFirst.ReciverID = 0
	s.readFirst.Data = nil
	return C.int(sd)
}

func (s *Selector) TakeInt64() C.longlong {
	sd := s.readFirst.Data.(int64)
	s.readFirst.ReciverID = 0
	s.readFirst.Data = nil
	return C.longlong(sd)
}

func (s *Selector) TakeBytes() (out *C.char, outLen C.int) {
	goBytes := s.readFirst.Data.([]byte)
	outPtr := C.CBytes(goBytes)
	s.readFirst.ReciverID = 0
	s.readFirst.Data = nil
	return (*C.char)(outPtr), C.int(len(goBytes))
}

const (
	INTSize = unsafe.Sizeof(C.int(0))
	PTRSize = unsafe.Sizeof(uintptr(0))
)

func (s *Selector) TakeFrames() (outFrames **C.char, outFramesCount C.int, outFramesLen *C.int) {
	frames := s.readFirst.Data.([][]byte)
	outLen := len(frames)
	outFramesLen = (*C.int)(C.malloc(C.size_t(outLen) * C.size_t(INTSize)))
	outFrames = (**C.char)(C.malloc(C.size_t(outLen) * C.size_t(PTRSize)))
	for frameI := range outLen {
		data := frames[frameI]
		ptr := C.CBytes(data)
		*(*C.int)(unsafe.Add(unsafe.Pointer(outFramesLen), frameI*4)) = C.int(len(data))
		*(*unsafe.Pointer)(unsafe.Add(unsafe.Pointer(outFrames), uintptr(frameI)*PTRSize)) = ptr
	}
	outFramesCount = C.int(outLen)
	s.readFirst.ReciverID = 0
	s.readFirst.Data = nil
	return outFrames, outFramesCount, outFramesLen
}

func (s *Selector) TakeZMCFrames() (
	outFrames **C.char, outFramesCount C.int, outFramesLen *C.int,
	FramesLenPBExp C.int, FramesPBExp C.int, FramePBExp C.int, FramePBPtr uintptr,
	recvTime C.longlong,
) {
	zf := s.readFirst.Data.(ZMCFrames)
	frames := zf.Frames
	outLen := len(frames)
	if outLen == 0 {
		if zf.PB.Ptr != 0 {
			panic("zero zmc frame must create directly by ZMCFrame{}")
		}
		s.readFirst.ReciverID = 0
		s.readFirst.Data = nil
		return
	}
	outFramesLenPB := GetPinnedBytes(outLen * int(INTSize))
	outFramesPB := GetPinnedBytes(outLen * int(PTRSize))
	outFramesLen = (*C.int)(unsafe.Pointer(outFramesLenPB.Ptr))
	outFrames = (**C.char)(unsafe.Pointer(outFramesPB.Ptr))
	for frameI := range outLen {
		data := frames[frameI]
		*(*C.int)(unsafe.Add(unsafe.Pointer(outFramesLen), frameI*4)) = C.int(len(data))
		*(*unsafe.Pointer)(unsafe.Add(unsafe.Pointer(outFrames), uintptr(frameI)*PTRSize)) = unsafe.Pointer(&data[0])
	}
	outFramesCount = C.int(outLen)
	s.readFirst.ReciverID = 0
	s.readFirst.Data = nil
	return outFrames,
		outFramesCount, outFramesLen,
		C.int(outFramesPB.Exp), C.int(outFramesLenPB.Exp),
		C.int(zf.PB.Exp), zf.PB.Ptr,
		C.longlong(zf.RecvTime)
}

func main() {}
