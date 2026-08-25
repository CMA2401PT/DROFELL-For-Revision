package main

import (
	"encoding/binary"
	"io"
	"net"
	"runtime"
	"slices"
	"sync"
	"time"
	"unsafe"
)

// #include <stdlib.h>
import "C"

type ConnHandler struct {
	selector  *Selector
	pinner    runtime.Pinner
	selfPtr   uintptr
	sendChan  chan dataToSend
	closeOnce sync.Once
	closed    bool
	closeConn func() error
	writer    io.WriteCloser
	reader    io.Reader
}

func (h *ConnHandler) Ptr() uintptr {
	if h == nil {
		return 0
	}
	return h.selfPtr
}

func (h *ConnHandler) Close() {
	h.closeOnce.Do(func() {
		h.closed = true
		if h.writer != nil {
			h.writer.Close()
		}
		removePin(h.selfPtr)
	})
}

func (h *ConnHandler) readLoop(recvID ReciverID) {
	for {
		zf, err := readFramesZMC(h.reader)
		if err != nil {
			h.selector.SendToPython(recvID, ZMCFrames{})
			h.Close()
			return
		}
		if len(zf.Frames) == 0 {
			// pressure frame
			continue
		}
		h.selector.SendToPython(recvID, zf)
	}
}

func (h *ConnHandler) sendLoop() {
	for d := range h.sendChan {
		h.zmcSend(d)
		h.selector.NotifyReleasePyObj(int32(d.keepAliveObjID))
	}
}

// func (h *ConnHandler) Send(inFrames **C.char, inFramesCount C.int, inFramesLen *C.int) {
// 	h.mu.Lock()
// 	defer h.mu.Unlock()
// 	usafeInFramesLen := unsafe.Slice(inFramesLen, int32(inFramesCount))
// 	usafeInFrames := unsafe.Slice(inFrames, int32(inFramesCount))
// 	cvtInFrames := make([][]byte, inFramesCount)
// 	for frameI := range int32(inFramesCount) {
// 		frameLen := usafeInFramesLen[frameI]
// 		// goBytes := C.GoBytes(unsafe.Pointer(usafeInFrames[frameI]), frameLen)
// 		goBytes := unsafe.Slice((*byte)(unsafe.Pointer(usafeInFrames[frameI])), frameLen)
// 		cvtInFrames[frameI] = goBytes
// 	}
// 	writeFrames(h.Conn, cvtInFrames)
// }

// func (h *ConnHandler) ZMCSend(inFrames **C.char, inFramesCount C.int, inFramesLen *C.int) {
// 	h.mu.Lock()
// 	defer h.mu.Unlock()
// 	usafeInFramesLen := unsafe.Slice(inFramesLen, int32(inFramesCount))
// 	usafeInFrames := unsafe.Slice(inFrames, int32(inFramesCount))
// 	cvtInFrames := make([][]byte, inFramesCount)
// 	for frameI := range int32(inFramesCount) {
// 		frameLen := usafeInFramesLen[frameI]
// 		// goBytes := C.GoBytes(unsafe.Pointer(usafeInFrames[frameI]), frameLen)
// 		goBytes := unsafe.Slice((*byte)(unsafe.Pointer(usafeInFrames[frameI])), frameLen)
// 		cvtInFrames[frameI] = goBytes
// 	}
// 	writeFrames(h.Conn, cvtInFrames)
// }

type dataToSend struct {
	count          int
	ptrs           []*C.char
	lens           []C.int
	keepAliveObjID C.int
}

func (h *ConnHandler) zmcSend(data dataToSend) {
	count := data.count
	ptrs := data.ptrs
	lens := data.lens

	totalSize := 4 + 4*count // totalSize + numFrames + lengths
	for i := 0; i < count; i++ {
		totalSize += int(lens[i])
	}

	// header: totalSize (4) + numFrames (4) + frame lengths (4*numFrames)
	header := make([]byte, 4+4+4*count)
	binary.BigEndian.PutUint32(header[0:4], uint32(totalSize))
	binary.BigEndian.PutUint32(header[4:8], uint32(count))
	for i := 0; i < count; i++ {
		binary.BigEndian.PutUint32(header[8+i*4:12+i*4], uint32(lens[i]))
	}

	// write header
	if _, err := h.writer.Write(header); err != nil {
		return
	}

	// write all frames data
	for i := 0; i < count; i++ {
		frame := unsafe.Slice((*byte)(unsafe.Pointer(ptrs[i])), lens[i])
		if _, err := h.writer.Write(frame); err != nil {
			return
		}
	}
}

func NewConnHandler(s *Selector, w io.WriteCloser, r io.Reader) *ConnHandler {
	h := &ConnHandler{
		selector: s,
		writer:   w,
		reader:   r,
		pinner:   runtime.Pinner{},
		selfPtr:  0,
		sendChan: make(chan dataToSend, CONN_HANDER_SEND_CHAN_SIZE),
	}
	h.selfPtr = addPin(unsafe.Pointer(h))
	return h
}

func (h *ConnHandler) ZMCSend(inFrames **C.char, keepAliveObjID C.int, inFramesCount C.int, inFramesLen *C.int) {
	count := int(inFramesCount)
	// ptrs := unsafe.Slice(inFrames, count)
	// lens := unsafe.Slice(inFramesLen, count)
	// copy meta
	ptrs := slices.Clone(unsafe.Slice(inFrames, count))
	lens := slices.Clone(unsafe.Slice(inFramesLen, count))
	d := dataToSend{count: count, ptrs: ptrs, lens: lens, keepAliveObjID: keepAliveObjID}

	if h == nil {
		return
	}

	if ASYNC_ZMC_SEND {
		select {
		case h.sendChan <- d:
		default:
			panic("send chan too small")
		}
	} else {
		h.zmcSend(d)
		h.selector.NotifyReleasePyObj(int32(d.keepAliveObjID))
	}

}

func (s *Selector) TCPConnect(address string, callbackID ReciverID) {
	go func() {
		conn, err := net.Dial("tcp", address)
		if err != nil {
			s.SendToPython(callbackID, err)
			return
		}
		h := NewConnHandler(s, conn, conn)
		go h.sendLoop()
		s.SendToPython(callbackID, h)
	}()
}

func (s *Selector) TakeConnect(reciverID ReciverID) (ptr uintptr, ep *C.char) {
	h, ok := (s.readFirst.Data).(*ConnHandler)
	if !ok {
		err := (s.readFirst.Data).(error)
		s.readFirst.ReciverID = 0
		s.readFirst.Data = nil
		return 0, errToCString(err)
	} else {
		go h.readLoop(ReciverID(reciverID))
		s.readFirst.ReciverID = 0
		s.readFirst.Data = nil
		return h.Ptr(), errToCString(nil)
	}
}

func getHandler(ptr uintptr) *ConnHandler {
	return (*ConnHandler)(unsafe.Pointer(ptr))
}

func readFrames(r io.Reader) ([][]byte, error) {
	var totalSize uint32
	if err := binary.Read(r, binary.BigEndian, &totalSize); err != nil {
		return nil, err
	}

	if totalSize == 0 {
		return nil, nil
	}

	buf := make([]byte, totalSize)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	return unPackFramesBase(buf)
}

type ZMCFrames struct {
	RecvTime int64
	PB       PinnedBytes
	Frames   [][]byte
}

func readFramesZMC(r io.Reader) (zf ZMCFrames, err error) {
	var totalSize uint32
	if err := binary.Read(r, binary.BigEndian, &totalSize); err != nil {
		return zf, err
	}
	t := time.Now().UnixNano()
	defer func() {
		zf.RecvTime = t
	}()

	if totalSize == 4 {
		// fmt.Println(totalSize)
		// appending 0000
		binary.Read(r, binary.BigEndian, &totalSize)
		if totalSize != 0 {
			panic("should be 0")
		}
		return zf, nil
	}

	pb := GetPinnedBytes(int(totalSize))
	buf := pb.SliceFromPart(0, int(totalSize))
	if _, err := io.ReadFull(r, buf); err != nil {
		PutPinnedBytes(pb)
		return zf, err
	}
	frame, err := unPackFramesBase(buf)
	if err != nil {
		PutPinnedBytes(pb)
		return zf, err
	}
	return ZMCFrames{PB: pb, Frames: frame}, nil
}

func unPackFramesBase(buf []byte) ([][]byte, error) {
	// Parse numFrames
	if len(buf) < 4 {
		return nil, io.ErrUnexpectedEOF
	}
	numFrames := binary.BigEndian.Uint32(buf[:4])

	lengthsOffset := 4
	dataOffset := 4 + int(numFrames)*4

	if len(buf) < dataOffset {
		return nil, io.ErrUnexpectedEOF
	}

	frames := make([][]byte, numFrames)
	cursor := dataOffset
	for i := uint32(0); i < numFrames; i++ {
		// Read frame length
		start := lengthsOffset + int(i)*4
		length := binary.BigEndian.Uint32(buf[start : start+4])

		// Slice frame directly from buffer
		if cursor+int(length) > len(buf) {
			return nil, io.ErrUnexpectedEOF
		}
		frames[i] = buf[cursor : cursor+int(length)]
		cursor += int(length)
	}

	return frames, nil
}

func writeFrames(w io.Writer, frames [][]byte) (err error) {
	numFrames := len(frames)
	totalSize := 4 + 4*numFrames
	for _, frame := range frames {
		totalSize += len(frame)
	}

	buf := make([]byte, 4+4+4*numFrames) // totalSize + numFrames + lengths
	binary.BigEndian.PutUint32(buf[0:4], uint32(totalSize))
	binary.BigEndian.PutUint32(buf[4:8], uint32(numFrames))
	for i, frame := range frames {
		binary.BigEndian.PutUint32(buf[8+i*4:12+i*4], uint32(len(frame)))
	}

	if _, err := w.Write(buf); err != nil {
		return err
	}
	for _, frame := range frames {
		if _, err := w.Write(frame); err != nil {
			return err
		}
	}
	return nil
}
