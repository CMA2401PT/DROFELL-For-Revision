package main

// #include <stdlib.h>
import "C"
import "time"

func (s *Selector) DebugEventAfter(seconds C.double, reciverID ReciverID, data string) {
	go func() {
		t := time.NewTimer(secondToDuration(seconds))
		select {
		case <-s.ctx.Done():
			return
		case <-t.C:
			if data == "string" {
				s.SendToPython(reciverID, "some string")
			} else if data == "bytes" {
				s.SendToPython(reciverID, []byte("SOME BYTES"))
			} else if data == "frames" {
				s.SendToPython(reciverID, [][]byte{[]byte("BYTES1"), []byte("BYTES2")})
			} else {
				s.SendToPython(reciverID, nil)
			}
		}
	}()
}

//export SelectorDebugEventAfter
func SelectorDebugEventAfter(selector uintptr, seconds C.double, reciverID C.int, data *C.char) {
	getSelector(selector).DebugEventAfter(seconds, ReciverID(reciverID), C.GoString(data))
}

func (s *Selector) DebugCallbackAfter(periodSecond C.double, reciverID ReciverID, data string) {
	go func() {
		t := time.NewTicker(secondToDuration(periodSecond))
		for {
			select {
			case <-s.ctx.Done():
				return
			case <-t.C:
				if data == "string" {
					s.SendToPython(reciverID, "some sting")
				} else if data == "bytes" {
					s.SendToPython(reciverID, []byte("SOME BYTES"))
				} else if data == "frames" {
					s.SendToPython(reciverID, [][]byte{[]byte("BYTES1"), []byte("BYTES2")})
				} else {
					s.SendToPython(reciverID, nil)
				}
			}
		}
	}()
}

//export SelectorDebugCallbackAfter
func SelectorDebugCallbackAfter(selector uintptr, periodSecond C.double, reciverID C.int, data *C.char) {
	getSelector(selector).DebugCallbackAfter(periodSecond, ReciverID(reciverID), C.GoString(data))
}
