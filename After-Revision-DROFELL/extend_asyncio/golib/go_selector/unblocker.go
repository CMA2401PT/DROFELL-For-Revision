package main

/*
#include <stdint.h>
#include <stdbool.h>

void call_registered_closure(uintptr_t id);
// void unregister_closure(uintptr_t id);
bool call_registered_poll_closure(uintptr_t id);
*/
import "C"
import "runtime"

//go:nosplit
func callRegisteredPollClosure(id uintptr) bool {
	cOk := C.call_registered_poll_closure(C.uintptr_t(id))
	return bool(cOk)
}

func (s *Selector) DelagateCBlockingEnclosure(id uintptr, reciverID ReciverID) {
	go func() {
		C.call_registered_closure(C.uintptr_t(id))
		s.SendToPython(reciverID, nil)
	}()
}

func (s *Selector) DelagateCPollEnclosure(id uintptr, reciverID ReciverID) {
	if !s.busyWait {
		go func() {
			for {
				cOk := callRegisteredPollClosure(id)
				if cOk {
					s.SendToPython(reciverID, nil)
					return
				} else {
					runtime.Gosched()
				}
			}
		}()
	} else {
		enclosure := func() (bool, ReciverID) {
			cOk := callRegisteredPollClosure(id)
			if cOk {
				return true, reciverID
			} else {
				return false, 0
			}
		}
		s.delegatePollBlokcings = append(s.delegatePollBlokcings, enclosure)
	}
}
