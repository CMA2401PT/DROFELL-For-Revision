package main

import (
	"runtime"
	"sync"
	"unsafe"
)

var pins sync.Map = sync.Map{}

func removePin(ptr uintptr) {
	pinner, _ := pins.LoadAndDelete(ptr)
	p := (pinner.(runtime.Pinner))
	p.Unpin()
}

func addPin(ptr unsafe.Pointer) uintptr {
	pinner := runtime.Pinner{}
	pinner.Pin(ptr)
	pins.Store(uintptr(ptr), pinner)
	return uintptr(ptr)
}
