package main

import (
	"fmt"
	"os"
)

func (s *Selector) PipeReader(readFd uintptr, reciverID ReciverID) uintptr {
	r := os.NewFile(readFd, fmt.Sprintf("pipe-reader-%d", readFd))
	h := NewConnHandler(s, nil, r)
	go h.readLoop(reciverID)
	return h.Ptr()
}

type pipeWriterTrimClose struct {
	base *os.File
}

func (t *pipeWriterTrimClose) Close() error {
	return nil
}

func (t *pipeWriterTrimClose) Write(data []byte) (int, error) {
	return t.base.Write(data)
}

func (s *Selector) PipeWriter(writeFd uintptr) uintptr {
	w := os.NewFile(writeFd, fmt.Sprintf("pipe-writer-%d", writeFd))
	trim := &pipeWriterTrimClose{base: w}
	h := NewConnHandler(s, trim, nil)
	go h.sendLoop()
	return h.Ptr()
}
