package main

import (
	"log"
	"net"
)

// #include <stdlib.h>
import "C"

func OpenTCPServer(address string, handler func(conn net.Conn)) {
	ln, err := net.Listen("tcp", address)
	if err != nil {
		log.Fatal("Error starting TCP server:", err)
	}
	defer ln.Close()
	for {
		conn, err := ln.Accept()
		if err != nil {
			log.Println("Error accepting TCP connection:", err)
			continue
		}
		go func(c net.Conn) {
			handler(c)
			c.Close()
		}(conn)
	}
}

func (s *Selector) TCPServe(address string, callbackID ReciverID) {
	go func() {
		ln, err := net.Listen("tcp", address)
		if err != nil {
			s.SendToPython(callbackID, err)
			return
		}
		for {
			conn, err := ln.Accept()
			if err != nil {
				s.SendToPython(callbackID, err)
				return
			}
			h := NewConnHandler(s, conn, conn)
			go h.sendLoop()
			s.SendToPython(callbackID, h)
		}
	}()
}
