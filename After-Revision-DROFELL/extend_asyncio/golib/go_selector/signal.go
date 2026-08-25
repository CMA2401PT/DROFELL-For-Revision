package main

import (
	"fmt"
	"os"
	"os/signal"
	"syscall"
)

func init() {
	go func() {
		sigs := make(chan os.Signal, 1)
		signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
		<-sigs
		fmt.Println("Received interrupt in Go")
		os.Exit(1) // 或设置退出标志等
	}()
}
