package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"path/filepath"
	"time"
)

const (
	// 配置参数
	TCPAddress   = "127.0.0.1:8888"
	NumPings     = 100
	KeepPressure = false
	coolDown     = 10 * time.Millisecond
)

var UnixSocketPath = filepath.Join(os.TempDir(), "drofell_test_unix_socket")

func handleTCPConnection(conn net.Conn) {
	defer conn.Close()
	for {
		buffer := make([]byte, 4)
		_, err := conn.Read(buffer)
		if err != nil {
			log.Println("Error reading:", err)
			return
		}
		_, err = conn.Write(buffer) // Echo back
		if err != nil {
			log.Println("Error writing:", err)
			return
		}
	}
}

func handleUnixConnection(conn net.Conn) {
	defer conn.Close()
	for {
		buffer := make([]byte, 4)
		_, err := conn.Read(buffer)
		if err != nil {
			log.Println("Error reading:", err)
			return
		}
		_, err = conn.Write(buffer) // Echo back
		if err != nil {
			log.Println("Error writing:", err)
			return
		}
	}
}

func startTCPServer() {
	ln, err := net.Listen("tcp", TCPAddress)
	if err != nil {
		log.Fatal("Error starting TCP server:", err)
	}
	defer ln.Close()

	log.Println("TCP server listening on", TCPAddress)
	for {
		conn, err := ln.Accept()
		if err != nil {
			log.Println("Error accepting TCP connection:", err)
			continue
		}
		go handleTCPConnection(conn)
	}
}

func startUnixServer() {
	// 删除旧的 Unix socket 文件
	if _, err := os.Stat(UnixSocketPath); err == nil {
		err := os.Remove(UnixSocketPath)
		if err != nil {
			log.Fatal("Error removing old Unix socket:", err)
		}
	}

	ln, err := net.Listen("unix", UnixSocketPath)
	if err != nil {
		log.Fatal("Error starting Unix server:", err)
	}
	defer ln.Close()

	log.Println("Unix server listening on", UnixSocketPath)
	for {
		conn, err := ln.Accept()
		if err != nil {
			log.Println("Error accepting Unix connection:", err)
			continue
		}
		go handleUnixConnection(conn)
	}
}

func measureRTT(socketType string, address string) ([]float64, error) {
	var rtts []float64
	var conn net.Conn
	var err error

	if socketType == "unix" {
		conn, err = net.Dial("unix", address)
	} else {
		conn, err = net.Dial("tcp", address)
	}
	if err != nil {
		panic(err)
	}

	keepAlive := make([]byte, 4)
	for i := 0; i < NumPings; i++ {
		// 连接到服务器
		if KeepPressure {
			st := time.Now()
			for time.Since(st) < coolDown {
				time.Sleep(time.Millisecond * 1)
				conn.Write(keepAlive)
				io.ReadFull(conn, keepAlive) // Wait for response
			}
		} else {
			time.Sleep(coolDown)
		}

		// 发送数据并测量 RTT
		start := time.Now()
		buf := make([]byte, 4)
		_, err := conn.Write([]byte("ping"))
		if err != nil {
			return nil, fmt.Errorf("Error sending data: %w", err)
		}

		io.ReadFull(conn, buf) // Wait for response
		if err != nil {
			return nil, fmt.Errorf("Error reading data: %w", err)
		}

		rtt := time.Since(start).Seconds() * 1000 // 毫秒
		rtts = append(rtts, rtt)

	}
	conn.Close()

	return rtts, nil
}

func plotRTT(rttsTCP []float64, rttsUnix []float64) {
	bs, _ := json.Marshal(map[string]any{
		"tcp":  rttsTCP,
		"unix": rttsUnix,
	})
	os.WriteFile("out.json", bs, 0755)
}

func main() {
	go startTCPServer()
	go startUnixServer()

	// 等待服务器启动
	time.Sleep(1 * time.Second)

	// 测量 TCP 和 Unix Socket RTT
	fmt.Println("Measuring TCP RTT...")
	rttsTCP, err := measureRTT("tcp", TCPAddress)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Measuring Unix Socket RTT...")
	rttsUnix, err := measureRTT("unix", UnixSocketPath)
	if err != nil {
		log.Fatal(err)
	}

	// 打印结果
	fmt.Printf("TCP RTTs: min=%.3f ms, max=%.3f ms, avg=%.3f ms\n",
		min(rttsTCP), max(rttsTCP), average(rttsTCP))

	fmt.Printf("Unix RTTs: min=%.3f ms, max=%.3f ms, avg=%.3f ms\n",
		min(rttsUnix), max(rttsUnix), average(rttsUnix))

	// 绘制图像
	plotRTT(rttsTCP, rttsUnix)
}

func min(arr []float64) float64 {
	min := arr[0]
	for _, v := range arr {
		if v < min {
			min = v
		}
	}
	return min
}

func max(arr []float64) float64 {
	max := arr[0]
	for _, v := range arr {
		if v > max {
			max = v
		}
	}
	return max
}

func average(arr []float64) float64 {
	var sum float64
	for _, v := range arr {
		sum += v
	}
	return sum / float64(len(arr))
}
