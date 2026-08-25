package main

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"time"
)

const (
	// 配置参数
	TCPAddress           = "127.0.0.1:8888"
	NumPings             = 100
	KeepPressure         = false
	KeepPressureInterval = time.Millisecond
	coolDown             = 10 * time.Millisecond
)

func server() {
	ln, err := net.Listen("tcp", TCPAddress)
	if err != nil {
		panic(err)
	}
	fmt.Println("Listening on :8888")
	for {
		conn, err := ln.Accept()
		if err != nil {
			continue
		}
		go handleConn(conn)
	}
}

// 随机填充 [][]byte，每个子切片大小为 elemSize，共 count 个
func randomFill(count, elemSize int) [][]byte {
	result := make([][]byte, count)
	for i := range result {
		buf := make([]byte, elemSize)
		_, err := rand.Read(buf) // 使用 crypto/rand 保证每次都不同
		if err != nil {
			log.Fatalf("rand.Read failed: %v", err)
		}
		result[i] = buf
	}
	return result
}

// 判断两个 [][]byte 是否完全一致（长度和内容都一致）
func equalBytesSlices(a, b [][]byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !bytes.Equal(a[i], b[i]) {
			return false
		}
	}
	return true
}

func client() ([]float64, error) {
	conn, err := net.Dial("tcp", TCPAddress)
	if err != nil {
		panic(err)
	}
	defer conn.Close()

	// 生成测试数据
	// frame := []byte("ping")
	// frames := [][]byte{frame}

	var rtts []float64
	// 测试多个 RTT
	for i := 0; i < NumPings; i++ {
		if KeepPressure && coolDown > time.Millisecond {
			st := time.Now()
			for time.Since(st) < coolDown {
				time.Sleep(time.Millisecond)
				writeFrames(conn, [][]byte{})
				readFrames(conn) // Wait for response
			}
		} else {
			time.Sleep(coolDown)
		}
		sendFrames := randomFill(2, 5)
		start := time.Now()
		if err := writeFrames(conn, sendFrames); err != nil {
			panic(err)
		}
		fmt.Println(time.Since(start).Seconds() * 1000)
		ret, err := readFrames(conn)
		if err != nil {
			panic(err)
		}

		rtt := time.Since(start).Seconds() * 1000 // 毫秒
		if !equalBytesSlices(ret, sendFrames) {
			panic("not equal")
		}
		rtts = append(rtts, rtt)
	}
	conn.Close()
	return rtts, nil
}

func handleConn(conn net.Conn) {
	defer conn.Close()
	for {
		frames, err := readFrames(conn)
		if err != nil {
			if err != io.EOF {
				fmt.Println("Read error:", err)
			}
			return
		}
		// echo back
		if err := writeFrames(conn, frames); err != nil {
			fmt.Println("Write error:", err)
			return
		}
	}
}

// readFrames reads one [][]byte frame group from conn
func readFrames(r io.Reader) ([][]byte, error) {
	var totalSize uint32
	if err := binary.Read(r, binary.BigEndian, &totalSize); err != nil {
		return nil, err
	}

	buf := make([]byte, totalSize)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}

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

// writeFrames packs and writes all frames in a single write
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

// func writeFrames(w io.Writer, frames [][]byte) (err error) {
// 	var totalSize uint32 = 4 // initial size for numFrames field
// 	for _, frame := range frames {
// 		totalSize += 4 + uint32(len(frame)) // 4 for each length + data
// 	}

// 	buf := bytes.NewBuffer(make([]byte, 0, totalSize+4))
// 	binary.Write(buf, binary.BigEndian, totalSize)           // total payload size (excluding itself)
// 	binary.Write(buf, binary.BigEndian, uint32(len(frames))) // number of frames

// 	for _, frame := range frames {
// 		binary.Write(buf, binary.BigEndian, uint32(len(frame))) // each frame size
// 	}

// 	for _, frame := range frames {
// 		buf.Write(frame) // actual data
// 	}

// 	_, err = w.Write(buf.Bytes())
// 	return nil
// }

func plotRTT(rttsTCP []float64) {
	bs, _ := json.Marshal(map[string]any{
		"tcp": rttsTCP,
	})
	os.WriteFile("out.json", bs, 0755)
}

func main() {
	// go server()

	// 等待服务器启动
	// time.Sleep(1 * time.Second)

	fmt.Println("Measuring TCP RTT...")
	rttsTCP, err := client()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("TCP RTTs: min=%.3f ms, max=%.3f ms, avg=%.3f ms\n",
		min(rttsTCP), max(rttsTCP), average(rttsTCP))

	plotRTT(rttsTCP)
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
