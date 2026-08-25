package main

import (
	"fmt"
	"time"
)

/*
#include <x86intrin.h>
void cpu_nop() {
    __asm__ __volatile__("nop");
}
*/
import "C"

// func BusyWait() {
// 	// for _ = range 100 {
// 	// 	C.cpu_nop()
// 	// }
// 	runtime.Gosched()
// }

func measureSleepSimple(sleepTime time.Duration, warmup int, iterations int) time.Duration {
	maxT := time.Duration(0)
	for i := 0; i < warmup; i++ {
		time.Sleep(sleepTime)
	}
	for i := 0; i < iterations; i++ {
		start := time.Now()
		time.Sleep(sleepTime)
		t := time.Since(start)
		if t > maxT {
			maxT = t
		}
	}
	return maxT
}

func getSleepSafeThreshold() time.Duration {
	iterations := 5
	warmup := 2
	sleepTimes := []time.Duration{}
	for i := range 3 {
		sleepTimes = append(sleepTimes, time.Duration(i)*time.Millisecond)
	}
	for i := range 10 {
		safeThres := time.Duration(i) * time.Millisecond
		ok := true
		for _, st := range sleepTimes {
			maxOff := measureSleepSimple(st, warmup, iterations)
			if maxOff-st > safeThres {
				ok = false
				break
			}
		}
		if ok {
			return safeThres
		}
	}
	panic(fmt.Sprintf("cannot found safe thres, max offset @ 1ms: %v", measureSleepSimple(time.Millisecond, warmup, iterations)))
}

type HighResoltionTimer time.Duration

// func (t HighResoltionTimer) WaitLowRes(d time.Duration, isCacel func() bool) chan struct{} {
// 	start := time.Now()
// 	c := make(chan struct{}, 1)
// 	go func() {
// 		expectEnd := start.Add(d)
// 		lowResolutionSleepTime := (d - time.Duration(t))
// 		if lowResolutionSleepTime > time.Millisecond {
// 			<-time.NewTimer(lowResolutionSleepTime).C
// 			if isCacel() {
// 				return
// 			}
// 		}
// 		for time.Now().Before(expectEnd) {
// 			BusyWait()
// 			if isCacel() {
// 				return
// 			}
// 		}
// 		c <- struct{}{}
// 	}()
// 	return c
// }

func (t HighResoltionTimer) WaitHighRes(start time.Time, d time.Duration) (lowResW <-chan time.Time, highResSpin func() (waitMore bool)) {
	expectEnd := start.Add(d)
	lowResolutionSleepTime := (d - time.Duration(t))
	if lowResolutionSleepTime > 0 {
		lowResW = time.NewTimer(lowResolutionSleepTime).C
	} else {
		lowResRW := make(chan time.Time)
		close(lowResRW)
		lowResW = lowResRW
	}

	highResW := func() (waitMore bool) {
		return time.Now().Before(expectEnd)
	}
	return lowResW, highResW
}

var highResoltionTimer HighResoltionTimer

func init() {
	if HIGH_RESOLUTION_TIMER {
		safeThres := getSleepSafeThreshold()
		fmt.Printf("safe thres: %v\n", safeThres)
		highResoltionTimer = HighResoltionTimer(safeThres)
	}
}
