package main

import (
	"runtime"
	"time"
)

type WorkCtrl struct {
	startTime time.Time
	load      time.Duration // 总工作量（独占下需要的时间）
	progress  time.Duration // 当前已完成工作
	workName  string        // 任务名称
	done      chan struct{} // 用于阻塞调用者
}

// 提交任务的统一入口
var WorkControlChan = make(chan *WorkCtrl, 128)

var penalty = 1.0 // 每增加一个任务，执行速度减缓

func workLoop() {
	var active []*WorkCtrl
	var waitDuration time.Duration
	stepStartTime := time.Now()

	updateWorkProgress := func() {
		newStartTime := time.Now()
		stepDuration := newStartTime.Sub(stepStartTime)
		stepStartTime = newStartTime

		concurrency := len(active)
		if concurrency == 0 || stepDuration <= 0 {
			return
		}

		// 模拟每个任务实际获得的执行时间
		stepPerTask := time.Duration(float64(stepDuration) / (penalty * float64(concurrency)))
		for _, t := range active {
			t.progress += stepPerTask
		}
	}

	notifyWorkFinish := func() {
		stillActive := make([]*WorkCtrl, 0, len(active))
		for _, t := range active {
			if t.progress >= t.load {
				close(t.done)
			} else {
				stillActive = append(stillActive, t)
			}
		}
		active = stillActive
	}

	updateTimer := func() {
		// if len(active) == 0 {
		// 	if timer != nil {
		// 		timer.Stop()
		// 		timer = nil
		// 	}
		// 	return
		// }
		if len(active) == 0 {
			return
		}

		// 找出最早完成的任务所需剩余时间
		earliest := active[0].load - active[0].progress
		for _, t := range active[1:] {
			remain := t.load - t.progress
			if remain < earliest {
				earliest = remain
			}
		}

		waitDuration = time.Duration(float64(earliest) * penalty * float64(len(active)))
		// wakeTime=stepStartTime.Add(delay)

		// if timer == nil {
		// 	timer = time.NewTimer(delay)
		// } else {
		// 	if !timer.Stop() {
		// 		select {
		// 		case <-timer.C:
		// 		default:
		// 		}
		// 	}
		// 	timer.Reset(delay)
		// }
	}

	onNewWork := func(newWork *WorkCtrl) {
		updateWorkProgress()
		newWork.progress = time.Since(newWork.startTime)
		notifyWorkFinish()
		active = append(active, newWork)
		updateTimer()
	}

	onTaskFinish := func() {
		updateWorkProgress()
		notifyWorkFinish()
		updateTimer()
	}

	for {
		if len(active) == 0 {
			newWork := <-WorkControlChan
			stepStartTime = time.Now()
			onNewWork(newWork)
		} else {
			low, high := highResoltionTimer.WaitHighRes(stepStartTime, waitDuration)
			select {
			case newWork := <-WorkControlChan:
				onNewWork(newWork)
			case <-low:
				for high() {
					select {
					case newWork := <-WorkControlChan:
						onNewWork(newWork)
					default:
						runtime.Gosched()
					}
				}
				onTaskFinish()
			}
		}
	}
}

func init() {
	go workLoop()
}

// 外部调用接口，阻塞直到完成
func Work(duration time.Duration, name string) {
	startTime := time.Now()
	done := make(chan struct{})
	ctrl := &WorkCtrl{
		startTime: startTime,
		load:      duration,
		progress:  0,
		workName:  name,
		done:      done,
	}
	WorkControlChan <- ctrl
	<-done
	// actualDelta := time.Since(startTime)
	// fmt.Printf("work %v actual takes %v to finish.\n", name, actualDelta)
}
