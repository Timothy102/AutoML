package main

import (
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/mjibson/go-dsp/fft"
)

// stats + getting ready for tf model

//trend

type Statistician struct{
	data []float64
	mode string
	stats map[string] float64
	time time.Time
	size int
	lambdaFunction func(float64) float64
	decrease bool
}

func (s* Statistician) min() float64 {
	min := 1e8
	for _, k := range s.data {
		if k < min {
			min = k
		}
	}
	return min
}

func (s* Statistician) max() float64 {
	max := -1e8
	for _, k := range s.data {
		if k > max {
			max = k
		}
	}
	return max
}

func (s* Statistician) mean_trend(space float64) float64 {
	rand.Seed(42)
	var trend float64
	for i := range s.data {
		if i%2 == 0 {
			trend += math.Abs(float64(s.data[i]) - float64(rand.Intn(len(s.data))))
		}
	}
	return trend / float64(len(s.data))
}

func (s* Statistician) min_max_ratio() float64 {
	return (min(s.data) - max(s.data)) / max(s.data)
}

func (s* Statistician) frequency() float64 {
	var count, zaporedno int
	for i := range s.data {
		if s.data[i] < s.data[i+1] {
			zaporedno += 1
		}else{
			zaporedno = 0, count += 1
		}
	}
	return count
}

func (s* Statistician) sort() []float64 {
	for i := len(s.data); i > 0; i-- {
		for j := 1; j < i; j++ {
			if s.data[j-1] > s.data[j] {
				swap(s.data, j)
			}
		}
	}
	return s.data
}
func swap(ps []float64, index int) {
	val := ps[index]
	ps[index] = ps[index-1]
	ps[index-1] = val
}

func (s* Statistician) mediana()float64{
	return sort(s.data)[len(s.data)/2]
}

func (s* Statistician) mean() float64{
	var summa float64
	for _,k := range s.data{
		summa += k
	}
	return summa / float64(len(s.data))
}

func (s* Statistician) variance() float64 {
	var sum float64
	avg := mean(s.data)
	for _, p := range s.data {
		sum += math.Pow(p.X-avg, 2)
	}
	return sum / float64(len(s.data))
}


func overlap(data1, data2 []float64) float64{
	m1 := mean(data1)/2*math.Sqrt(variance(data1))
	m2 := mean(data2)/2*math.Sqrt(variance(data2))
	return math.Log(m1/m2)
}
func (s* Statistician) taylorSeriesApproximationScore() float64{
	k:= plot.SinusEstimate(s.data)
	return overlap(s.data, k)
}

func (s* Statistician) relativnaNapaka() float64{
	return math.Abs(max(s.data) - mean(s.data))/mean(s.data)
}

func (s*Statistician) fourier_transform() float64{
	var step float64
	a := fft.FFTReal(s.data)
	for i := range a{
		step += a[i+1]-a[i]
	}
	return step
}


type Model interface{
	// tle bo verjetno kr od neuralnetworka
}

func (s*Statistician) clearDataIndex() float64{
	k := s.taylorSeriesApproximationScore()
	f := s.fourier_transform()
	return 2*k*s/(k+s)
}

func (s* Statistician) getTemporalComponent() float64{
	return s.min_max_ratio()+s.mean_trend()+s.frequency()
}


func sigmoid(x float64) float64{
	return 1/(1+math.Exp(-x))
}
// the clearer the model, the more you can do statistics and simple 1d convolutional
// the rougher, the deeper model
def (s* Statistician) GetBestModelScore() (float64, error){
	clearValue := s.clearDataIndex()
	tempValue := s.getTemporalComponent()

	if clearValue == 0 && tempValue == 0{
		return nil, fmt.Errorf("You have given flat data, check seasonality")
	}
	if clearValue < tempValue{
		return clearValue/tempValue, nil
	}
	return tempValue/clearValue-2*(tempValue-clearValue), nil
}