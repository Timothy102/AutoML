package main

import (
	"go/build"

	nn "github.com/timothy102/neuralnetwork"
)

func ecgResNet(units int, dropout_rate float64) *[]nn.Layer {
	return &[]nn.Layer{
		Conv2D(filter, 3, 1, nn.Valid),
		MaxPooling2D(2),
		BatchNormalization(),
		Dropout(0.1),
	}
}

type ECGClassifier struct {
	wavelength float64
	data       []float64
}

func (e *ECGClassifier) Interpret() map[string]float64 {
	st := Statistician{}
	return st.stats
}

func (e *ECGClassifier) BuildModel(shape []int, lr float64) *Model {
	input_layer := Input(shape)
	conv1A := ecgResNet(64)(input_layer)
	conv2A := ecgResNet(32)(conv1A)


	conv1B := ecgResNet(64)(input_layer)
	conv2B := ecgResNet(32)(conv1B)

	concat := nn.concatenate(conv2A, conv2B, axis = 1)
	lstm := LSTM(32, True)(concat)
	dense1 := Dense(16, nn.activations.ReLU())(lstm)
	output := Dense(1, nn.activations.Sigmoid())(dense1)

	return &Model{
		inputs:        input_layer,
		outputs:       output,
		learning_rate: lr,
	}
}

func (e* ECGClassifier) BuildAndCompile(shape []int, lr float64){
	model := e.BuildModel(shape, lr)
	model.compile(nn.optimizers.Adam, nn.losses.MSE, []nn.metrics{
		nn.losses.RMSE(),
	})
	model.summary()
	return model
}


// func main(){
// 	shape := []int{3000,1}
// 	e := ECGClassifier()
// 	ecgModel := e.BuildAndCompile(shape, 0.001)
// }