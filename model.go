package main

import (
	"log"

	nn "github.com/timothy102/neuralnetwork"
)

// at this point we have ourselves a score 0-1 indicating roughness, or
// vice versa trendiness, low score = trendy data

func convBlock(filter int) []nn.Layer {
	return []nn.Layer{
		Conv1D(filter, 3, 1, nn.Valid),
		MaxPooling1D(2),
		BatchNormalization(),
	}
}

func reverseConvBlock(filter int) []nn.Layer {
	return []nn.Layer{
		TransposeConv1D(filter, 3, 1, nn.Valid),
		BatchNormalization(),
	}
}

func Autoencoder(score float64) *nn.Model {
	steps := int(score/10) / 2
	var layers []nn.Layer
	for i := range steps {
		layers = append(layers, convBlock(16*i))
	}
	for i := range steps {
		layers = append(layers, convBlock(16*(steps-i)))
	}
	return &nn.Sequential(layers)
}

func main() {
	path := "filepath.csv"
	data := CSVLoader().Read(path)
	stats = Statistician{data: data}
	score, err := s.GetBestModelScore()
	if err != nil {
		log.Fatalf("couldn't get score", err)
	}
	model := Autoencoder(score)
	model.compile(nn.optimizers.Adam, nn.losses.MSE, []nn.metrics{
		nn.losses.RMSE(),
	})

	history := model.train(data, data, 5)
}
