# AutoML: Time Series in Golang From Scratch

This repository represents the AutoML-Time Series division written purely in Golang from scratch. The package allows you to import any time series data you'd like and it manages to come up with a model that will be succesful at predicting what's to come next. 


```go

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
```



This code snippet is the entire workflow needed at your end. 


To follow up, we have designed a remote branch as an ECG classifier. Here is how it works! :)

```go

shape := []int{3000,1}
e := ECGClassifier()
ecgModel := e.BuildAndCompile(shape, 0.001)

```

# Contact

LinkedIn : https://www.linkedin.com/in/tim-cvetko-32842a1a6/

Medium : https://cvetko-tim.medium.com/
