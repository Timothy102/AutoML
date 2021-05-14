build:
	docker run --rm -v "$PWD":/usr/src/myapp -w /usr/src/myapp -e GOOS=windows -e GOARCH=386 golang:1.16 go build -v
	$ for GOOS in darwin linux; do
	>   for GOARCH in 386 amd64; do
	>     export GOOS GOARCH
	>     go build -v -o myapp-$GOOS-$GOARCH
	>   done
	> done