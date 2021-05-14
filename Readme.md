## Client/Server Package 

This package sends and Image from client to server, the server applies a styleGAN and returns the image to client for saving.
Currently it is designed to only accept content image, we can update it to accept a unique content and style image pair very easily.

Platform and Packages used are as listed below.
Python3 is a pre-requisite. The following packages will be required as well:
tensoflow 2 and keras is used to build and train the network
1. tf2
2.socket
3.struct
4.cv2
5.base64

Please ensure the style and content images are saved in the client working directory 
First run the server from the command line as shown below

```shell
python soc_server.py

```

Then run the client on  seprate terminal 

```shell 
python soc_client.py

```

Please ensure you are running the soc server scripts and not the others(gRPC) the gRPC files are yet to be debugged.

 