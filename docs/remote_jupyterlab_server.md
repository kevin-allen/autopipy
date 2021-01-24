# Running code on a remote server.

If your computer does not have a good GPU, you can install DLC and autopipy on the computer with the GPU. If you start the jupyter lab server, you should be able to access it from a different computer using its IP address.

First ssh into the computer with the GPU, activate the right conda environment and launch jupyter lab.

```
ssh a230-pc73 -X 
conda activate DLC-GPU
cd ~/repo
jupyter lab --no-browser
```

Open a second terminal on your local computer.
This command forwards the remote port 8888 onto our local machine’s port 8888.

``` 
ssh -N -L 8888:localhost:8888 kevin@a230-pc73
```

Then, point your browser to 127.0.0.1:8888. You will be asked for a tocken or password.
Use the token that was printed when you started jupyter lab.
