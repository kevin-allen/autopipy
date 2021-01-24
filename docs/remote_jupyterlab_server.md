# Running code on a remote server.

If your computer does not have a good GPU, you can install DLC and autopipy on a computer with the GPU. If you launch a jupyter lab server on the remote-GPU computer, you can then use your browser on your computer to access the jupyter lab server running on the remote-GPU computer. This is a 2-step process.


First ssh into the computer with the GPU, activate the right conda environment and launch jupyter lab.

```
ssh a230-pc73 -X 
conda activate DLC-GPU
cd ~/repo
jupyter lab --no-browser
```

In a second terminal on your local computer, run the following ssh command.
This command forwards the remote port 8888 onto our local machine’s port 8888.

``` 
ssh -N -L 8888:localhost:8888 kevin@a230-pc73
```

Then, point your browser to 127.0.0.1:8888. You will be asked for a tocken or password.
Use the token that was printed when you started jupyter lab.
