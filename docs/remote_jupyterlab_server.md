# Running code on a remote server.

If your computer does not have a good GPU, you can install DLC and autopipy on a computer with the GPU. If you launch a jupyter lab server on the remote-GPU computer, you can then use your browser on your computer to access the jupyter lab server running on the remote-GPU computer. This is a 3-step process.


*  ssh into the computer with the GPU, activate the right conda environment and launch jupyter lab.

```
ssh a230-pc73
source .bashrc # in case conda is not in your PATH
conda activate DLC-GPU  # or source ~/python_virtual_environments/autopi37/bin/activate
cd ~/repo
jupyter lab --no-browser --allow-root
```


In the terminal, you will see an address looking like : `http://localhost:8888/?token=f74e097bf3a029eab5534bc9a17ba8fd54cdb26962ee34d8`.

Copy it.

*  In a second terminal on your local computer, run the following ssh command.
This command forwards the remote port 8888 onto our local machine’s port 8888.

``` 
ssh -N -L 8888:localhost:8888 kevin@a230-pc73
```

*  Now paste the addressed of the jupyter server in your browser. The address should start with `http://localhost:8888/?token=`

You should now have a jupyter lab page in your browser.
