
# Install
1. Create a conda virtual environment and activate it

    ```Shell
    conda create -n line-det python=3.7 -y
    conda activate line-det
    ```

2. Install dependencies

    ```Shell
    # If you dont have pytorch
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch 

    pip install -r requirements.txt
    ```



