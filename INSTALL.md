
# Install
1. Clone the project

    ```Shell
    git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection
    cd Ultra-Fast-Lane-Detection
    ```

2. Create a conda virtual environment and activate it

    ```Shell
    conda create -n line-det python=3.7 -y
    conda activate line-det
    ```

3. Install dependencies

    ```Shell
    # If you dont have pytorch
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch 

    pip install -r requirements.txt
    ```



