# Near-optimal Active Reconstruction

This repository contains the accompanying code to my [bachelor thesis](https://danielyang.de/Bachelor%20Thesis.pdf) on "Near-optimal Active Reconstruction".

<img src="https://user-images.githubusercontent.com/44120873/232784452-7d9f2d6c-19b1-49b3-a7c3-310685b45464.gif" height="500">

It consists of two interactive notebooks:
- [`2D_simulation.ipynb`](notebooks/2D_simulation.ipynb) provides an interactive environment for running and visualizing different algorithms.
- [`2D_experiments.ipynb`](notebooks/2D_experiments.ipynb) provides an interactive environment for evaluating different algorithms by running multiple simulation experiments.


## Setup

There are two options for running the interactive notebooks.

> For the best performance and compatibility, we recommend the LOCAL setup.

### LOCAL (Jupyter Notebook, VS Code with Jupyter Extension)

1. Clone this Git repository and install all requirements.
    ```
    git clone https://github.com/danielyxyang/active_reconstruction.git
    cd active_reconstruction
    pip install -r requirements.txt
    ```
2. Open the folder `active_reconstruction` in Jupyter or VS Code.
3. Open the desired notebook in the `notebooks` folder.
4. Set the `ENV` parameter in the first cell to `"LOCAL"`.
5. Run all cells.

### COLAB (Google Colab)

1. Open the desired notebook from the `notebooks` folder in Google Colab.
2. Se the `ENV` parameter in the first cell to `"COLAB"`.
3. Run all cells.

## License

This code is released under the MIT license. See [LICENSE](LICENSE) for details.