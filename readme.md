# Genetic

The entire code is based on the `testpara.csv` dataset and is divided into four parts. The first part is the `visualization.ipynb` section, which involves visual analysis of the data. The second part consists of `main_none.py` and `test_none.py`, where the input is the given six parameters, and the output is the fitness value. The third part comprises `main_copy.py` and `test_copy.py`, where the angle $\theta$ is converted to coordinate points, taking 4+4 inputs. The fourth part includes `main.py` and `test.py`, which perform positional encoding on the input.

### Training and test codes

These codes section primarily consists of two main parts. In `main.py`, `main_copy.py` and `main_none.py`, the focus is on training the MLP model and preprocessing the data (conversion to coordinates/positional encoding). I store some training parameters in a txt file and also save some images for adjusting the MLP network parameters.In `test.py`, `test_copy.py` and `test_none.py`, the trained MLP model is tested on the test dataset.

## Requirements

- python 3.9.7
- pytorch 1.13
- scikit-learn

## Usage

- Firstly, place `testpara.csv` and `main.py` in the same directory.
- Then, install the required dependencies for the program (run "python main_copy.py" and install the necessary packages based on error messages).
- Use "python main_none.py" and "python test_none.py," "python main_copy.py" and "python test_copy.py," and "python main.py" and "python test.py" for parameter tuning.
