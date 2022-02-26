import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

class Memorize:

  def __init__(self, d, level):

    self.d = d
    self.level = level

  def _generate_coeffs(self, l):

    if l == 0:

      return 1

    return -1*((self.d - l + 1) / l) * self._generate_coeffs(l-1)

  def x_tilda(self, df, col):

    coefficients = []

    for i in range(0, self.level):

      coefficients.append(self._generate_coeffs(i))

    df[f'x_tilda_{col}'] = None

    for i in range(self.level, len(df) + 1):

      x_vector = df[col][i-self.level:i]
      coeff_vector = np.array(coefficients)
      result = np.dot(x_vector[::-1], coeff_vector)
      df.iloc[i-1,-1] = result

    return df.dropna(axis=0)
