import os
import numpy as np
import pickle as pkl


class Writer:
  """
  A class to handle VUMPS I/O. Maintains a 'console file' to store
  console output, a 'data file' to save observables to, and a
  pickle directory to save the final wavefunction.

  MEMBERS
  -------
  self.directory: Path to the output directory.
  self.pickle_directory: Path to the directory where .pkl files are saved.
  self.console_file: Path to the file where console output is saved.
  self.data_file : Path to the file where numeric data is saved.

  PUBLIC METHODS
  --------------
  console_write: Prints a string to console and then appends it to
                 self.print_file.

  """
  def __init__(self, dirpath: str,
               consolefilename="console_output.txt",
               datafilename="data.txt",
               timingfilename="timing.txt",
               data_headers=None,
               timing_headers=None):
    """
    Instantiates the writer and prepares the output directories.

    PARAMETERS
    ----------
    dirpath: Path to the directory where output is to be saved. It will
             be created if it doesn't exist. A subdirectory
             dirpath/pickles will also be created.

    consolefilename: Saves the strings fed to Writer.write().
    datafilename : Saves numeric data fed to Writer.data_write().
    timingfilename : Saves timing data fed to Writer.timing_write().
    headers  : A list of strings. Each will be written at the beginning
               of datafile as a header. For example, headers=["A", "B"]
               will result in 'datafile' beginning with
               # [0] = A
               # [1] = B
               This is meant to indicate that e.g.
               A = np.loadtxt(datafilename)[:, 0].
    """
    if not os.path.exists(dirpath):
      os.makedirs(dirpath)
    self.directory = dirpath
    self.pickle_directory = os.path.join(self.directory, "pickles")

    if not os.path.exists(self.pickle_directory):
      os.makedirs(self.pickle_directory)

    self.console_file = self._initialize_file(consolefilename)
    self.data_file = self._initialize_file(datafilename,
                                           headers=data_headers)
    self.timing_file = self._initialize_file(timingfilename,
                                             headers=timing_headers)
    self.timing_headers = timing_headers

  def _initialize_file(self, filename, headers=None):
    """
    Opens a file in the output directory with a given filename,
    and optionally writes a set of headers at the beginning. Returns
    the path to the file.
    """
    the_file = os.path.join(self.directory, filename)
    if headers is not None:
      the_header = ["# [" + str(i) + "] = " + header
                    + "\n" for i, header in enumerate(headers)]
      the_header = ''.join(the_header)
      with open(the_file, "w") as f:
        f.write(the_header)
    return the_file

  def write(self, outstring, verbose=True):
    """
    Prints a string to console and then appends it, along with a newline,
    to self.consolefile. If verbose is False, saves to self.consolefile
    without printing to console.
    """
    if verbose:
      print(outstring)
    with open(self.console_file, "a+") as f:
      f.write(outstring+"\n")

  def data_write(self, data):
    """
    Appends the data in the array 'data' to self.datafile. data should
    represent a row of that file, e.g. each computed observable
    at a given timestep in order.
    """
    to_write = data.reshape((1, data.size))
    with open(self.data_file, "ab") as f:
      np.savetxt(f, to_write)

  def timing_write(self, Niter, timing_data):
    """
    Write the entries in timing_data matching self.timing_headers
    as a row to the timing file in the appropriate order.
    """
    if self.timing_headers is None:
      raise ValueError("No timing headers specified.")
    out = [Niter]
    for key in self.timing_headers:
      if key != "N":
        try:
          out.append(timing_data[key])
        except KeyError:
          self.write("Warning: key '"+key+"' wasn't in timing dict.")
    out = np.array(out)
    to_write = out.reshape((1, out.size))
    with open(self.timing_file, "ab") as f:
      np.savetxt(f, to_write)

  def pickle(self, to_pickle, timestep: int, name=None):
    """
    Pickles the data in to_pickle under the name
    self.pickle_directory/name_t{timestep}.pkl.
    """
    if name is not None:
      fend = name + "_t" + str(timestep) + ".pkl"
    else:
      fend = "_t" + str(timestep) + ".pkl"
    fname = os.path.join(self.pickle_directory, fend)
    self.write("Pickling to " + fname)
    with open(fname, "wb") as f:
      pkl.dump(to_pickle, f)
