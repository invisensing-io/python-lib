##
# @file File.py
# @brief This file contains the implementation of the File class.
# @date 2024-09-13
# @section libraries_main Libraries/Modules
# - struct standard library (https://docs.python.org/3/library/struct.html#module-struct)
#   - Access to method `struct.unpack`.
# - numpy standard library (https://numpy.org/)
#   - Access to numpy module for data manipulation.
# - os standard library (https://docs.python.org/3/library/os.html#module-os)
#   - Access to os module for get size of file.
# - IntEnum standard library (https://docs.python.org/3/library/enum.html#intenum)
#   - Access to IntEnum class for enumerated types.
#
# Copyright (c) 2024 Invisensing.  All rights reserved.

# Imports
import struct
import numpy as np
import os
from enum import IntEnum

## Global Constants
#    Constants used in the application.
#    This class contains global constants, including physical constants.
#
class Constants:
    """
    Constants used in the application.
    This class contains global constants, including physical constants.
    """
    LIGHT_SPEED_IN_FIBER = 206856796  # m/s

HEADER_SIZE = 128
HEADER_FORMAT = '32siiiiii72x'

class h(IntEnum):
    TIMESTAMP = 0
    LINE_SIZE = 1
    TRIG_FREQUENCY = 2
    SAMPLE_SIZE = 3
    SAMPLE_RATE = 4
    FLAGS = 5
    RANGE = 6

class hflags(IntEnum):
    H_DEMODULATED   = 0x1     # True if data is demodulated
    H_FLOAT         = 0x2     # True if data is in float format
    H_AC            = 0x4     # True if AC coupling was used by the DAQ
    H_HIZ           = 0x8     # True if High Impedance was used by the DAQ

class File:

    def __init__(self, filename):
        """
        Initializes a new file
        Raises an OSError if the file couldn't be opened
        """
        self.__filename = filename
        self.__handle = open(filename, "rb")
        header_data = self.__handle.read(HEADER_SIZE)
        if (not header_data):
            return None
        header = struct.unpack(HEADER_FORMAT, header_data)
        self.__timestamp = header[h.TIMESTAMP]
        self.__line_size = header[h.LINE_SIZE]
        self.__trigger_frequency = header[h.TRIG_FREQUENCY]
        self.__sample_size = header[h.SAMPLE_SIZE]
        self.__sample_rate = header[h.SAMPLE_RATE]
        self.__flags = header[h.FLAGS]
        self.__range = header[h.RANGE]
        self.__decoded_timestamp = self.__timestamp.decode('utf-8').strip('\x00')
        self.__file_size = os.path.getsize(filename)
        self.__num_lines = (self.__file_size - HEADER_SIZE) // (self.__line_size * self.__sample_size)
        self.__lines_left = self.__num_lines
        dtype = 'float' if self.__flags & hflags.H_FLOAT else 'int'
        dsize = self.__sample_size * 8
        self.__dtype = eval(f'np.{dtype}{dsize}')

    def __del__(self):
        self.__handle.close()

    def get_distance(self) -> float:
        """
        Returns the physical distance, in meters, covered by one line of data
        """
        return (Constants.LIGHT_SPEED_IN_FIBER * self.__line_size) / (2 * self.__sample_rate)

    def get_line_size(self) -> int:
        """
        Returns how many samples are on a line
        """
        return self.__line_size

    def get_trigger_frequency(self) -> int:
        """
        Returns the number of lines acquired per second
        """
        return self.__trigger_frequency
    
    def get_data_type(self) -> type:
        """
        Returns the data type of the samples
        """
        return self.__dtype

    def get_sample_rate(self) -> int:
        """
        Returns the sample rate of the DAQ, in Hz
        """
        return self.__sample_rate
    
    def get_range(self) -> float:
        """
        Returns the voltage range in volts, range is between -range and range
        """
        return self.__range / 1000

    def get_timestamp(self) -> str:
        """
        Returns the timestamp of the file
        """
        return self.__timestamp

    def is_demodulated(self) -> bool:
        """
        Returns true if the data is demodulated
        """
        return self.__flags & hflags.H_DEMODULATED

    def is_acquisition_ac(self) -> bool:
        """
        Returns true if the DAQ's coupling was in AC mode, false if it was in DC
        """
        return self.__flags & hflags.H_AC
    
    def is_acquisition_hiz(self) -> bool:
        """
        Returns true if the DAQ acquired data in high impedance mode
        """
        return self.__flags & hflags.H_HIZ

    def get_lines(self, n = 1) -> np.ndarray:
        """
        Returns n lines of data as a numpy array of shape [n, line_size]
        If less than n lines are left in the file, returns all the lines left
        If the file is empty, raises an IOError
        If n <= 0, raises an ValueError
        """
        if (n <= 0):
            raise ValueError('n must be a positive integer')
        if self.__lines_left == 0:
            raise OSError('End of file')
        n = min(n, self.__lines_left)
        self.__lines_left -= n
        return np.frombuffer(
            self.__handle.read(self.__line_size * self.__sample_size * n),
            dtype = self.get_data_type()
        ).reshape([n, self.__line_size])

    def get_num_lines(self) -> int:
        """
        Returns how much lines of data are in the file
        """
        return self.__num_lines

    def get_lines_left(self) -> int:
        """
        Returns how much lines have not been read
        """
        return self.__lines_left

    def get_duration(self) -> float:
        """
        Returns how much time, in seconds, was recorded in the file
        """
        return self.__num_lines / self.__trigger_frequency
    
def export(filename,
           data: np.ndarray, 
           timestamp, 
           trigger_frequency: int, 
           sample_rate: int,
           range: float,
           is_demodulated = True,
           is_ac = True,
           is_hiz = True
           ):
    """
    Exports a numpy array to a file
    Raises OSError if writing fails
    """
    flags = 0
    flags |= is_demodulated * hflags.H_DEMODULATED
    flags |= hflags.H_FLOAT if 'float' in str(data.dtype) else 0
    if (is_ac):
        flags |= hflags.H_AC
    if (is_hiz):
        flags |= hflags.H_HIZ
    header = struct.pack(HEADER_FORMAT,
                         timestamp,
                         data.shape[1],
                         trigger_frequency,
                         data.itemsize,
                         sample_rate,
                         flags,
                         int(range * 1000))
    fileout = open(filename, 'wb')
    fileout.write(header)
    fileout.write(data.tobytes())
    fileout.close()
