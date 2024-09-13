import invisensing.File as iFile
import numpy as np
import os
import pytest
import filecmp

def test_open_file():
    file = iFile.File('tests/assets/demodulated.dat')
    assert file.get_data_type() == np.float32
    assert file.get_line_size() == 487
    assert file.get_trigger_frequency() == 1000
    assert file.get_sample_rate() == 100000000
    assert file.is_acquisition_ac()
    assert file.is_demodulated()
    assert not file.is_acquisition_hiz()
    assert file.get_data_type() == np.float32
    assert file.get_duration() == 10
    assert 490 < file.get_distance() < 510
    assert file.get_num_lines() == 10000
    assert file.get_range() == 2

def test_read_file():
    file = iFile.File('tests/assets/demodulated.dat')
    data = file.get_lines()
    assert file.get_lines_left() == 9999
    assert data.shape == (1, 487)
    assert data.any()
    data = file.get_lines(50)
    assert file.get_lines_left() == 9949
    assert data.shape == (50, 487)
    assert data.any()
    data = file.get_lines(9940)
    assert file.get_lines_left() == 9
    assert data.shape == (9940, 487)
    assert data.any()
    data = file.get_lines(50)
    assert file.get_lines_left() == 0
    assert data.shape == (9, 487)
    assert data.any()
    with pytest.raises(OSError) as err:
        data = file.get_lines()
    with pytest.raises(ValueError) as err:
        data = file.get_lines(-1)

def test_export_file():
    file = iFile.File('tests/assets/demodulated.dat')
    data = file.get_lines(10000)
    try:
        os.mkdir('tests/output')
    except OSError:
        pass
    iFile.export('tests/output/output.dat',
                data,
                file.get_timestamp(),
                file.get_trigger_frequency(),
                file.get_sample_rate(),
                file.get_range(),
                file.is_demodulated(),
                file.is_acquisition_ac(),
                file.is_acquisition_hiz())
    assert filecmp.cmp('tests/assets/demodulated.dat', 'tests/output/output.dat', shallow = False)