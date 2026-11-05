# Changelog

Changes to the `invisensing` package, one section per version published on
PyPI. Each section lists what a script written for the previous version may
see differently, then what is new.

## 1.5.0

Follows the producer's content-type registry to revision 4: the software
front end (a band-pass at the beat frequency, then the arctan and the
magnitude, computed from the raw samples) may now feed every demodulation
method that read the card's arctan and magnitude.

### Changed — behaviour that differs from 1.4.0

- **`sw_amplitude_sac` (11) has no single method**: `content.method_of(11)`
  and `content.info(11).method` are `None` (1.4.0: `"PHASE_SAC"`), as for
  the card's `hw_amplitude_f32` (16) — the software front end's magnitude
  is written by several methods (the AMPLITUDE method on the software
  arctan streams it alone; PHASE_SAC, DUI and DUI_STATEFUL on the software
  arctan keep it beside their phase). It is a stream value too:
  `content.info(11).file_or_stream` is `"file+stream"` (1.4.0: `"file"`).
  Its unit (`front_end_count`), sample type and reading are unchanged.
- **Files of content type 29 to 32 are typed** where 1.4.0 named them
  `unknown(<n>)`, read them from their flags and warned: 29
  (`dui_arctan_phase`) and 30 (`dui_stateful_arctan_phase`) are
  `Mode.PHASE`, `float32` radians, `is_phase`; a capture of the 31 / 32
  stream is `Mode.PHASE_AMPLITUDE`, whose amplitude is the software front
  end's (`get_magnitude_volts()` refuses it, as for 15).

### Added

- **Content types 29–32** in `invisensing.content` (registry revision 4):
  `dui_arctan_phase` (29, method `DUI_ARCTAN`), `dui_stateful_arctan_phase`
  (30, `DUI_STATEFUL_ARCTAN`), and the stream-only pairs
  `interleaved_dui_arctan_phase_amplitude` (31, split into 29 + 11) and
  `interleaved_dui_stateful_arctan_phase_amplitude` (32, split into
  30 + 11). `is_phase` gives 2, 4, 9, 10, 17, 18, 19, 23, 29, 30;
  `is_stream_only` 6, 7, 8, 15, 20, 21, 22, 31, 32.

## 1.4.0

Reads the Audace data format v1: what a file holds comes from its content
type. Files written before it keep the reading 1.3.0 gave them, except for
the changes below.

### Changed — behaviour that differs from 1.3.0

- **The constant of header bit `0x020` is now `FLAG_RESERVED`** (the bit is
  never set by any producer). Its 1.3.0 name, which named a card vendor, is
  the only public name removed: a script that imports it gets an
  `ImportError` and uses `FLAG_RESERVED` instead.
- **Distances use 2·10⁸ m/s in the fibre**, the value the acquisition
  converts metres with (1.3.0: 206 856 796 m/s). `File.distance` and the
  legacy `Constants.LIGHT_SPEED_IN_FIBER` are 3.3 % smaller on every file
  that carries no spatial step (× 0.96685): 487 positions at 100 MS/s read
  487.0 m instead of 503.7 m. A file of data format v1 carries its own
  `spatial_step_m`.
- **A single column flagged `UNSIGNED` reads as unsigned integers**: a file
  read from its flags (content type 0, see below) whose column is neither
  `FLOAT` nor `INTERLEAVED` returns the unsigned integer of its width
  (`uint16` for 2-byte samples, `uint32`, `uint64`) in every format; 1.3.0
  returned signed integers, so a magnitude above 32 767 came back negative.
  Interleaved pairs are unchanged (`int16`, the magnitude lane read by
  `get_magnitude()` as `uint16`).
- **`File.dtype` is the dtype `read_lines()` returns**: `uint8` for 1-byte
  samples (1.3.0 said `int8`, or raised), `int16` for 2-byte samples flagged
  `FLOAT` (1.3.0 said `float16`).
- **Files written before v1 that carry a content type** (from June 2026,
  `format_version` 0) are classified by it, no longer by their flags:
  - the files of a split pair are no longer `RAW`: `hw_arctan` (3) is
    `Mode.ARCTAN`, `hw_amplitude` (5) `Mode.AMPLITUDE` (`uint16`), `hw_i` /
    `hw_q` (13 / 14) `Mode.I` / `Mode.Q`;
  - 5 with 4-byte samples is the `float32` magnitude (`hw_amplitude_f32`,
    `Mode.AMPLITUDE`), also without the `FLOAT` flag;
  - 8 with `INTERLEAVED` is `Mode.PHASE_AMPLITUDE`: `get_phase()` returns
    the phase member, `get_amplitude()` the amplitude (1.3.0: `Mode.PHASE`
    over both members);
  - 6 and 7 with `INTERLEAVED` stay `Mode.IQ` and `Mode.ARCTAN_MAGNITUDE`.

  Still read from their flags, exactly as 1.3.0 read them
  (`Header.mode_from_flags`): content type 0; a value this SDK does not know
  and a sample size the content type contradicts (both with a warning); a
  `sw_custom` (12) file of a Sandbox older than v1 (`format_version` 0, or
  the `PHASE` or `INTERLEAVED` bit set), which kept its input's flags; and a
  file written before v1 under a paired content type (6, 7, 8, …) without
  the `INTERLEAVED` flag — the console's archive exports, which kept their
  source's header over one column per zone (content type 8 with flags
  `0x83`: one `float32` column per zone, `Mode.PHASE`, any zone count).
- **The `is_*` properties follow the content type** on the same files:
  `is_demodulated`, `is_phase`, `is_float`, `is_unsigned`, `is_interleaved`
  (and `positions_per_line`) come from it. `is_mozart` is answered by the
  content type alone: `True` for `sw_mozart` and `sw_mozart_plus`, `None`
  for `raw` and for an unknown value, `False` otherwise. The files listed
  above as read from their flags keep the flag answers. `is_ac` and
  `is_hiz` are the flags, as before.
- **SEG-Y**: the 3200-byte textual header is read, ASCII or EBCDIC. 1.3.0
  decoded every ASCII header as EBCDIC and found none of its fields (sample
  rate, trigger frequency, range, timestamp, flags), so a SEG-Y file's
  duration, times, mode and dtype change; a `float32` phase no longer reads
  as integers. Integer samples are read at their own width (1.3.0 read
  every integer SEG-Y as 4-byte samples).
- **The card's trigger rational** is accepted when `clock / ticks` is within
  `f / q + 1 Hz` of the trigger frequency `f`, `q = ⌊clock / 10 MHz⌋`, the
  acquisition's own guard up to 10 MHz (1.3.0: within 1 % or 1 Hz). A
  recording above about 2.5 MHz is timed on its card's rate instead of the
  nominal frequency.
- **A DAT cut short in the middle of a line** warns at open
  (`UserWarning`); `File.trailing_bytes` / `DatReader.trailing_bytes` give
  the bytes that are never read.

### Added

- **Data format v1.** Content types 0–28 (`invisensing.content`: the
  registry and its helpers `info`, `quantity_of`, `members`, `member_of`,
  `sample_type`, `sample_size`, `is_phase`, `is_amplitude`,
  `is_stream_only`, `split_children`, `method_of`, `unit`, `to_physical`,
  …), checked against the producer's registry. New header fields
  `format_version`, `duration_s`, `spatial_step_m`, `temporal_step_s` on
  `Header` and `File`, and `FORMAT_VERSION`. `File.content_type`,
  `content_type_name`, `quantity`, `unit`, `source_content_type` (archive
  exports, 27 / 28) and `sibling_path` (the other file of a split pair);
  `Header.normalized_content_type`, `sample_encoding`, `mode_from_flags`,
  `sample_size_contradicts_content_type`, `derived_spatial_step_m`,
  `derived_temporal_step_s`.
- **Modes and extractors** for one quantity per file: `Mode.AMPLITUDE`,
  `ARCTAN`, `I`, `Q`, `CUSTOM`, `VARIANCE`, `LOG_VARIANCE` and
  `PHASE_AMPLITUDE`; `get_amplitude()`, `get_variance()`,
  `get_log_variance()`.
- **Card codes**: `Header.card` / `File.card` is `"p18"`, `"p21"`, `"s125"`
  or `"s500"` on every format, or `None`; the card tag a run wrote before
  v1 reads as its code.
- **`.h5`**, the extension of the HDF5 files written since v1 (`.hdf5` is
  still read).
- **`export_dat`** takes `header=` (every field of a file's header is
  copied, as the acquisition's conversions do), `content_type=`, `card=` and
  `acquisition_ns=`; with `header=` every other field is optional and
  overrides it. A content type of the registry gives a file of data format
  v1, anything else a legacy copy. `File.export_dat(path, start=, count=)`
  writes a range of a file's lines with its whole header. A DAT never
  carries an inexact trigger frequency: such a recording is refused
  (`ValueError`), keep it in HDF5. Numbers are taken with `int()` / `str()`
  as in 1.3.0 (`sample_rate=250e6` works).
- **Corrected line times** from the card's time map (HDF5
  `time_correction == "cardclock-v1"`): `File.line_time_ns(i)` gives the
  corrected time, `line_time_ns(i, nominal=True)` the count; `times_ns()`,
  `time_correction`, `time_correction_state`, `time_uncertainty_ns`,
  `trig_frequency_measured_hz`; `Header.line_offset`, `time_flags`,
  `time_corrected`, `line_timing`; `TIME_FLAG_MAP_V1`,
  `TIME_FLAG_CORRECTED`. The duration of a corrected file is measured on
  its corrected times. A file without the time map is timed exactly as
  1.3.0 timed it.
- **HDF5 compression**: `File.compression` (`"NONE"`, `"ZSTD"`,
  `"ZSTD_LOSSY_1E4"`), `quantization_step_rad` and
  `quantization_max_error_rad`.

### Packaging

- The sdist ships its tests with their small fixtures, `Cargo.lock` and
  this changelog, and no longer `.github`.
- The examples (README, documentation, `assets/basic_usage.py`) dispatch
  on `File.mode` with `if` / `elif`: they run on Python 3.9, which has no
  `match` statement.
