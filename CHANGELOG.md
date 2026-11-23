# Changelog

Changes to the `invisensing` package, one section per version published on
PyPI. Each section lists what a script written for the previous version may
see differently, then what is new.

## Unreleased

Reads the node's clock at the run's START, which every HDF5 file of the
Audace FileWriter records since 2026-11-22 (Audace-Lib
`docs/DATA_FORMAT.md` §8.7).

Nothing a 1.7.0 script reads changes: the additions are new names.

### Added

- **`File.host_clock_synchronized`** (`HostClockSync`: `SYNCHRONIZED`,
  `NOT_SYNCHRONIZED`, `UNKNOWN`) and **`File.start_time_guard`**
  (`StartTimeGuard`: `OPEN`, `WAIVED`, `LIFTED`, `LIFTED_PERMANENT`,
  `UNKNOWN`; `.forced` is `True` for a START forced past a lifted time
  guard). `str` enums, equal to the stored tags; `None` on a file that does
  not record them (older HDF5, DAT, SEG-Y, TDMS). An unknown tag reads
  `UNKNOWN`.

### Fixed

- The test of the arctan's range compared float32 samples with the float64
  π: the raw −32768 is float32(π), one ulp above it. It now compares with π
  as float32 holds it (test only, no library change).

## 1.7.0

Reads the speed of light in the fibre a recording was acquired at. The
Audace acquisition now takes the fibre's group index `n` as a parameter
(default 1.4682, Corning SMF-28e+ at 1550 nm) and turns time into distance
at `v = round(c / n)` m/s instead of the fixed 2·10⁸ m/s; it writes that
speed into every file (Audace-Lib a288414, `docs/DATA_FORMAT.md` §3.5).
Every metre this SDK reports follows the file's speed. A recording made
before the field carries none and was acquired at 2·10⁸ m/s (index
1.49896229): it reads exactly as with 1.6.0, bit for bit.

It also reads the metadata of two new kinds of recordings: files shorter
than one second (file durations with decimals, e.g. 0.1 s) and the files of
the analog-trigger recording mode.

### Changed — behaviour that differs from 1.6.0

- **`File.distance`, `File.spatial_step_m` and `File.gauge_length_m` use
  the file's speed**: `v / (2 · sample_rate)` per position. On a recording
  at 1.4682 they are 2.0952 % longer than at 2·10⁸ m/s (0.408381 m per
  position at 250 MS/s instead of 0.4 m). `Header.derived_spatial_step_m`
  and the exact read-back of a DAT's float32 gauge (`k · v / (2 ·
  sample_rate)`) follow it too. Files without a speed: unchanged.
- **`Header` has two more fields**: `light_speed_in_fiber_m_s` (u32, m/s,
  as stored; `0` when absent) and `fiber_group_index` (float, the index
  exactly as configured when the file stores it — HDF5, TDMS, SEG-Y; `0.0`
  otherwise). `Header(...)` takes both as keywords (defaults `0` and
  `0.0`), `repr(header)` shows them when they are not 0, and
  `HEADER_LAYOUT` lists the speed (bytes 124..128, u32).
- **DAT exports write header bytes 124..128**: `File.export_dat()` and
  `export_dat(header=…)` keep the header's speed (0, absent, for a file
  without one). 1.6.0 always wrote 0 there.
- **SEG-Y**: the line `FIBER GROUP INDEX: <n> (LIGHT SPEED <v> M/S)` that
  Audace-Lib writes after the gauge is read.
- `File` warns, once at open, when a header's bytes 124..128 hold a value
  that is not the speed of an index from 1 to 100; its metres are then read
  at 2·10⁸ m/s.
- `repr(File)` names the index when the file carries a speed (or is
  relabelled); unchanged for every other file.
- The registry copy (`tests/assets/content_types_v1.json`) follows the
  producer's revision 5: the header rows 120 (`gauge_length_m`) and 124
  (`light_speed_in_fiber_m_s`), and the key `light_speed_in_fiber_m_s` is
  replaced by `speed_of_light_m_s`, `legacy_light_speed_in_fiber_m_s`,
  `legacy_fiber_group_index`, `default_fiber_group_index`,
  `fiber_group_index_min` and `fiber_group_index_max`.
- **`File.duration_s` reads `None` on every file shorter than one second**
  (the header floors to whole seconds and 0 means unknown): use
  `File.duration`, measured from the lines, or `File.duration_exact_s`.

### Added

- **`File.light_speed_in_fiber_m_s`** (int, m/s) and
  **`File.fiber_group_index`** (float): the speed every metre is labelled
  with and its index — the exact configured index where the file stores it
  (HDF5, TDMS, SEG-Y), else the index of the speed: 1.4682 for
  204 190 477 m/s, the legacy 1.49896229 for 2·10⁸ m/s (and for a file
  without a speed), `c / v` otherwise.
- **`File(path, fiber_group_index=n)`**: labels a recording at the group
  index `n` — typically an old recording whose fibre is known
  (`fiber_group_index=1.4682`). Only labels change, never the samples: the
  speed becomes `round(c / n)`, the spatial step, the gauge length and the
  trigger pre-roll are converted to it (`Header.with_fiber_group_index`),
  and with the gauge the strain factors; `File.header` is the relabelled
  header, so an export carries the new speed.
  `File.recorded_fiber_group_index` and
  `File.recorded_light_speed_in_fiber_m_s` keep the file's own. Any number
  from 1 to 100 is accepted; anything below 1, above 100, NaN or infinite
  raises `ValueError`, a value that is not a number `TypeError`.
- `export_dat(…, fiber_group_index=n)`: stamps the speed of `n` (a new
  file's steps follow it; with `header=` the header is relabelled first).
- `light_speed_of_group_index(n)` (`round(c / n)`, half away from zero, the
  acquisition's derivation), `group_index_of_light_speed(v)`, and the
  constants `SPEED_OF_LIGHT_M_S`, `DEFAULT_FIBER_GROUP_INDEX` (1.4682),
  `LEGACY_FIBER_GROUP_INDEX` (1.49896229), `LEGACY_LIGHT_SPEED_IN_FIBER_M_S`
  (200 000 000), `FIBER_GROUP_INDEX_MIN` (1) and `FIBER_GROUP_INDEX_MAX`
  (100). `Header.light_speed_m_s` and `Header.group_index` give the speed
  and the index a header's metres use.
- `File.duration_exact_s`: the exact duration of the file's lines, from
  the HDF5 attribute `duration_exact_s` (sub-second and analog-trigger
  files; `None` otherwise).
- `File.first_line`: the card-line index of the file's first line (HDF5
  attribute `first_line`, same files).
- `File.analog_trigger`: an `AnalogTrigger` record (option `rising_edge` or
  `above_threshold`, threshold in mV and counts, CH2 full scale, signed
  offset, hold, event and part numbers, trigger line, sample and time, peak,
  start line and pre-trigger lines, why the start was truncated, ignored
  crossings, and on the event's last file its end line and reason), read
  from the `analog_trigger_*` attributes of a file whose `recording_mode` is
  `analog_trigger`; `None` for any other file. `File.analog_trigger_index`
  is the trigger line's index in the file. CH2, the trigger input, is never
  recorded.

### Unchanged

- `NANOSTRAIN_PER_RAD_METRE` stays 110.37: it uses the fibre core's phase
  index (n = 1.46) and the photo-elastic term, not the group index, and does
  not follow it. A strain still divides by the file's gauge length, which
  is a distance and follows the file's speed.
- The legacy `invisensing.File.Constants.LIGHT_SPEED_IN_FIBER` stays
  200 000 000: the legacy speed, what every recording made before the field
  was acquired at.

## 1.6.0

Reads the gauge length a producer applied to a phase, and can read a
phase as strain, in nm/m, or as strain rate, in nm/m/s. Audace writes the
gauge length into a phase file only from the release that adds it: strain
and strain rate need a recording made with that release or a later one,
and refuse every earlier recording.

### Changed — behaviour that differs from 1.5.0

- **`Header` has a `gauge_length_m` field** (metres, float; `0.0` when
  absent): `Header(...)` takes it as a keyword (default `0.0`),
  `repr(header)` shows it when it is not 0, and `HEADER_LAYOUT` lists it
  (bytes 120..124, float32). A DAT's float32 is read back as the float64
  the other formats store (`k · 1e8 / sample_rate` for a whole `k`).
- **DAT exports write header bytes 120..124**: `File.export_dat()` and
  `export_dat(header=…)` keep the gauge length of a phase content type and
  write 0 for every other (the amplitude of a split pair, a raw file with a
  stray value, a legacy copy, pairs), as the acquisition's writers do.
  1.5.0 always wrote 0 there. Files without a gauge export as before.
- **SEG-Y**: the `C25 GAUGE LENGTH: <m> METRES` line Audace-Lib writes on a
  phase is read; the NOTE lines that follow it are read as before.

### Added

- **`File.gauge_length_m`**: the gauge length the producer applied to the
  phase, in metres — DAT header bytes 120..124, HDF5 attribute and TDMS
  property `gauge_length_m`, SEG-Y line `GAUGE LENGTH` — or `None` (every
  recording made before the Audace release that writes it, a MOZART /
  MOZART_PLUS phase, a non-phase file, a file read from its flags).
- **`File(path, phase_unit="nanostrain")`**: the phase is read as strain,
  in nm/m, `phase_rad / gauge_length_m * NANOSTRAIN_PER_RAD_METRE`, by every
  read path (`read_lines`, `read_all`, iteration, `get_phase`, `channels`,
  `get_lines`). The factor is computed once in float64 and rounded once to
  float32; each phase sample takes one float32 multiply by it (the dtype
  stays `float32`). On phase/amplitude pairs only the phases are
  converted. `File.unit` is `"nm/m"`, `File.quantity` `"strain"`;
  `File.mode` stays `Mode.PHASE`. `export_dat()` still writes radians.
  `phase_unit="rad"` (the default) reads every file as before.
- Refused at open with a `ValueError` naming the content type: a file that
  holds no phase, and a phase that does not record its gauge length. No
  gauge length is ever assumed.
- **`NANOSTRAIN_PER_RAD_METRE`** = 110.37 nm/m per rad/m, from the DAS
  relation `ΔΦ = 2β(n + Cε)·ε·L` (β = 2π/λ) at λ = 1550.12 nm and a core index
  n = 1.46.
- `File.phase_unit`, `File.nanostrain_per_rad` (the float32 factor of a
  file with a gauge), `export_dat(gauge_length_m=…)` (a phase content type
  only).
- The conversion runs in the native core, in place, with AVX-512F or AVX2
  kernels chosen at run time, each bit-identical to the scalar loop:
  `invisensing._core.simd_level()` names the kernel in use,
  `INVISENSING_SIMD=scalar|avx2|avx512` caps it.
- `quantization_step_rad` and `quantization_max_error_rad` stay in
  radians.
- **`File(path, phase_unit="strain_rate")`**: the phase is read as strain
  rate, in nm/m/s — the strain increment from the previous line times the
  line rate: `strain_rate[n] = (strain[n] − strain[n−1]) ×
  trig_frequency_real_hz` = `(phase[n] − phase[n−1]) ×
  NANOSTRAIN_PER_RAD_METRE / gauge_length_m × trig_frequency_real_hz`
  (the real rate the card fired at; the requested one on a file that does
  not carry the exact clock). The factor is computed once in float64 and
  rounded once to float32; each phase sample takes one float32 subtraction
  of the previous line's stored phase, then one multiply. Every read path
  honours it and carries the previous line across reads: reading in blocks
  of any size gives the bits of `read_all()`, and a read after
  `seek_lines(k)` reads line `k − 1` first. **Line 0 reads NaN** (no
  previous line; the previous file of a recording is never opened), and so
  does every line that follows a **gap of the time axis** (`File.time_gaps`:
  shots the card lost, on an HDF5 stamped from the card's time map) — an
  increment over more than one line period is never divided by one. On
  pairs only the phase lanes are converted (the amplitude keeps its bits).
  `File.unit` is `"nm/m/s"`, `File.quantity` `"strain rate"`. Refused at
  open like `"nanostrain"`, and when the trigger frequency is unknown (0).
- `File.strain_rate_per_rad` (the float32 factor) and `File.time_gaps`
  (`(first, end)` ranges of the lines more than 1.5 line periods after the
  previous one on the corrected time axis; `()` for a file timed by its
  count).
- The strain rate runs in the native core, in place, in one pass, with the
  same AVX-512F / AVX2 / scalar kernels choice, each bit-identical to the
  scalar loop (no FMA, no reassociation): `invisensing._core.
  strain_rate_factor()` and `phase_to_strain_rate()`.

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
