//! Generator of the data format v1 fixtures beside it (`expected.json` and
//! the files it lists), written by Audace-Lib's own writers — the path the
//! acquisition's recorder and the console take: a stream header as a
//! producer stamps it (`stamp_v1`, card code), turned into file headers by
//! `file_layout` (split children) / `to_file_header`, then written by
//! `audace::formats`. Also legacy files (content type 0, and a pre-v1 file
//! carrying an interleaved value) for the legacy reading path.
//!
//! Not part of the package: it builds against an Audace-Lib checkout, as
//! `src/main.rs` of a crate whose `Cargo.toml` holds
//!
//! ```toml
//! [dependencies]
//! audace = { path = "<Audace-Lib>", features = ["formats"] }
//! hdf5 = { package = "hdf5-metno", version = "0.15" }
//! serde_json = "1"
//! sha2 = "0.10"
//! ```
//!
//! built with `AUDACE_LIB=<Audace-Lib>` in the environment and
//! `<Audace-Lib>/SNAPSHOT_COMMIT` holding the commit it was built from
//! (`expected.json`'s `audace_lib_commit`), then run with the output
//! directory as its argument. It also writes a pre-v1 two-card DAT
//! (`legacy/v0_ct4_dual_card.dat`), which is not kept here: its slot holds
//! a legacy card tag; the tests build that case from the registry instead.

use std::path::{Path, PathBuf};

use audace::formats;
use audace::header::{self as hd, FileLayout};
use audace::types::{header_flags as hf, CardCode, ContentType, DataHeader, FileCompression, FileFormat};
use audace::TriggerTiming;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const ANCHOR_NS: i64 = 1_790_000_000_123_456_789;

fn stream_header(ct: ContentType, line_size: i32, sample_size: i32, flags: i32) -> DataHeader {
    let mut h = DataHeader::default();
    h.line_size = line_size;
    h.sample_size = sample_size;
    h.flags = flags;
    h.content_type = ct;
    h.content_type_raw = ct as u8;
    h.sample_rate = 62_500_000; // p21 Phase at downsampling 4: 1.6 m
    h.trig_frequency = 50;
    h.range = 1000;
    h.pulse_width = 100;
    h.num_channels = 1;
    h.trigger_delay = 120;
    hd::set_acquisition_ns(&mut h, ANCHOR_NS);
    hd::set_trig_timing(&mut h, TriggerTiming::from_u64(250_000_000, 5_000_000));
    hd::set_card_code(&mut h, Some(CardCode::P21));
    hd::stamp_v1(&mut h);
    h
}

fn sha(bytes: &[u8]) -> String {
    let mut s = Sha256::new();
    s.update(bytes);
    s.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn ext(fmt: FileFormat) -> &'static str {
    match fmt {
        FileFormat::HDF5 => "h5",
        FileFormat::SEGY => "sgy",
        FileFormat::TDMS => "tdms",
        _ => "dat",
    }
}

/// De-interleave `[m0, m1, …]` pairs of `size`-byte samples.
fn member(data: &[u8], size: usize, which: usize) -> Vec<u8> {
    data.chunks_exact(2 * size).flat_map(|p| p[which * size..(which + 1) * size].to_vec()).collect()
}

struct Out {
    root: PathBuf,
    entries: Vec<Value>,
}

impl Out {
    fn write(&mut self, rel: &str, fmt: FileFormat, h: &DataHeader, data: &[u8], extra: Value) {
        let path = self.root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        formats::write(fmt, &path, h, data).unwrap_or_else(|e| panic!("{rel}: {e}"));
        self.record(rel, fmt, h, data, extra);
    }

    fn record(&mut self, rel: &str, fmt: FileFormat, h: &DataHeader, data: &[u8], extra: Value) {
        let lines = hd::num_lines(h, data.len() as u64);
        let enc = hd::sample_encoding(h);
        let mut e = json!({
            "path": rel,
            "format": ext(fmt),
            "content_type": h.content_type_byte(),
            "content_type_name": h.content_type.as_tag(),
            "format_version": h.format_version,
            "flags": h.flags,
            "line_size": h.line_size,
            "sample_size": h.sample_size,
            "sample_rate": h.sample_rate,
            "trig_frequency": h.trig_frequency,
            "trigger_delay": h.trigger_delay,
            "positions_per_line": hd::positions_per_line(h),
            "num_lines": lines,
            "duration_s": hd::duration_s_for_lines(h, lines),
            "spatial_step_m": hd::exact_spatial_step_m(h),
            "temporal_step_s": hd::exact_temporal_step_s(h),
            "spatial_step_m_f32": h.spatial_step_m,
            "temporal_step_s_f32": h.temporal_step_s,
            "card": hd::card_code(h).map(|c| c.as_code()),
            "acquisition_ns": hd::acquisition_ns(h),
            "trig_clock_hz": h.trig_clock_hz,
            "trig_period_ticks": h.trig_period_ticks,
            "encoding": {"bytes": enc.bytes, "float": enc.float, "unsigned": enc.unsigned},
            "data_sha256": sha(data),
        });
        if let (Value::Object(m), Value::Object(x)) = (&mut e, extra) {
            m.extend(x);
        }
        self.entries.push(e);
    }
}

fn f32_bytes(v: impl IntoIterator<Item = f32>) -> Vec<u8> {
    v.into_iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn main() {
    let root = PathBuf::from(std::env::args().nth(1).expect("usage: gen-v1-fixtures <out_dir>"));
    std::fs::create_dir_all(&root).unwrap();
    let mut out = Out { root: root.clone(), entries: Vec::new() };
    let positions: usize = 6;
    let lines: usize = 120; // 2.4 s at 50 Hz -> duration_s 2

    // ── DUI_COHERENT with amplitude kept (22): phase/ (19) + amplitude/ (16) ──
    let flags22 = hf::DEMODULATED | hf::FLOAT | hf::PHASE | hf::INTERLEAVED;
    let s22 = stream_header(ContentType::InterleavedDuiCoherentPhaseAmplitude, 2 * positions as i32, 4, flags22);
    let pairs22 = f32_bytes((0..lines * positions).flat_map(|k| {
        let phase = (k as f32) * 0.001 - 0.75;
        let amp = 1000.0 + (k % 97) as f32 * 0.5;
        [phase, amp]
    }));
    let FileLayout::Split(children) = hd::file_layout(&s22).unwrap() else { panic!("22 splits") };
    for fmt in [FileFormat::DAT, FileFormat::HDF5] {
        for (i, c) in children.iter().enumerate() {
            let data = member(&pairs22, 4, i);
            let rel = format!("split_{}/{}/2026-11-03_10-00-00.{}", ext(fmt), c.subdir, ext(fmt));
            out.write(&rel, fmt, &c.header, &data, json!({"mode": if i == 0 {"phase"} else {"amplitude"},
                "sibling": format!("split_{}/{}/2026-11-03_10-00-00.{}", ext(fmt), children[1 - i].subdir, ext(fmt))}));
        }
    }

    // ── Card arctan/magnitude tap (7): arctan/ (3, i16) + magnitude/ (5, u16) ──
    let flags7 = hf::DEMODULATED | hf::INTERLEAVED | hf::UNSIGNED;
    let s7 = stream_header(ContentType::InterleavedHwArctanAmplitude, 2 * positions as i32, 2, flags7);
    let pairs7: Vec<u8> = (0..lines * positions)
        .flat_map(|k| {
            let arctan = ((k as i32 * 977) % 65536 - 32768) as i16;
            let mag = (30_000 + (k * 131) % 35_000) as u16; // crosses 32767
            let mut b = arctan.to_le_bytes().to_vec();
            b.extend_from_slice(&mag.to_le_bytes());
            b
        })
        .collect();
    let FileLayout::Split(children7) = hd::file_layout(&s7).unwrap() else { panic!("7 splits") };
    for fmt in [FileFormat::DAT, FileFormat::HDF5, FileFormat::SEGY, FileFormat::TDMS] {
        for (i, c) in children7.iter().enumerate() {
            if i == 0 && matches!(fmt, FileFormat::SEGY | FileFormat::TDMS) {
                continue; // the magnitude member is the interesting one there (u16)
            }
            let data = member(&pairs7, 2, i);
            let rel = format!("arctan_magnitude_{}/{}/2026-11-03_10-00-00.{}", ext(fmt), c.subdir, ext(fmt));
            out.write(&rel, fmt, &c.header, &data, json!({"mode": if i == 0 {"arctan"} else {"amplitude"}}));
        }
    }

    // ── SEG-Y and TDMS of a phase column (19) ──
    for fmt in [FileFormat::SEGY, FileFormat::TDMS] {
        let c = &children[0];
        let data = member(&pairs22, 4, 0);
        let rel = format!("phase_{}/2026-11-03_10-00-00.{}", ext(fmt), ext(fmt));
        out.write(&rel, fmt, &c.header, &data, json!({"mode": "phase"}));
    }

    // ── Raw ADC on s500 (1, i16), AC + HIZ, 0.2 m ──
    let mut sraw = stream_header(ContentType::Raw, positions as i32, 2, hf::AC | hf::HIZ);
    sraw.sample_rate = 500_000_000;
    hd::set_card_code(&mut sraw, Some(CardCode::S500));
    sraw.spatial_step_m = 0.0;
    hd::stamp_v1(&mut sraw);
    let raw: Vec<u8> = (0..lines * positions).flat_map(|k| ((k as i32 % 4000) as i16 - 2000).to_le_bytes()).collect();
    let FileLayout::Single(hraw) = hd::file_layout(&sraw).unwrap() else { panic!("raw is single") };
    out.write("raw/2026-11-03_10-00-00.dat", FileFormat::DAT, &hraw, &raw, json!({"mode": "raw"}));
    out.write("raw/2026-11-03_10-00-00.h5", FileFormat::HDF5, &hraw, &raw, json!({"mode": "raw"}));

    // ── Card I/Q tap (6): i_component/ (13) + q_component/ (14) ──
    let s6 = stream_header(ContentType::InterleavedHwIq, 2 * positions as i32, 2, hf::DEMODULATED | hf::INTERLEAVED);
    let pairs6: Vec<u8> = (0..2 * lines * positions).flat_map(|k| ((k as i32 * 37) % 30000 - 15000).to_le_bytes()[..2].to_vec()).collect();
    let FileLayout::Split(children6) = hd::file_layout(&s6).unwrap() else { panic!("6 splits") };
    for (i, c) in children6.iter().enumerate() {
        let data = member(&pairs6, 2, i);
        let rel = format!("iq_dat/{}/2026-11-03_10-00-00.dat", c.subdir);
        out.write(&rel, FileFormat::DAT, &c.header, &data, json!({"mode": if i == 0 {"i"} else {"q"}}));
    }

    // ── An inexact cadence (75 kHz on 250 MHz): HDF5 only; exact f64 step ──
    let mut s19 = stream_header(ContentType::DuiCoherentPhase, positions as i32, 4, hf::DEMODULATED | hf::FLOAT | hf::PHASE);
    s19.trig_frequency = 75_000;
    hd::set_trig_timing(&mut s19, TriggerTiming::from_u64(250_000_000, 3333));
    s19.temporal_step_s = 0.0;
    hd::stamp_v1(&mut s19);
    let h19 = hd::to_file_header(&s19).unwrap();
    let data19 = f32_bytes((0..120 * positions).map(|k| (k as f32).sin()));
    out.write("inexact/2026-11-03_10-00-00.h5", FileFormat::HDF5, &h19, &data19, json!({"mode": "phase"}));
    // The same, ZSTD-compressed through the streaming writer.
    {
        let rel = "inexact/2026-11-03_10-00-00_zstd.h5";
        let path = root.join(rel);
        let mut w = formats::open_stream_writer_with(FileFormat::HDF5, &path, FileCompression::ZSTD).unwrap();
        w.begin(&h19, Some(120)).unwrap();
        w.write_block(0, &data19).unwrap();
        w.finish().unwrap();
        out.record(rel, FileFormat::HDF5, &h19, &data19, json!({"mode": "phase", "compression": "ZSTD"}));
    }

    // ── Archive exports: 27 viz_variance and 28 viz_logvar (zone spacing, row period) ──
    for (ct, name) in [(ContentType::VizVariance, "variance"), (ContentType::VizLogvar, "log_variance")] {
        let mut h = DataHeader::default();
        h.content_type = ct;
        h.content_type_raw = ct as u8;
        h.line_size = 5;
        h.sample_size = 4;
        h.sample_rate = 62_500_000;
        h.trig_frequency = 4;
        h.range = 1000;
        hd::set_acquisition_ns(&mut h, ANCHOR_NS);
        hd::set_card_code(&mut h, Some(CardCode::P21));
        h.spatial_step_m = 10.0;
        h.temporal_step_s = 0.25;
        let h = hd::to_file_header(&h).unwrap();
        let variance: Vec<f32> = (0..40 * 5).map(|k| (k as f32) * 0.5).collect();
        let values: Vec<f32> = if ct == ContentType::VizLogvar {
            variance.iter().map(|v| (1.0 + v.max(0.0)).log10()).collect()
        } else {
            variance.clone()
        };
        let data = f32_bytes(values);
        for fmt in [FileFormat::DAT, FileFormat::HDF5] {
            let rel = format!("archive/{}.{}", h.content_type.as_tag(), ext(fmt));
            out.write(&rel, fmt, &h, &data, json!({"mode": name, "variance_sha256": sha(&f32_bytes(variance.clone()))}));
            if fmt == FileFormat::HDF5 {
                let f = hdf5::File::open_rw(root.join(&rel)).unwrap();
                formats::write_source_content_type_attr(&f, ContentType::DuiCoherentPhase).unwrap();
                let last = out.entries.last_mut().unwrap();
                last["source_content_type"] = json!(ContentType::DuiCoherentPhase as u8);
            }
        }
    }

    // ── Legacy: content type 0, interleaved arctan/magnitude by the flags ──
    let mut legacy = DataHeader::default();
    legacy.line_size = 2 * positions as i32;
    legacy.sample_size = 2;
    legacy.flags = hf::DEMODULATED | hf::INTERLEAVED | hf::UNSIGNED;
    legacy.sample_rate = 250_000_000;
    legacy.trig_frequency = 50;
    legacy.range = 1000;
    let ts = b"2026-01-05 10:00:00";
    legacy.timestamp[..ts.len()].copy_from_slice(ts);
    out.write("legacy/ct0_arctan_magnitude.dat", FileFormat::DAT, &legacy, &pairs7, json!({"mode": "arctan_magnitude"}));

    // ── Legacy: a v0 file (2026-06..v1) carrying 8, the DUI pair, split off ──
    let mut v0 = s22.clone();
    v0.content_type = ContentType::InterleavedHwPhaseAmplitude;
    v0.content_type_raw = 8;
    v0.format_version = 0;
    v0.spatial_step_m = 0.0;
    v0.temporal_step_s = 0.0;
    hd::set_card_tag(&mut v0, ""); // a single-card run before v1: no tag
    out.write("legacy/v0_ct8_phase_amplitude.dat", FileFormat::DAT, &v0, &pairs22, json!({"mode": "phase_amplitude"}));
    // …and a v0 dual-card file carrying the legacy variant tag of the p18 card
    // (written from the registry fixture's `cards.legacy_tags`, never spelled here).
    let fixture: Value = serde_json::from_str(
        &std::fs::read_to_string(Path::new(env!("AUDACE_LIB")).join("tests/fixtures/content_types_v1.json")).unwrap(),
    )
    .unwrap();
    let legacy_p18 = fixture["cards"].as_array().unwrap().iter().find(|c| c["code"] == "p18").unwrap()["legacy_tags"][0]
        .as_str()
        .unwrap()
        .to_string();
    let mut v0dual = s22.clone();
    v0dual.content_type = ContentType::HwArctanDui;
    v0dual.content_type_raw = 4;
    v0dual.line_size = positions as i32;
    v0dual.flags = hf::DEMODULATED | hf::FLOAT | hf::PHASE;
    v0dual.format_version = 0;
    v0dual.spatial_step_m = 0.0;
    v0dual.temporal_step_s = 0.0;
    hd::set_card_tag(&mut v0dual, &legacy_p18);
    out.write("legacy/v0_ct4_dual_card.dat", FileFormat::DAT, &v0dual, &member(&pairs22, 4, 0), json!({"mode": "phase"}));

    let doc = json!({
        "audace_lib_commit": std::fs::read_to_string(Path::new(env!("AUDACE_LIB")).join("SNAPSHOT_COMMIT")).unwrap().trim(),
        "files": out.entries,
    });
    std::fs::write(root.join("expected.json"), serde_json::to_string_pretty(&doc).unwrap() + "\n").unwrap();
    println!("wrote {} files to {}", doc["files"].as_array().unwrap().len(), root.display());
}
