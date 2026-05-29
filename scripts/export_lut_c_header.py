"""
Export LUT-KAN model to C header for ESP32 or Arduino Mega 2560 (AVR).
Author: Oleksandr Kuznetsov

--target esp32  (default) — direct Flash XIP or DRAM_ATTR
--target avr             — PROGMEM chunks ≤27KB + pgm_read_*_far
"""
import argparse, json, textwrap
from pathlib import Path
import numpy as np
import torch

AVR_CHUNK = 27000   # bytes per chunk; < 32768 (AVR signed-16 limit)

def _f32(a): return np.asarray(a, dtype=np.float32)

def _load_lut_layer(p):
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    q   = d["q_table"]; sc = _f32(d["scale"]); ym = _f32(d["y_min"])
    kn  = _f32(d["knots"])
    L   = int(np.asarray(d["L"]).item())
    E,K,Lc = q.shape; assert L==Lc and K==len(kn)-1
    ebs = _f32(np.asarray(d.get("edge_base_scale",   np.array([], dtype=np.float32))))
    ess = _f32(np.asarray(d.get("edge_spline_scale", np.array([], dtype=np.float32))))
    eom = _f32(np.asarray(d.get("edge_out_scale",    np.array([], dtype=np.float32))))
    return dict(q_table=q, scale=sc, y_min=ym, knots=kn, L=L, E=E, K=K,
                scheme=str(np.asarray(d["scheme"]).item()),
                oob=str(np.asarray(d["oob_behavior"]).item()),
                boundary=str(np.asarray(d["boundary_mode"]).item()),
                edge_base_scale=ebs if ebs.size else None,
                edge_spline_scale=ess if ess.size else None,
                edge_out_scale=eom if eom.size else None)

def _c_float_array(name, vals, comment="", target="esp32", sram=False):
    v = _f32(vals).flatten()
    items = [f"{x:.8f}f" for x in v]
    if target=="avr":  q = "const PROGMEM"
    elif sram:         q = "DRAM_ATTR"
    else:              q = "const"
    lines = [f"static {q} float {name}[{len(items)}] = {{  // {comment}"]
    for i in range(0,len(items),8): lines.append("    "+", ".join(items[i:i+8])+",")
    lines.append("};"); return "\n".join(lines)

def _c_int8_array(name, vals, comment="", target="esp32", sram=False):
    v = vals.flatten().astype(np.int8)
    items = [str(int(x)) for x in v]
    if target=="avr":  q = "const PROGMEM"
    elif sram:         q = "DRAM_ATTR"
    else:              q = "const"
    lines = [f"static {q} int8_t {name}[{len(items)}] = {{  // {comment}"]
    for i in range(0,len(items),16): lines.append("    "+", ".join(items[i:i+16])+",")
    lines.append("};"); return "\n".join(lines)

# ── AVR chunked arrays ──────────────────────────────────────────────────────
def _avr_int8_chunked(base_name, vals, comment=""):
    """Split int8 array into ≤AVR_CHUNK-byte PROGMEM chunks; return (decls, init_lines, reader_fn)."""
    flat = vals.flatten().astype(np.int8)
    total = len(flat)
    chunk_n = AVR_CHUNK   # elements (int8 = 1 byte each)
    chunks = [flat[i:i+chunk_n] for i in range(0, total, chunk_n)]
    n = len(chunks)

    decls = []
    for ci, ch in enumerate(chunks):
        items = [str(int(x)) for x in ch]
        lines = [f"static const int8_t {base_name}_C{ci}[{len(items)}] PROGMEM = {{  // {comment} chunk {ci}"]
        for i in range(0,len(items),16): lines.append("    "+", ".join(items[i:i+16])+",")
        lines.append("};")
        decls.append("\n".join(lines))

    # Far address variables
    init = [f"static uint32_t _fa_{base_name}[{n}];"]
    init_fn = [f"    // {base_name}"]
    for ci in range(n):
        init_fn.append(f"    _fa_{base_name}[{ci}] = pgm_get_far_address({base_name}_C{ci});")

    # Reader function
    offsets = [i*chunk_n for i in range(n)]
    sizes   = [len(ch) for ch in chunks]
    reader = [f"static inline int8_t kan_read_{base_name}(uint32_t idx) {{"]
    for ci in range(n-1):
        reader.append(f"    if (idx < {offsets[ci]+sizes[ci]}u) "
                      f"return (int8_t)pgm_read_byte_far(_fa_{base_name}[{ci}] + idx - {offsets[ci]}u);")
    reader.append(f"    return (int8_t)pgm_read_byte_far(_fa_{base_name}[{n-1}] + idx - {offsets[n-1]}u);")
    reader.append("}")

    return "\n".join(decls), init, init_fn, "\n".join(reader)

def _avr_float_chunked(base_name, vals, comment=""):
    """Split float array into ≤AVR_CHUNK/4-element PROGMEM chunks."""
    flat = _f32(vals).flatten()
    total = len(flat)
    chunk_n = AVR_CHUNK // 4  # float elements per chunk
    chunks = [flat[i:i+chunk_n] for i in range(0, total, chunk_n)]
    n = len(chunks)

    decls = []
    for ci, ch in enumerate(chunks):
        items = [f"{x:.8f}f" for x in ch]
        lines = [f"static const float {base_name}_C{ci}[{len(items)}] PROGMEM = {{  // {comment} chunk {ci}"]
        for i in range(0,len(items),8): lines.append("    "+", ".join(items[i:i+8])+",")
        lines.append("};")
        decls.append("\n".join(lines))

    init = [f"static uint32_t _fa_{base_name}[{n}];"]
    init_fn = [f"    // {base_name}"]
    for ci in range(n):
        init_fn.append(f"    _fa_{base_name}[{ci}] = pgm_get_far_address({base_name}_C{ci});")

    offsets = [i*chunk_n for i in range(n)]
    sizes   = [len(ch) for ch in chunks]
    reader = [f"static inline float kan_read_{base_name}(uint32_t idx) {{"]
    for ci in range(n-1):
        reader.append(f"    if (idx < {offsets[ci]+sizes[ci]}u) "
                      f"return pgm_read_float_far(_fa_{base_name}[{ci}] + (idx - {offsets[ci]}u)*4u);")
    reader.append(f"    return pgm_read_float_far(_fa_{base_name}[{n-1}] + (idx - {offsets[n-1]}u)*4u);")
    reader.append("}")

    return "\n".join(decls), init, init_fn, "\n".join(reader)

def _extract_test_samples(run_dir, n_features):
    ds = torch.load(run_dir/"dataset.pt", map_location="cpu")
    X  = ds["test_input"].numpy(); y = ds["test_label"].numpy().flatten().astype(int)
    return X[np.where(y==1)[0][0]].astype(np.float32), X[np.where(y==0)[0][0]].astype(np.float32)

def _fmt_sample(name, vec):
    items = [f"{v:.4f}f" for v in vec]
    lines = [f"static const float {name}[N_FEATURES] = {{"]
    for i in range(0,len(items),8): lines.append("    "+", ".join(items[i:i+8])+",")
    lines.append("};"); return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",      required=True)
    ap.add_argument("--lut-dir",      required=True)
    ap.add_argument("--out",          required=True)
    ap.add_argument("--dataset-name", default="CICIDS2017")
    ap.add_argument("--f1",           default="0.000")
    ap.add_argument("--roc-auc",      default="0.000")
    ap.add_argument("--sram",         action="store_true")
    ap.add_argument("--target",       default="esp32", choices=["esp32","avr"])
    args = ap.parse_args()

    tgt = args.target
    run_dir = Path(args.run_dir); lut_dir = Path(args.lut_dir)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    mf = json.loads((lut_dir/"manifest.json").read_text(encoding="utf-8"))
    L = mf["L"]; scheme = mf["scheme"]; oob_policy = mf["oob_policy"]
    print(f"LUT config: L={L}, scheme={scheme}, oob={oob_policy}  target={tgt}")

    md = torch.load(run_dir/"trained_model.pt", map_location="cpu")
    rw = md["architecture"]["width"]
    width = [int(w[0]) for w in rw] if rw and isinstance(rw[0],(list,tuple)) else [int(w) for w in rw]
    print(f"Architecture: {width}"); n_layers = len(width)-1

    layers = []
    for li in mf["layers"]:
        p = lut_dir / Path(li["path"]).name
        print(f"  Loading layer {li['layer_idx']}: {p.name}")
        lyr = _load_lut_layer(p); lyr["in_dim"]=li["in_dim"]; lyr["out_dim"]=li["out_dim"]
        layers.append(lyr)

    nf = width[0]; K = layers[0]["K"]
    atk, nrm = _extract_test_samples(run_dir, nf)
    oob_zero = 1 if oob_policy in ("zero_spline","zero") else 0
    total_kb = sum(lyr["q_table"].size + lyr["scale"].size*4 for lyr in layers) / 1024
    print(f"Estimated Flash: {total_kb:.1f} KB")

    lines = []

    # ── File header & platform block ────────────────────────────────────────
    if tgt == "avr":
        board = "Arduino Mega 2560 (ATmega2560 16MHz, 8KB SRAM, 256KB Flash)"
        sram_str = "PROGMEM/Flash (~3.5KB SRAM runtime)"
        plat = """\
#pragma once
#include <Arduino.h>
#include <avr/pgmspace.h>
/* AVR: far-PROGMEM access (ELPM, works for all of 256KB Flash) */
#define IRAM_ATTR
"""
    else:
        board = "ESP32-C3 SuperMini (RISC-V 160MHz, 320KB SRAM)"
        sram_str = "DRAM (SRAM)" if args.sram else "Flash/XIP"
        plat = """\
#pragma once
#include <math.h>
#include <string.h>
#ifdef ARDUINO_ARCH_ESP32
#include "esp_attr.h"
#else
#define IRAM_ATTR
#define DRAM_ATTR
#endif
"""

    lines.append(f"""\
/*
 * LUT-KAN — generated by scripts/export_lut_c_header.py
 * Board : {board}
 * Arch  : {width}   LUT L={L} {scheme} {oob_policy}
 * Data  : {args.dataset_name}  F1={args.f1}  ROC-AUC={args.roc_auc}
 * Flash : ~{total_kb:.0f} KB   SRAM: {sram_str}
 */""")
    lines.append(plat)

    lines.append(f"#define KAN_N_FEATURES {width[0]}")
    lines.append(f"#define KAN_N_LAYERS   {n_layers}")
    lines.append(f"#define KAN_LUT_L      {L}")
    lines.append(f"#define KAN_N_SEGS     {K}")
    lines.append(f"#define KAN_OOB_ZERO   {oob_zero}")
    lines.append(f'#define KAN_F1      "{args.f1}"')
    lines.append(f'#define KAN_ROC_AUC "{args.roc_auc}"')
    lines.append(f'#define KAN_DATASET "{args.dataset_name}"')
    lines.append(f'#define KAN_FLASH_KB "{int(total_kb)}"')
    lines.append(f'#define KAN_SRAM_MODE "{sram_str}"')
    for i,w in enumerate(width): lines.append(f"#define KAN_DIM{i} {w}")
    lines.append("")

    # ── Arrays ──────────────────────────────────────────────────────────────
    if tgt == "avr":
        # AVR path: chunked PROGMEM + far-address readers
        all_init_calls = []
        reader_fns = []

        # Shared knots (small, always fits)
        kn = layers[0]["knots"]
        lines.append(_c_float_array("KAN_KNOTS", kn, f"[{K+1}] knots", target="avr"))
        # knots: small enough for near read, but use far for consistency
        d,iv,ic,rf = _avr_float_chunked("KAN_KNOTS", kn, "knots")
        # Actually knots is tiny (<200B), just use near:
        lines.append("")
        lines.append("// KAN_KNOTS accessor (near - tiny array)")
        lines.append("static inline float kan_read_KAN_KNOTS(uint32_t idx) {")
        lines.append("    return pgm_read_float_near(&KAN_KNOTS[idx]);")
        lines.append("}")
        lines.append("")

        for li, lyr in enumerate(layers):
            E,Ks,Ls = lyr["q_table"].shape
            lines.append(f"// Layer {li}: [{lyr['in_dim']}→{lyr['out_dim']}]")

            # Q-table
            qname = f"KAN_L{li}_QTABLE"
            d,iv,ic,rf = _avr_int8_chunked(qname, lyr["q_table"],
                                            f"[{E}×{Ks}×{Ls}] int8")
            lines.append(d); lines.append(""); lines.append("\n".join(iv)); lines.append(rf); lines.append("")
            all_init_calls += ic

            # Scale
            sname = f"KAN_L{li}_SCALE"
            d,iv,ic,rf = _avr_float_chunked(sname, lyr["scale"],
                                             f"[{E}×{Ks}] scale")
            lines.append(d); lines.append(""); lines.append("\n".join(iv)); lines.append(rf); lines.append("")
            all_init_calls += ic

            # Optional spline-component arrays (small)
            if lyr["edge_base_scale"] is not None:
                bname = f"KAN_L{li}_BASE_SCALE"
                lines.append(_c_float_array(bname, lyr["edge_base_scale"],
                                            f"[{E}] alpha", target="avr"))
                d2,iv2,ic2,rf2 = _avr_float_chunked(bname, lyr["edge_base_scale"])
                lines.append(""); lines.append("\n".join(iv2)); lines.append(rf2); lines.append("")
                all_init_calls+=ic2

            if lyr["edge_spline_scale"] is not None:
                bname = f"KAN_L{li}_SPLINE_SCALE"
                lines.append(_c_float_array(bname, lyr["edge_spline_scale"],
                                            f"[{E}] beta", target="avr"))
                d2,iv2,ic2,rf2 = _avr_float_chunked(bname, lyr["edge_spline_scale"])
                lines.append(""); lines.append("\n".join(iv2)); lines.append(rf2); lines.append("")
                all_init_calls+=ic2

            if lyr["edge_out_scale"] is not None:
                bname = f"KAN_L{li}_OUT_SCALE"
                lines.append(_c_float_array(bname, lyr["edge_out_scale"],
                                            f"[{E}] out_scale", target="avr"))
                d2,iv2,ic2,rf2 = _avr_float_chunked(bname, lyr["edge_out_scale"])
                lines.append(""); lines.append("\n".join(iv2)); lines.append(rf2); lines.append("")
                all_init_calls+=ic2

        # Init function (must call from setup())
        lines.append("// Call once from setup() — computes 32-bit far addresses")
        lines.append("static void kan_avr_init() {")
        lines.append("\n".join(all_init_calls))
        lines.append("}")
        lines.append("")

    else:
        # ESP32 path: simple direct arrays
        lines.append(_c_float_array("KAN_KNOTS", layers[0]["knots"],
                                    f"[{K+1}] knots", target=tgt))
        lines.append("")
        for li,lyr in enumerate(layers):
            E,Ks,Ls = lyr["q_table"].shape
            lines.append(f"// Layer {li}: [{lyr['in_dim']}→{lyr['out_dim']}]")
            lines.append(_c_int8_array(f"KAN_L{li}_QTABLE", lyr["q_table"],
                                       f"[{E}×{Ks}×{Ls}] int8",
                                       target=tgt, sram=args.sram))
            lines.append("")
            lines.append(_c_float_array(f"KAN_L{li}_SCALE", lyr["scale"],
                                        f"[{E}×{Ks}] scale",
                                        target=tgt, sram=args.sram))
            lines.append("")
            if lyr["edge_base_scale"] is not None:
                lines.append(_c_float_array(f"KAN_L{li}_BASE_SCALE",
                                            lyr["edge_base_scale"],
                                            f"[{E}] alpha", target=tgt, sram=args.sram))
                lines.append("")
            if lyr["edge_spline_scale"] is not None:
                lines.append(_c_float_array(f"KAN_L{li}_SPLINE_SCALE",
                                            lyr["edge_spline_scale"],
                                            f"[{E}] beta", target=tgt, sram=args.sram))
                lines.append("")
            if lyr["edge_out_scale"] is not None:
                lines.append(_c_float_array(f"KAN_L{li}_OUT_SCALE",
                                            lyr["edge_out_scale"],
                                            f"[{E}] out_scale", target=tgt, sram=args.sram))
                lines.append("")

    # ── Test samples ─────────────────────────────────────────────────────────
    lines.append("#define N_FEATURES KAN_N_FEATURES")
    lines.append(_fmt_sample("KAN_SAMPLE_ATTACK", atk))
    lines.append(""); lines.append(_fmt_sample("KAN_SAMPLE_NORMAL", nrm)); lines.append("")

    # ── SiLU ─────────────────────────────────────────────────────────────────
    lines.append("""\
static inline float kan_silu(float x) {
    return x / (1.0f + expf(-x));
}
""")

    # ── Layer forward functions ───────────────────────────────────────────────
    for li, lyr in enumerate(layers):
        in_d = lyr["in_dim"]; out_d = lyr["out_dim"]
        K_l = lyr["q_table"].shape[1]; L_l = lyr["q_table"].shape[2]
        has_spline = (lyr["edge_base_scale"] is not None)

        if tgt == "avr":
            knot_rd = "kan_read_KAN_KNOTS"
            qtab_rd = f"kan_read_KAN_L{li}_QTABLE"
            scal_rd = f"kan_read_KAN_L{li}_SCALE"
            bscl_rd = f"kan_read_KAN_L{li}_BASE_SCALE"
            sscl_rd = f"kan_read_KAN_L{li}_SPLINE_SCALE"
            oscl_rd = f"kan_read_KAN_L{li}_OUT_SCALE"
        else:
            knot_rd = None  # use direct access

        fwd = f"IRAM_ATTR static void kan_layer{li}_forward(const float* input, float* output) {{\n"
        fwd += f"    static int   _seg[{in_d}];\n"
        fwd += f"    static int   _qi[{in_d}];\n"
        fwd += f"    static float _lam[{in_d}];\n"
        fwd += f"    static int   _ok[{in_d}];\n"

        if tgt == "avr":
            fwd += f"    const float x_lo = {knot_rd}(0);\n"
            fwd += f"    const float x_hi = {knot_rd}({K_l});\n"
        else:
            fwd += f"    const float x_lo = KAN_KNOTS[0];\n"
            fwd += f"    const float x_hi = KAN_KNOTS[{K_l}];\n"

        fwd += f"""    const float rng  = (x_hi > x_lo + 1e-12f) ? (x_hi - x_lo) : 1.0f;
    for (int i = 0; i < {in_d}; i++) {{
        float x = input[i];
        if (KAN_OOB_ZERO && (x < x_lo || x > x_hi)) {{
            _ok[i]=0; _seg[i]=0; _qi[i]=0; _lam[i]=0.0f; continue;
        }}
        _ok[i] = 1;
        if (x < x_lo) x = x_lo;
        if (x > x_hi) x = x_hi;
        int seg = (int)((x - x_lo) / rng * {K_l});
        if (seg < 0) seg = 0;
        if (seg >= {K_l}) seg = {K_l} - 1;\n"""

        if tgt == "avr":
            fwd += f"        float sl = {knot_rd}((uint32_t)seg);\n"
            fwd += f"        float sh = {knot_rd}((uint32_t)seg + 1);\n"
        else:
            fwd += f"        float sl = KAN_KNOTS[seg];\n"
            fwd += f"        float sh = KAN_KNOTS[seg + 1];\n"

        fwd += f"""        float sw = (sh > sl + 1e-12f) ? (sh - sl) : 1.0f;
        float u  = (x - sl) / sw * {L_l - 1};
        int qi = (int)u;
        if (qi < 0) qi = 0;
        if (qi >= {L_l} - 1) qi = {L_l} - 2;
        _seg[i] = seg; _qi[i] = qi; _lam[i] = u - (float)qi;
    }}
    for (int j = 0; j < {out_d}; j++) {{
        float acc = 0.0f;
        const int base_e = j * {in_d};
        for (int i = 0; i < {in_d}; i++) {{
            if (!_ok[i]) continue;
            int e = base_e + i;
            int seg = _seg[i]; int qi = _qi[i]; float lam = _lam[i];\n"""

        if tgt == "avr":
            fwd += f"            float sv = {scal_rd}((uint32_t)e * {K_l} + seg);\n"
            fwd += f"            uint32_t tof = (uint32_t)e * {K_l} * {L_l} + (uint32_t)seg * {L_l};\n"
            fwd += f"            float v0 = sv * (float){qtab_rd}(tof + qi);\n"
            fwd += f"            float v1 = sv * (float){qtab_rd}(tof + qi + 1);\n"
            fwd += f"            float phi = (1.0f - lam) * v0 + lam * v1;\n"
            if has_spline:
                fwd += f"            phi = {oscl_rd}((uint32_t)e) * ({bscl_rd}((uint32_t)e) * kan_silu(input[i]) + {sscl_rd}((uint32_t)e) * phi);\n"
        else:
            fwd += f"            float sv = KAN_L{li}_SCALE[e * {K_l} + seg];\n"
            fwd += f"            int tof = e * {K_l} * {L_l} + seg * {L_l};\n"
            fwd += f"            float v0 = sv * (float)KAN_L{li}_QTABLE[tof + qi];\n"
            fwd += f"            float v1 = sv * (float)KAN_L{li}_QTABLE[tof + qi + 1];\n"
            fwd += f"            float phi = (1.0f - lam) * v0 + lam * v1;\n"
            if has_spline:
                fwd += f"            phi = KAN_L{li}_OUT_SCALE[e] * (KAN_L{li}_BASE_SCALE[e] * kan_silu(input[i]) + KAN_L{li}_SPLINE_SCALE[e] * phi);\n"

        fwd += "            acc += phi;\n        }\n        output[j] = acc;\n    }\n}\n"
        lines.append(fwd)

    # ── kan_infer ─────────────────────────────────────────────────────────────
    mh = max(width[1:-1]) if len(width)>2 else width[0]
    ib = max(mh, width[0])
    inf = [f"IRAM_ATTR static float kan_infer(const float* input) {{"]
    inf += [f"    static float buf_a[{ib}];", f"    static float buf_b[{ib}];",
            "    float* buf_in = (float*)input;", "    float* buf_out = buf_a;"]
    for li in range(n_layers):
        inf += [f"    memset(buf_out, 0, sizeof(float)*{width[li+1]});",
                f"    kan_layer{li}_forward(buf_in, buf_out);",
                "    buf_in = buf_out;",
                "    buf_out = (buf_out == buf_a) ? buf_b : buf_a;"]
    inf += ["    return 1.0f / (1.0f + expf(-buf_in[0]));", "}"]
    lines.append("\n".join(inf))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Header written: {out_path}")
    print(f"     Flash: ~{total_kb:.1f} KB  |  target={tgt}")
    if tgt == "avr":
        print(f"     SRAM runtime: ~3500 B  |  call kan_avr_init() from setup()!")

if __name__ == "__main__":
    main()
