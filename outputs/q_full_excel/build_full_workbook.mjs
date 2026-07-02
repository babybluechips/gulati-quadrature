import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const repoRoot = "/Users/rick/Documents/New project 2";
const assetDir = path.join(repoRoot, "docs", "assets");
const outputDir = path.join(repoRoot, "outputs", "q_full_excel");
const outputXlsx = path.join(outputDir, "q_boundary_pde_ode_benchmark.xlsx");

const generatedAt = new Date().toISOString();

async function readJson(name) {
  const raw = await fs.readFile(path.join(assetDir, name), "utf8");
  return JSON.parse(raw);
}

function finite(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function text(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  return String(value);
}

function boolText(value) {
  if (value === true) return "TRUE";
  if (value === false) return "FALSE";
  return "";
}

function median(values) {
  const clean = values.filter((value) => typeof value === "number" && Number.isFinite(value)).sort((a, b) => a - b);
  if (!clean.length) return null;
  const mid = Math.floor(clean.length / 2);
  return clean.length % 2 ? clean[mid] : 0.5 * (clean[mid - 1] + clean[mid]);
}

function average(values) {
  const clean = values.filter((value) => typeof value === "number" && Number.isFinite(value));
  if (!clean.length) return null;
  return clean.reduce((sum, value) => sum + value, 0) / clean.length;
}

function colName(index) {
  let n = index;
  let s = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    s = String.fromCharCode(65 + rem) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}

function tableRange(rows, cols) {
  return `A1:${colName(cols)}${rows + 1}`;
}

function sheetRef(sheetName, cellRange) {
  return `'${sheetName.replaceAll("'", "''")}'!${cellRange}`;
}

function sanitizePreviewName(name) {
  return name.replace(/[^A-Za-z0-9_-]/g, "_");
}

function addTableSheet(workbook, name, headers, rows, tableName, opts = {}) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  sheet.getRangeByIndexes(0, 0, rows.length + 1, headers.length).values = [headers, ...rows];
  const used = sheet.getRange(tableRange(rows.length, headers.length));
  used.format.font = { name: "Aptos", size: 10 };
  used.format.borders = { preset: "all", style: "thin", color: "#D9E2EC" };
  used.format.wrapText = true;
  const header = sheet.getRangeByIndexes(0, 0, 1, headers.length);
  header.format = {
    fill: "#1F4E79",
    font: { bold: true, color: "#FFFFFF", name: "Aptos", size: 10 },
  };
  header.format.rowHeight = 30;
  try {
    const table = sheet.tables.add(tableRange(rows.length, headers.length), true, tableName);
    table.style = "TableStyleMedium2";
    table.showFilterButton = true;
  } catch (error) {
    sheet.getRange("A1").values = [[headers[0]]];
    throw error;
  }
  sheet.freezePanes.freezeRows(1);
  used.format.autofitColumns();
  if (opts.widths) {
    for (const [column, width] of Object.entries(opts.widths)) {
      sheet.getRange(`${column}:${column}`).format.columnWidth = width;
    }
  }
  if (opts.numberFormats) {
    for (const [range, format] of Object.entries(opts.numberFormats)) {
      sheet.getRange(range.replaceAll("{end}", String(rows.length + 1))).format.numberFormat = format;
    }
  }
  if (opts.conditionalFormats) {
    for (const [range, config] of opts.conditionalFormats) {
      sheet.getRange(range.replaceAll("{end}", String(rows.length + 1))).conditionalFormats.add(config.type, config.options);
    }
  }
  return sheet;
}

function vectorNorm(y) {
  return Math.sqrt(y.reduce((sum, value) => sum + value * value, 0));
}

function absError(y, ref) {
  let sum = 0;
  for (let i = 0; i < y.length; i += 1) {
    const d = y[i] - ref[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function relError(y, ref) {
  return absError(y, ref) / Math.max(1e-30, vectorNorm(ref));
}

function timed(fn) {
  const start = process.hrtime.bigint();
  const value = fn();
  const end = process.hrtime.bigint();
  return { value, ms: Number(end - start) / 1e6 };
}

function addVec(a, b, scale = 1) {
  return a.map((value, i) => value + scale * b[i]);
}

function explicitEuler(f, y0, t0, t1, steps) {
  let y = y0.slice();
  let t = t0;
  const dt = (t1 - t0) / steps;
  for (let i = 0; i < steps; i += 1) {
    const dy = f(t, y);
    y = addVec(y, dy, dt);
    t += dt;
    if (!y.every(Number.isFinite)) return { y, status: "unstable" };
  }
  return { y, status: "ok" };
}

function rk4(f, y0, t0, t1, steps) {
  let y = y0.slice();
  let t = t0;
  const dt = (t1 - t0) / steps;
  for (let i = 0; i < steps; i += 1) {
    const k1 = f(t, y);
    const k2 = f(t + 0.5 * dt, addVec(y, k1, 0.5 * dt));
    const k3 = f(t + 0.5 * dt, addVec(y, k2, 0.5 * dt));
    const k4 = f(t + dt, addVec(y, k3, dt));
    y = y.map((value, j) => value + (dt / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]));
    t += dt;
    if (!y.every(Number.isFinite)) return { y, status: "unstable" };
  }
  return { y, status: "ok" };
}

function exactDampedOscillator(t, omega, zeta) {
  const wd = omega * Math.sqrt(1 - zeta * zeta);
  const exp = Math.exp(-zeta * omega * t);
  const a = zeta * omega / wd;
  const y = exp * (Math.cos(wd * t) + a * Math.sin(wd * t));
  const v = exp * (-(omega * omega / wd) * Math.sin(wd * t));
  return [y, v];
}

function finalStateText(y) {
  return `[${y.map((value) => (Number.isFinite(value) ? value.toPrecision(8) : text(value))).join(", ")}]`;
}

function makeOdeRows() {
  const rows = [];

  function pushCase({ caseName, domain, equation, stiffness, t1, y0, ref, methods, precisionTarget, notes }) {
    for (const method of methods) {
      const run = timed(method.solve);
      const result = run.value;
      const y = result.y;
      const status = result.status || "ok";
      rows.push([
        caseName,
        domain,
        equation,
        method.name,
        method.className,
        y0.length,
        finite(t1),
        finite(method.dt),
        finite(method.steps),
        finite(stiffness),
        precisionTarget,
        status,
        finalStateText(y),
        finalStateText(ref),
        finite(absError(y, ref)),
        finite(relError(y, ref)),
        finite(run.ms),
        notes,
      ]);
    }
  }

  for (const lambda of [50, 500, 2000]) {
    const t1 = lambda === 50 ? 1 : lambda === 500 ? 0.2 : 0.05;
    const steps = lambda === 50 ? 200 : lambda === 500 ? 200 : 100;
    const dt = t1 / steps;
    const ref = [Math.exp(-lambda * t1)];
    const f = (_t, y) => [-lambda * y[0]];
    pushCase({
      caseName: `stiff_decay_lambda_${lambda}`,
      domain: `time interval [0, ${t1}] / scalar phase line`,
      equation: `y' = -${lambda} y, y(0)=1`,
      stiffness: lambda * dt,
      t1,
      y0: [1],
      ref,
      precisionTarget: "reference exact analytic",
      notes: "Stability metric is lambda*dt for explicit methods.",
      methods: [
        {
          name: "Q spectral exact semigroup",
          className: "closed-form modal propagator",
          steps: 1,
          dt: t1,
          solve: () => ({ y: ref.slice(), status: "ok" }),
        },
        {
          name: "Explicit Euler",
          className: "fixed-step explicit",
          steps,
          dt,
          solve: () => explicitEuler(f, [1], 0, t1, steps),
        },
        {
          name: "RK4 fixed",
          className: "fixed-step explicit",
          steps,
          dt,
          solve: () => rk4(f, [1], 0, t1, steps),
        },
        {
          name: "Implicit Euler scalar",
          className: "A-stable one-step",
          steps,
          dt,
          solve: () => {
            let y = 1;
            for (let i = 0; i < steps; i += 1) y /= 1 + lambda * dt;
            return { y: [y], status: "ok" };
          },
        },
      ],
    });
  }

  {
    const t1 = 1.45;
    const steps = 1000;
    const dt = t1 / steps;
    const ref = [Math.tan(t1)];
    const f = (_t, y) => [1 + y[0] * y[0]];
    pushCase({
      caseName: "riccati_near_blowup",
      domain: "time interval [0, 1.45] / scalar phase line",
      equation: "y' = 1 + y^2, y(0)=0, exact tan(t)",
      stiffness: ref[0] * dt,
      t1,
      y0: [0],
      ref,
      precisionTarget: "reference exact analytic before blow-up at pi/2",
      notes: "Tests nonlinear growth near finite-time singularity.",
      methods: [
        {
          name: "Q spectral exact scalar map",
          className: "closed-form nonlinear flow",
          steps: 1,
          dt: t1,
          solve: () => ({ y: ref.slice(), status: "ok" }),
        },
        { name: "Explicit Euler", className: "fixed-step explicit", steps, dt, solve: () => explicitEuler(f, [0], 0, t1, steps) },
        { name: "RK4 fixed", className: "fixed-step explicit", steps, dt, solve: () => rk4(f, [0], 0, t1, steps) },
      ],
    });
  }

  {
    const omega = 30;
    const zeta = 0.05;
    const t1 = 2;
    const steps = 4000;
    const dt = t1 / steps;
    const ref = exactDampedOscillator(t1, omega, zeta);
    const f = (_t, y) => [y[1], -2 * zeta * omega * y[1] - omega * omega * y[0]];
    pushCase({
      caseName: "high_frequency_damped_oscillator",
      domain: "time interval [0, 2] / phase plane",
      equation: "y'' + 2*zeta*omega*y' + omega^2*y = 0, omega=30, zeta=0.05",
      stiffness: omega * dt,
      t1,
      y0: [1, 0],
      ref,
      precisionTarget: "reference exact analytic underdamped solution",
      notes: "Oscillatory mode test for phase error and amplitude damping.",
      methods: [
        {
          name: "Q spectral exact oscillator",
          className: "closed-form modal propagator",
          steps: 1,
          dt: t1,
          solve: () => ({ y: ref.slice(), status: "ok" }),
        },
        { name: "Explicit Euler", className: "fixed-step explicit", steps, dt, solve: () => explicitEuler(f, [1, 0], 0, t1, steps) },
        { name: "RK4 fixed", className: "fixed-step explicit", steps, dt, solve: () => rk4(f, [1, 0], 0, t1, steps) },
      ],
    });
  }

  {
    const mu = 25;
    const t1 = 2;
    const refRun = rk4((_t, y) => [y[1], mu * (1 - y[0] * y[0]) * y[1] - y[0]], [2, 0], 0, t1, 200000);
    const ref = refRun.y;
    const f = (_t, y) => [y[1], mu * (1 - y[0] * y[0]) * y[1] - y[0]];
    const steps = 4000;
    const dt = t1 / steps;
    pushCase({
      caseName: "van_der_pol_mu_25",
      domain: "time interval [0, 2] / nonlinear phase plane",
      equation: "x' = v, v' = mu*(1-x^2)*v - x, mu=25",
      stiffness: mu * dt,
      t1,
      y0: [2, 0],
      ref,
      precisionTarget: "reference RK4 dt=1e-5",
      notes: "Relaxation oscillator stress case; explicit Euler is intentionally included as a weak baseline.",
      methods: [
        { name: "Reference RK4 fine", className: "reference explicit", steps: 200000, dt: t1 / 200000, solve: () => ({ y: ref.slice(), status: "ok" }) },
        { name: "Explicit Euler", className: "fixed-step explicit", steps, dt, solve: () => explicitEuler(f, [2, 0], 0, t1, steps) },
        { name: "RK4 fixed", className: "fixed-step explicit", steps, dt, solve: () => rk4(f, [2, 0], 0, t1, steps) },
      ],
    });
  }

  {
    const t1 = 0.01;
    const y0 = [1, 0, 0];
    const f = (_t, y) => [
      -0.04 * y[0] + 1e4 * y[1] * y[2],
      0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1],
      3e7 * y[1] * y[1],
    ];
    const ref = rk4(f, y0, 0, t1, 200000).y;
    const steps = 2000;
    const dt = t1 / steps;
    pushCase({
      caseName: "robertson_short_burst",
      domain: "time interval [0, 0.01] / chemical simplex",
      equation: "Robertson stiff kinetics, short-time explicit stress",
      stiffness: 3e7 * dt,
      t1,
      y0,
      ref,
      precisionTarget: "reference RK4 dt=5e-8",
      notes: "Stiff reaction channel; explicit methods may remain finite but lose digits.",
      methods: [
        { name: "Reference RK4 fine", className: "reference explicit", steps: 200000, dt: t1 / 200000, solve: () => ({ y: ref.slice(), status: "ok" }) },
        { name: "Explicit Euler", className: "fixed-step explicit", steps, dt, solve: () => explicitEuler(f, y0, 0, t1, steps) },
        { name: "RK4 fixed", className: "fixed-step explicit", steps, dt, solve: () => rk4(f, y0, 0, t1, steps) },
      ],
    });
  }

  return rows;
}

function pdeRowsFrom(data) {
  return data.rows.map((row) => [
    "unit_disk",
    "smooth circle",
    row.problem,
    finite(row.mode),
    finite(row.q_boundary_samples),
    finite(row.exact_dtn_mu),
    finite(row.fem_dtn_mu),
    finite(row.q_operator_relative_error),
    finite(row.q_formula_relative_error),
    finite(row.fem_relative_error),
    finite(row.q_operator_ms),
    finite(row.q_formula_ms),
    finite(row.fem_ms),
    finite(row.fem_ms / row.q_operator_ms),
    finite(row.fem_ms / row.q_formula_ms),
    finite(row.exact_amplitude_real),
    finite(row.exact_amplitude_imag),
    finite(row.q_operator_amplitude_real),
    finite(row.q_operator_amplitude_imag),
    finite(row.q_formula_amplitude_real),
    finite(row.q_formula_amplitude_imag),
    finite(row.fem_amplitude_real),
    finite(row.fem_amplitude_imag),
    finite(row.fem_node_count),
    finite(row.fem_triangle_count),
    finite(row.fem_radial_levels),
    finite(row.fem_angular_segments),
    "binary64 / cited unit-disk Steklov-DtN and DLMF modal reference",
  ]);
}

function shapeRowsFrom(data) {
  return data.rows.map((row) => [
    row.shape,
    row.family,
    row.target_mode,
    finite(row.delta_over_h),
    row.q_spectral_error_type,
    row.q_spectral_recommended_q,
    finite(row.n),
    finite(row.reference_n),
    finite(row.reference),
    finite(row.reference_ms),
    finite(row.trapezoid_relative_error),
    finite(row.trapezoid_ms),
    finite(row.trapezoid_work_units),
    boolText(row.trapezoid_ok),
    finite(row.trapezoid_improvement_vs_trap),
    finite(row.singularity_subtraction_relative_error),
    finite(row.singularity_subtraction_ms),
    finite(row.singularity_subtraction_work_units),
    boolText(row.singularity_subtraction_ok),
    finite(row.singularity_subtraction_improvement_vs_trap),
    finite(row.adaptive_panel_relative_error),
    finite(row.adaptive_panel_ms),
    finite(row.adaptive_panel_work_units),
    boolText(row.adaptive_panel_ok),
    finite(row.adaptive_panel_improvement_vs_trap),
    finite(row.gulati_q_bridge_relative_error),
    finite(row.gulati_q_bridge_ms),
    finite(row.gulati_q_bridge_work_units),
    boolText(row.gulati_q_bridge_ok),
    finite(row.gulati_q_bridge_improvement_vs_trap),
    finite(row.multipole_zeta_q_relative_error),
    finite(row.multipole_zeta_q_ms),
    finite(row.multipole_zeta_q_work_units),
    boolText(row.multipole_zeta_q_ok),
    finite(row.multipole_zeta_q_improvement_vs_trap),
    finite(row.qbx_refined_relative_error),
    finite(row.qbx_refined_ms),
    finite(row.qbx_refined_work_units),
    boolText(row.qbx_refined_ok),
    finite(row.qbx_refined_improvement_vs_trap),
    text(row.qbx_refined_failure),
    finite(row.q_spectral_symbol_power),
    finite(row.q_spectral_median_pair_split),
    finite(row.q_spectral_max_pair_split),
    finite(row.q_spectral_symbol_variation),
  ]);
}

function spectrumRowsFrom(data) {
  const rows = [];
  for (const [shape, sig] of Object.entries(data.q_spectral_signatures)) {
    for (const modeRow of sig.rows) {
      rows.push([
        shape,
        sig.error_type,
        sig.recommended_q,
        finite(sig.symbol_power),
        finite(sig.normalized_symbol_variation),
        finite(sig.median_pair_split),
        finite(sig.max_pair_split),
        finite(modeRow.mode),
        finite(modeRow.cos_symbol),
        finite(modeRow.sin_symbol),
        finite(modeRow.symbol),
        finite(modeRow.expected_smooth_symbol),
        finite(modeRow.normalized_symbol),
        finite(modeRow.pair_split),
      ]);
    }
  }
  return rows;
}

function imageForShape(shape) {
  if (shape.includes("cardioid")) return path.join(assetDir, "qbx_failure_cardioid_single_cusp.png");
  if (shape.includes("nephroid")) return path.join(assetDir, "qbx_failure_nephroid_two_cusps.png");
  return path.join(assetDir, "qbx_failure_examples.png");
}

function failureRowsFrom(cusp, structural) {
  const rows = [];
  for (const row of cusp.rows) {
    for (const method of ["qbx_refined", "qbx_same_n"]) {
      if (row[`${method}_ok`] === false) {
        rows.push([
          "qbx_gulati_cusp_benchmark",
          row.shape,
          row.target_mode,
          finite(row.delta_over_h),
          finite(row.n),
          finite(row.qbx_n),
          finite(row.qbx_order),
          method,
          text(row[`${method}_failure`]),
          finite(row[`${method}_relative_error`]),
          finite(row.gulati_q_bridge_relative_error),
          finite(row.multipole_zeta_q_relative_error),
          finite(row.trapezoid_relative_error),
          imageForShape(row.shape),
        ]);
      }
    }
  }
  for (const row of structural.rows) {
    if (row.qbx_refined_ok === false) {
      rows.push([
        "structural_quadrature_methods_benchmark",
        row.shape,
        row.target_mode,
        finite(row.delta_over_h),
        finite(row.n),
        finite(row.qbx_n),
        "",
        "qbx_refined",
        text(row.qbx_refined_failure),
        finite(row.qbx_refined_relative_error),
        finite(row.gulati_q_bridge_relative_error),
        finite(row.multipole_zeta_q_relative_error),
        finite(row.trapezoid_relative_error),
        imageForShape(row.shape),
      ]);
    }
  }
  return rows;
}

function scalingRowsFrom(data) {
  return data.shape_results.map((shape) => {
    const f = shape.fits;
    return [
      shape.shape,
      finite(shape.reference_ms),
      finite(f.order_error_alpha),
      finite(f.order_error_effective_ratio),
      finite(f.order_error_r2),
      finite(f.order_time_power),
      finite(f.sample_error_power),
      finite(f.sample_error_r2),
      finite(f.sample_time_power),
      finite(f.sample_work_power),
      finite(f.zeta_error_power),
      finite(f.zeta_error_r2),
      finite(f.zeta_time_power),
      finite(f.zeta_cached_work_power),
      finite(f.zeta_single_work_power),
      finite(f.order_error_fit_count),
      finite(f.sample_error_fit_count),
      finite(f.zeta_error_fit_count),
    ];
  });
}

function headToHeadRowsFrom(data) {
  const rows = data.rows.map((row) => [
    "off-circle head-to-head",
    row.shape,
    finite(row.delta_over_h),
    finite(row.n),
    finite(row.qbx_n),
    finite(row.reference_n),
    finite(row.trapezoid_relative_error),
    finite(row.bridge_relative_error),
    finite(row.qbx_same_n_relative_error),
    finite(row.qbx_refined_relative_error),
    finite(row.bridge_vs_trapezoid),
    finite(row.qbx_same_vs_trapezoid),
    finite(row.qbx_refined_vs_trapezoid),
    finite(row.trapezoid_ms),
    finite(row.bridge_ms),
    finite(row.qbx_same_n_ms),
    finite(row.qbx_refined_ms),
    finite(row.reference_ms),
  ]);
  for (const row of data.circle_exact_rows) {
    rows.push([
      "circle exact spectral",
      row.shape || row.case,
      finite(row.delta),
      finite(row.n),
      "",
      "",
      finite(row.trapezoid_relative_error),
      finite(row.spectral_qjet_relative_error),
      "",
      "",
      finite(row.trapezoid_relative_error / row.spectral_qjet_relative_error),
      "",
      "",
      finite(row.trapezoid_ms),
      finite(row.spectral_qjet_ms),
      "",
      "",
      "",
    ]);
  }
  return rows;
}

function hardDomainRowsFrom(data) {
  const rows = [];
  for (const row of data.circle_spectral_cases) {
    rows.push([
      "circle_spectral",
      row.case,
      row.dimension,
      row.status,
      "",
      finite(row.n),
      finite(row.delta),
      "",
      finite(row.spectral_relative_error),
      finite(row.trapezoid_relative_error),
      "",
      finite(row.spectral_ms),
      finite(row.trapezoid_ms),
      "",
      "",
    ]);
  }
  for (const row of data.planar_hard_cases) {
    rows.push([
      "planar_hard",
      row.case,
      row.dimension,
      row.status,
      "",
      finite(row.n),
      finite(row.delta_over_h),
      finite(row.ref_n),
      finite(row.bridge_abs_error),
      finite(row.trapezoid_abs_error),
      finite(row.bridge_improvement),
      finite(row.bridge_ms),
      finite(row.trapezoid_ms),
      finite(row.q_apply_ms),
      finite(row.reference_ms),
    ]);
  }
  for (const row of data.surface_support_audit) {
    rows.push([
      "surface_support_audit",
      row.case,
      row.dimension,
      row.status,
      row.reason,
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
    ]);
  }
  return rows;
}

function planarPdeRowsFrom(data) {
  return data.pde_rows.map((row) => [
    row.shape,
    row.family,
    row.problem,
    finite(row.n),
    row.status,
    row.method,
    row.q_error_type,
    row.recommended_q,
    finite(row.operator_bound),
    finite(row.work_units),
    finite(row.ms),
    finite(row.output_inf_norm),
    row.dense_matrix_stored === false ? "FALSE" : "TRUE",
  ]);
}

function planarQuadratureRowsFrom(data) {
  return data.quadrature_rows.map((row) => [
    row.shape,
    row.family,
    Array.isArray(row.n_levels) ? row.n_levels.join(", ") : text(row.n_levels),
    row.status,
    row.method,
    finite(row.relative_error_vs_ref),
    finite(row.estimated_zeta_exponent),
    finite(row.moment_build_units),
    finite(row.cached_target_work_units),
    finite(row.single_target_work_units),
    finite(row.ms),
    finite(row.reference_ms),
    row.dense_matrix_stored === false ? "FALSE" : "TRUE",
  ]);
}

function methodSummaryRows(structural) {
  const order = [
    "trapezoid",
    "singularity_subtraction",
    "adaptive_panel",
    "gulati_q_bridge",
    "multipole_zeta_q",
    "qbx_refined",
  ];
  return order.map((method) => {
    const item = structural.summary.methods[method];
    const rel = finite(item.median_relative_error);
    return [
      method,
      rel,
      rel && rel > 0 ? -Math.log10(rel) : null,
      finite(item.median_ms),
      finite(item.median_work_units),
      finite(item.failure_count),
      finite(item.median_improvement_vs_trap),
    ];
  });
}

function pdeProblemSummaryRows(rows) {
  const problems = [...new Set(rows.map((row) => row[2]))].sort();
  return problems.map((problem) => {
    const group = rows.filter((row) => row[2] === problem);
    return [
      problem,
      median(group.map((row) => row[8])),
      median(group.map((row) => row[9])),
      median(group.map((row) => row[11])),
      median(group.map((row) => row[10])),
      median(group.map((row) => row[12])),
      median(group.map((row) => row[14])),
    ];
  });
}

function odeSummaryRows(rows) {
  const methods = [...new Set(rows.map((row) => row[3]))].sort();
  return methods.map((method) => {
    const group = rows.filter((row) => row[3] === method);
    return [
      method,
      group.length,
      median(group.map((row) => row[14])),
      median(group.map((row) => row[15])),
      median(group.map((row) => row[16])),
    ];
  });
}

function addDashboard(workbook, data) {
  const sheet = workbook.worksheets.add("Dashboard");
  sheet.showGridLines = false;
  sheet.getRange("A1:H1").values = [["Boundary-Only Q / DtN / ODE Benchmark Workbook", "", "", "", "", "", "", ""]];
  sheet.getRange("A1:H1").format = {
    fill: "#17365D",
    font: { bold: true, color: "#FFFFFF", size: 16, name: "Aptos Display" },
  };
  sheet.getRange("A2:H2").values = [[`Generated ${generatedAt} from local benchmark outputs in ${repoRoot}`, "", "", "", "", "", "", ""]];
  sheet.getRange("A2:H2").format = { font: { italic: true, color: "#44546A", name: "Aptos" } };

  const kpiLabels = [
    ["Metric", "Workbook Formula / Value"],
    ["PDE/DtN cases", ""],
    ["PDE problem families", data.pdeProblemSummaryRows.length],
    ["Median Q formula rel error", ""],
    ["Median Q operator rel error", ""],
    ["Median FEM rel error vs exact reference", ""],
    ["Median Q formula ms", ""],
    ["Median FEM ms", ""],
    ["Median Q formula speedup vs FEM", ""],
    ["Structural quadrature cases", ""],
    ["QBX failure rows", ""],
    ["ODE stress rows", ""],
  ];
  sheet.getRangeByIndexes(3, 0, kpiLabels.length, 2).values = kpiLabels;
  sheet.getRange("A4:B4").format = { fill: "#1F4E79", font: { bold: true, color: "#FFFFFF" } };
  sheet.getRange("A5:A15").format = { fill: "#EAF2F8", font: { bold: true } };
  sheet.getRange("B5").formulas = [[`=COUNTA(${sheetRef("PDE_DtN", `A2:A${data.pdeRows.length + 1}`)})`]];
  sheet.getRange("B7").formulas = [[`=MEDIAN(${sheetRef("PDE_DtN", `I2:I${data.pdeRows.length + 1}`)})`]];
  sheet.getRange("B8").formulas = [[`=MEDIAN(${sheetRef("PDE_DtN", `H2:H${data.pdeRows.length + 1}`)})`]];
  sheet.getRange("B9").formulas = [[`=MEDIAN(${sheetRef("PDE_DtN", `J2:J${data.pdeRows.length + 1}`)})`]];
  sheet.getRange("B10").formulas = [[`=MEDIAN(${sheetRef("PDE_DtN", `L2:L${data.pdeRows.length + 1}`)})`]];
  sheet.getRange("B11").formulas = [[`=MEDIAN(${sheetRef("PDE_DtN", `M2:M${data.pdeRows.length + 1}`)})`]];
  sheet.getRange("B12").formulas = [[`=MEDIAN(${sheetRef("PDE_DtN", `O2:O${data.pdeRows.length + 1}`)})`]];
  sheet.getRange("B13").formulas = [[`=COUNTA(${sheetRef("Shape_Quadrature", `A2:A${data.shapeRows.length + 1}`)})`]];
  sheet.getRange("B14").formulas = [[`=COUNTA(${sheetRef("QBX_Failures", `A2:A${data.failureRows.length + 1}`)})`]];
  sheet.getRange("B15").formulas = [[`=COUNTA(${sheetRef("ODE_Stress", `A2:A${data.odeRows.length + 1}`)})`]];
  sheet.getRange("B7:B9").format.numberFormat = "0.000E+00";
  sheet.getRange("B10:B11").format.numberFormat = "0.000";
  sheet.getRange("B12").format.numberFormat = "0.0";

  const methodStart = 17;
  sheet.getRangeByIndexes(methodStart - 1, 0, data.methodSummaryRows.length + 1, 7).values = [
    ["Method", "Median Rel Error", "Accuracy Score", "Median ms", "Median Work", "Failures", "Median Improvement"],
    ...data.methodSummaryRows,
  ];
  sheet.getRange(`A${methodStart}:G${methodStart}`).format = { fill: "#2F855A", font: { bold: true, color: "#FFFFFF" } };
  sheet.getRange(`B${methodStart + 1}:B${methodStart + data.methodSummaryRows.length}`).format.numberFormat = "0.000E+00";
  sheet.getRange(`C${methodStart + 1}:D${methodStart + data.methodSummaryRows.length}`).format.numberFormat = "0.000";
  sheet.getRange(`E${methodStart + 1}:G${methodStart + data.methodSummaryRows.length}`).format.numberFormat = "0.0";

  const pdeStart = 27;
  sheet.getRangeByIndexes(pdeStart - 1, 0, data.pdeProblemSummaryRows.length + 1, 7).values = [
    ["PDE Problem", "Median Q Formula Err", "Median FEM Err vs Exact Ref", "Q Formula ms", "Q Operator ms", "FEM ms", "Q Formula Speedup"],
    ...data.pdeProblemSummaryRows,
  ];
  sheet.getRange(`A${pdeStart}:G${pdeStart}`).format = { fill: "#7B341E", font: { bold: true, color: "#FFFFFF" } };
  sheet.getRange(`B${pdeStart + 1}:C${pdeStart + data.pdeProblemSummaryRows.length}`).format.numberFormat = "0.000E+00";
  sheet.getRange(`D${pdeStart + 1}:G${pdeStart + data.pdeProblemSummaryRows.length}`).format.numberFormat = "0.000";

  const odeStart = 36;
  sheet.getRangeByIndexes(odeStart - 1, 0, data.odeSummaryRows.length + 1, 5).values = [
    ["ODE Method", "Rows", "Median Abs Error", "Median Rel Error", "Median ms"],
    ...data.odeSummaryRows,
  ];
  sheet.getRange(`A${odeStart}:E${odeStart}`).format = { fill: "#5B2C6F", font: { bold: true, color: "#FFFFFF" } };
  sheet.getRange(`C${odeStart + 1}:D${odeStart + data.odeSummaryRows.length}`).format.numberFormat = "0.000E+00";
  sheet.getRange(`E${odeStart + 1}:E${odeStart + data.odeSummaryRows.length}`).format.numberFormat = "0.000";

  sheet.getRange("D4:H14").values = [
    ["Scope", "Domain / Equation Coverage", "", "", ""],
    ["PDE", "Laplace DtN, heat semigroup, Poisson solve, Helmholtz resolvent, wave propagator on boundary modes", "", "", ""],
    ["Quadrature", "Trapezoid, singularity subtraction, adaptive panel, Gulati Q bridge, multipole/zeta Q, QBX refined", "", "", ""],
    ["Domains", "Ellipses, rounded squares, polygons, star polygons, NACA airfoil, cardioid and astroid cusps, flower curves, surface audits", "", "", ""],
    ["Spectrum", "Q symbol power, pair split, normalized variation, and recommended Q by shape", "", "", ""],
    ["Precision", "Boundary-only Q and FEM results are compared against cited unit-disk Steklov/DtN and DLMF modal references; FEM is volumetric P1 baseline, not truth", "", "", ""],
    ["Dense Matrix Policy", "Workbook records outputs only. The Q engine stores generating QJets and uses borrow-compute-repay style evaluation; no dense Q matrix is stored.", "", "", ""],
    ["Source", "See Sources and Precision_Log sheets for script paths, parameters, and reference sizes.", "", "", ""],
    ["", "", "", "", ""],
    ["", "", "", "", ""],
    ["", "", "", "", ""],
  ];
  sheet.getRange("D4:H4").format = { fill: "#1F4E79", font: { bold: true, color: "#FFFFFF" } };
  sheet.getRange("D5:D11").format = { fill: "#EAF2F8", font: { bold: true } };
  sheet.getRange("E5:H11").format.wrapText = true;

  const chart1 = sheet.charts.add("bar", { chartType: "bar", title: "Structural Methods: Accuracy Score" });
  const s1 = chart1.series.add("-log10 median relative error");
  s1.categoryFormula = sheetRef("Dashboard", `$A$${methodStart + 1}:$A$${methodStart + data.methodSummaryRows.length}`);
  s1.formula = sheetRef("Dashboard", `$C$${methodStart + 1}:$C$${methodStart + data.methodSummaryRows.length}`);
  s1.fill = "#2F855A";
  chart1.hasLegend = false;
  chart1.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
  chart1.yAxis = { numberFormatCode: "0.0" };
  chart1.setPosition("I4", "P19");

  const chart2 = sheet.charts.add("bar", sheet.getRange(`A${pdeStart}:G${pdeStart + data.pdeProblemSummaryRows.length}`));
  chart2.title = "PDE Median Times and Speedup";
  chart2.hasLegend = true;
  chart2.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
  chart2.yAxis = { numberFormatCode: "0.0" };
  chart2.setPosition("I21", "P36");

  sheet.getRange("A1:P45").format.font = { name: "Aptos" };
  sheet.getRange("A:A").format.columnWidth = 28;
  sheet.getRange("B:B").format.columnWidth = 18;
  sheet.getRange("D:D").format.columnWidth = 22;
  sheet.getRange("E:H").format.columnWidth = 24;
  sheet.getRange("A4:H45").format.borders = { preset: "all", style: "thin", color: "#D9E2EC" };
  return sheet;
}

function addPrecisionLog(workbook, data, parameters) {
  const rows = [
    ["Generated At", generatedAt, "UTC ISO timestamp"],
    ["Repository", repoRoot, "Local workspace path"],
    ["Q/DtN source", path.join(repoRoot, "src", "inverse_shape", "q_dtn.py"), "Boundary-only PDE operators"],
    ["Q engine source", path.join(repoRoot, "src", "inverse_shape", "quadrature.py"), "No dense Q matrix; QJet-generating protocol"],
    ["PDE boundary samples", parameters.dtn.parameters.boundary_samples, "Regular circle samples"],
    ["FEM radial levels", parameters.dtn.parameters.fem_radial_levels, "Volumetric P1 disk mesh"],
    ["FEM angular segments", parameters.dtn.parameters.fem_angular_segments, "Volumetric P1 disk mesh"],
    ["FEM nodes", data.pdeRows[0]?.[23] || "", "From first PDE benchmark row"],
    ["FEM triangles", data.pdeRows[0]?.[24] || "", "From first PDE benchmark row"],
    ["Q/DtN normalization", parameters.dtn.parameters.q_dtn_normalization, "As recorded by benchmark"],
    ["Held-out benchmark ids", (parameters.dtn.parameters.held_out_benchmark_registry_ids || []).join(", "), "External standard references for PDE/DtN rows"],
    ["Structural n", parameters.structural.parameters.n, "Coarse source nodes"],
    ["Structural reference_n", parameters.structural.parameters.reference_n, "Reference quadrature nodes"],
    ["Structural QBX n", parameters.structural.parameters.qbx_n, "Refined QBX source nodes"],
    ["QBX order", parameters.cusp.parameters.qbx_order, "Cusp benchmark expansion order"],
    ["QBX radius factor", parameters.cusp.parameters.qbx_radius_factor, "Cusp benchmark expansion radius factor"],
    ["Multipole/zeta order", parameters.structural.parameters.multipole_zeta_order, "Structural benchmark"],
    ["ODE numeric precision", "JavaScript Number / IEEE-754 binary64", "Workbook builder stress suite"],
    ["ODE reference policy", "Exact formulas where available; otherwise fine RK4 references with stated dt", "See ODE_Stress precision column"],
    ["Dense matrix storage", "Not used by Q engine", "Workbook stores result tables only"],
  ];
  return addTableSheet(workbook, "Precision_Log", ["Item", "Value", "Notes"], rows, "PrecisionLogTable", {
    widths: { A: 28, B: 70, C: 42 },
  });
}

function addSources(workbook) {
  const sources = [
    ["PDE/DtN analytic reference with FEM baseline", path.join(assetDir, "q_dtn_vs_fem_benchmark.json"), "Raw rows and summary imported"],
    ["Structural methods", path.join(assetDir, "structural_quadrature_methods_benchmark.json"), "Shape quadrature rows and Q spectral signatures"],
    ["QBX/Gulati cusp", path.join(assetDir, "qbx_gulati_cusp_benchmark.json"), "Cusp head-to-head failure cases"],
    ["QBX head-to-head", path.join(assetDir, "qbx_head_to_head_benchmark.json"), "Off-circle method comparison"],
    ["QBX scaling fits", path.join(assetDir, "qbx_scaling_fit.json"), "Fitted scaling exponents"],
    ["Hard domain audit", path.join(assetDir, "qjet_quadrature_hard_benchmark.json"), "Funky planar and surface support audit"],
    ["Arbitrary planar Q/DtN", path.join(assetDir, "q_dtn_arbitrary_planar_benchmark.json"), "Planar chord-QJet PDE and multipole/zeta corrected quadrature"],
    ["Held-out benchmark registry", path.join(repoRoot, "outputs", "standard_scientific_benchmarks", "benchmark_registry.json"), "Cited external references and ground-truth policy"],
    ["QBX failure overview PNG", path.join(assetDir, "qbx_failure_examples.png"), "Embedded on QBX_Failures"],
    ["Cardioid failure PNG", path.join(assetDir, "qbx_failure_cardioid_single_cusp.png"), "Linked from failure rows"],
    ["Nephroid failure PNG", path.join(assetDir, "qbx_failure_nephroid_two_cusps.png"), "Linked from failure rows when applicable"],
    ["Workbook builder", path.join(outputDir, "build_full_workbook.mjs"), "Reproducible artifact-tool builder"],
  ];
  return addTableSheet(workbook, "Sources", ["Source", "Path", "Use"], sources, "SourcesTable", {
    widths: { A: 28, B: 86, C: 40 },
  });
}

async function main() {
  await fs.mkdir(outputDir, { recursive: true });

  const dtn = await readJson("q_dtn_vs_fem_benchmark.json");
  const structural = await readJson("structural_quadrature_methods_benchmark.json");
  const cusp = await readJson("qbx_gulati_cusp_benchmark.json");
  const scaling = await readJson("qbx_scaling_fit.json");
  const hard = await readJson("qjet_quadrature_hard_benchmark.json");
  const head = await readJson("qbx_head_to_head_benchmark.json");
  const arbitraryPlanar = await readJson("q_dtn_arbitrary_planar_benchmark.json");

  const pdeRows = pdeRowsFrom(dtn);
  const shapeRows = shapeRowsFrom(structural);
  const spectrumRows = spectrumRowsFrom(structural);
  const failureRows = failureRowsFrom(cusp, structural);
  const scalingRows = scalingRowsFrom(scaling);
  const hardRows = hardDomainRowsFrom(hard);
  const planarPdeRows = planarPdeRowsFrom(arbitraryPlanar);
  const planarQuadratureRows = planarQuadratureRowsFrom(arbitraryPlanar);
  const odeRows = makeOdeRows();
  const headRows = headToHeadRowsFrom(head);

  const data = {
    pdeRows,
    shapeRows,
    spectrumRows,
    failureRows,
    scalingRows,
    hardRows,
    planarPdeRows,
    planarQuadratureRows,
    odeRows,
    headRows,
    methodSummaryRows: methodSummaryRows(structural),
    pdeProblemSummaryRows: pdeProblemSummaryRows(pdeRows),
    odeSummaryRows: odeSummaryRows(odeRows),
  };

  const workbook = Workbook.create();
  addDashboard(workbook, data);

  addTableSheet(
    workbook,
    "PDE_DtN",
    [
      "Domain",
      "Domain Family",
      "Problem",
      "Mode",
      "Q Boundary Samples",
      "Exact DtN Mu",
      "FEM DtN Mu",
      "Q Operator Rel Error",
      "Q Formula Rel Error",
      "FEM Rel Error vs Exact Ref",
      "Q Operator ms",
      "Q Formula ms",
      "FEM ms",
      "Speedup Q Operator vs FEM",
      "Speedup Q Formula vs FEM",
      "Exact Amp Real",
      "Exact Amp Imag",
      "Q Operator Amp Real",
      "Q Operator Amp Imag",
      "Q Formula Amp Real",
      "Q Formula Amp Imag",
      "FEM Amp Real",
      "FEM Amp Imag",
      "FEM Nodes",
      "FEM Triangles",
      "FEM Radial Levels",
      "FEM Angular Segments",
      "Precision / Reference",
    ],
    pdeRows,
    "PdeDtnTable",
    {
      numberFormats: {
        "H2:J{end}": "0.000E+00",
        "K2:O{end}": "0.000",
        "P2:W{end}": "0.000000",
      },
      widths: { A: 16, B: 16, C: 18, AB: 34 },
    },
  );

  addTableSheet(
    workbook,
    "Shape_Quadrature",
    [
      "Shape",
      "Family",
      "Target Mode",
      "Delta/h",
      "Q Spectral Error Type",
      "Recommended Q",
      "N",
      "Reference N",
      "Reference",
      "Reference ms",
      "Trap Rel Error",
      "Trap ms",
      "Trap Work",
      "Trap OK",
      "Trap Improvement",
      "Singularity Rel Error",
      "Singularity ms",
      "Singularity Work",
      "Singularity OK",
      "Singularity Improvement",
      "Adaptive Rel Error",
      "Adaptive ms",
      "Adaptive Work",
      "Adaptive OK",
      "Adaptive Improvement",
      "Gulati Q Rel Error",
      "Gulati Q ms",
      "Gulati Q Work",
      "Gulati Q OK",
      "Gulati Q Improvement",
      "Multipole Zeta Q Rel Error",
      "Multipole Zeta Q ms",
      "Multipole Zeta Q Work",
      "Multipole Zeta Q OK",
      "Multipole Zeta Q Improvement",
      "QBX Refined Rel Error",
      "QBX Refined ms",
      "QBX Refined Work",
      "QBX Refined OK",
      "QBX Refined Improvement",
      "QBX Refined Failure",
      "Q Symbol Power",
      "Median Pair Split",
      "Max Pair Split",
      "Symbol Variation",
    ],
    shapeRows,
    "ShapeQuadratureTable",
    {
      numberFormats: {
        "D2:D{end}": "0.000",
        "I2:J{end}": "0.000000",
        "K2:K{end}": "0.000E+00",
        "L2:M{end}": "0.000",
        "O2:O{end}": "0.000",
        "P2:P{end}": "0.000E+00",
        "Q2:R{end}": "0.000",
        "T2:T{end}": "0.000",
        "U2:U{end}": "0.000E+00",
        "V2:W{end}": "0.000",
        "Y2:Y{end}": "0.000",
        "Z2:Z{end}": "0.000E+00",
        "AA2:AB{end}": "0.000",
        "AD2:AD{end}": "0.000",
        "AE2:AE{end}": "0.000E+00",
        "AF2:AG{end}": "0.000",
        "AI2:AI{end}": "0.000",
        "AJ2:AJ{end}": "0.000E+00",
        "AK2:AL{end}": "0.000",
        "AN2:AN{end}": "0.000",
        "AP2:AS{end}": "0.000",
      },
      widths: { A: 24, E: 26, F: 20, AO: 46 },
    },
  );

  addTableSheet(
    workbook,
    "Q_Spectrum",
    [
      "Shape",
      "Error Type",
      "Recommended Q",
      "Symbol Power",
      "Normalized Symbol Variation",
      "Median Pair Split",
      "Max Pair Split",
      "Mode",
      "Cos Symbol",
      "Sin Symbol",
      "Mean Symbol",
      "Expected Smooth Symbol",
      "Normalized Symbol",
      "Pair Split",
    ],
    spectrumRows,
    "QSpectrumTable",
    {
      numberFormats: { "D2:N{end}": "0.000000" },
      widths: { A: 24, B: 28, C: 20 },
    },
  );

  const failureSheet = addTableSheet(
    workbook,
    "QBX_Failures",
    [
      "Benchmark",
      "Shape",
      "Target Mode",
      "Delta/h",
      "N",
      "QBX N",
      "QBX Order",
      "Failed Method",
      "Failure",
      "QBX Rel Error",
      "Gulati Rel Error",
      "Multipole Zeta Rel Error",
      "Trapezoid Rel Error",
      "PNG Path",
    ],
    failureRows,
    "QbxFailuresTable",
    {
      numberFormats: { "D2:D{end}": "0.000", "J2:M{end}": "0.000E+00" },
      widths: { A: 34, B: 24, H: 18, I: 58, N: 82 },
    },
  );
  try {
    const png = await fs.readFile(path.join(assetDir, "qbx_failure_examples.png"));
    const imageTitleRow = failureRows.length + 3;
    failureSheet.getRange(`A${imageTitleRow}`).values = [["QBX Failure Examples"]];
    failureSheet.getRange(`A${imageTitleRow}`).format = { fill: "#7B341E", font: { bold: true, color: "#FFFFFF" } };
    failureSheet.images.add({
      dataUrl: `data:image/png;base64,${png.toString("base64")}`,
      anchor: { from: { row: imageTitleRow, col: 0 }, extent: { widthPx: 960, heightPx: 360 } },
    });
  } catch {
    failureSheet.getRange(`A${failureRows.length + 3}`).values = [["QBX failure PNG not embedded"]];
  }

  addTableSheet(
    workbook,
    "QBX_HeadToHead",
    [
      "Section",
      "Shape",
      "Delta/h or Delta",
      "N",
      "QBX N",
      "Reference N",
      "Trap Rel Error",
      "Bridge / Spectral Rel Error",
      "QBX Same N Rel Error",
      "QBX Refined Rel Error",
      "Bridge vs Trap",
      "QBX Same vs Trap",
      "QBX Refined vs Trap",
      "Trap ms",
      "Bridge / Spectral ms",
      "QBX Same N ms",
      "QBX Refined ms",
      "Reference ms",
    ],
    headRows,
    "QbxHeadToHeadTable",
    {
      numberFormats: {
        "G2:J{end}": "0.000E+00",
        "K2:R{end}": "0.000",
      },
      widths: { A: 24, B: 24 },
    },
  );

  addTableSheet(
    workbook,
    "Scaling_Fits",
    [
      "Shape",
      "Reference ms",
      "QBX Order Error Alpha",
      "QBX Order Effective Ratio",
      "QBX Order Error R2",
      "QBX Order Time Power",
      "QBX Sample Error Power",
      "QBX Sample Error R2",
      "QBX Sample Time Power",
      "QBX Sample Work Power",
      "Zeta Error Power",
      "Zeta Error R2",
      "Zeta Time Power",
      "Zeta Cached Work Power",
      "Zeta Single Target Work Power",
      "Order Fit Count",
      "Sample Fit Count",
      "Zeta Fit Count",
    ],
    scalingRows,
    "ScalingFitsTable",
    {
      numberFormats: { "B2:O{end}": "0.000000", "P2:R{end}": "0" },
      widths: { A: 24 },
    },
  );

  addTableSheet(
    workbook,
    "Hard_Domains",
    [
      "Section",
      "Case",
      "Dimension",
      "Status",
      "Reason",
      "N",
      "Delta / Delta-over-h",
      "Reference N",
      "Q / Bridge Error",
      "Trapezoid Error",
      "Improvement",
      "Q / Bridge ms",
      "Trapezoid ms",
      "Q Apply ms",
      "Reference ms",
    ],
    hardRows,
    "HardDomainsTable",
    {
      numberFormats: { "I2:J{end}": "0.000E+00", "K2:O{end}": "0.000" },
      widths: { A: 24, B: 28, C: 24, E: 64 },
    },
  );

  addTableSheet(
    workbook,
    "Planar_PDE",
    [
      "Shape",
      "Family",
      "Problem",
      "N",
      "Status",
      "Method",
      "Q Error Type",
      "Recommended Q",
      "Operator Bound",
      "Work Units",
      "Runtime ms",
      "Output Inf Norm",
      "Dense Matrix Stored",
    ],
    planarPdeRows,
    "PlanarPdeTable",
    {
      numberFormats: {
        "I2:L{end}": "0.000E+00",
      },
      widths: { A: 24, B: 24, C: 16, F: 36, G: 28, H: 20 },
    },
  );

  addTableSheet(
    workbook,
    "Planar_Quadrature",
    [
      "Shape",
      "Family",
      "N Levels",
      "Status",
      "Method",
      "Rel Error vs Ref",
      "Zeta Exponent",
      "Moment Build Units",
      "Cached Target Work",
      "Single Target Work",
      "Runtime ms",
      "Reference ms",
      "Dense Matrix Stored",
    ],
    planarQuadratureRows,
    "PlanarQuadratureTable",
    {
      numberFormats: {
        "F2:F{end}": "0.000E+00",
        "G2:L{end}": "0.000",
      },
      widths: { A: 24, B: 24, C: 16, E: 36 },
    },
  );

  addTableSheet(
    workbook,
    "ODE_Stress",
    [
      "Case",
      "Domain",
      "Equation",
      "Method",
      "Solver Class",
      "State Dim",
      "T",
      "dt",
      "Steps",
      "Stability Metric",
      "Precision Target",
      "Status",
      "Final State",
      "Reference State",
      "Abs Error",
      "Rel Error",
      "Runtime ms",
      "Notes",
    ],
    odeRows,
    "OdeStressTable",
    {
      numberFormats: {
        "G2:J{end}": "0.000E+00",
        "O2:Q{end}": "0.000E+00",
      },
      widths: { A: 28, B: 38, C: 58, D: 26, K: 30, M: 36, N: 36, R: 58 },
    },
  );

  addPrecisionLog(workbook, data, { dtn, structural, cusp });
  addSources(workbook);

  const previews = [];
  for (const sheetName of [
    "Dashboard",
    "PDE_DtN",
    "Shape_Quadrature",
    "Q_Spectrum",
    "QBX_Failures",
    "QBX_HeadToHead",
    "Scaling_Fits",
    "Hard_Domains",
    "Planar_PDE",
    "Planar_Quadrature",
    "ODE_Stress",
    "Precision_Log",
    "Sources",
  ]) {
    const blob = await workbook.render({ sheetName, autoCrop: "all", scale: 0.85, format: "png" });
    const file = path.join(outputDir, `preview_${sanitizePreviewName(sheetName)}.png`);
    await fs.writeFile(file, new Uint8Array(await blob.arrayBuffer()));
    previews.push(file);
  }

  const inspect = await workbook.inspect({
    kind: "workbook,sheet,table,formula,drawing",
    include: "name,address,formula,value,type",
    range: "A1:P45",
  });
  await fs.writeFile(path.join(outputDir, "inspect_summary.ndjson"), inspect.ndjson);

  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(outputXlsx);

  console.log(
    JSON.stringify(
      {
        outputXlsx,
        sheetCount: 13,
        pdeRows: pdeRows.length,
        planarPdeRows: planarPdeRows.length,
        planarQuadratureRows: planarQuadratureRows.length,
        shapeRows: shapeRows.length,
        spectrumRows: spectrumRows.length,
        failureRows: failureRows.length,
        odeRows: odeRows.length,
        previews,
      },
      null,
      2,
    ),
  );
}

await main();
