#!/usr/bin/env node
"use strict";

const fs = require("fs");
const { chromium } = require("playwright");

const args = process.argv.slice(2);
const getArg = (name, fallback) => {
  const index = args.indexOf(name);
  if (index >= 0 && index + 1 < args.length) return args[index + 1];
  return fallback;
};

const BASE_URL = getArg("--url", process.env.Q_ENGINE_UI_URL || "http://127.0.0.1:8790/");
const CHROME_PATH = getArg(
  "--chrome",
  process.env.CHROME_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
);
const HEADLESS = !args.includes("--headed");

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function certificateByName(payload, name) {
  return payload?.production_certificate?.certificates?.find((item) => item.name === name);
}

function assertPassCertificate(payload, name, value) {
  const cert = certificateByName(payload, name);
  assert(cert, `missing production certificate: ${name}`);
  assert(cert.state === "pass", `${name} certificate did not pass: ${JSON.stringify(cert)}`);
  if (value !== undefined) assert(cert.value === value, `${name} certificate value mismatch: ${JSON.stringify(cert)}`);
  return cert;
}

async function rows(page, id) {
  return page.evaluate(
    (rootId) =>
      [...document.getElementById(rootId).querySelectorAll(".metric")].map((row) => [
        row.querySelector(".name")?.textContent?.trim(),
        row.querySelector(".value")?.textContent?.trim(),
      ]),
    id,
  );
}

async function row(page, id, name) {
  return page.evaluate(
    ({ rootId, rowName }) =>
      [...document.getElementById(rootId).querySelectorAll(".metric")]
        .find((item) => item.querySelector(".name")?.textContent?.trim() === rowName)
        ?.querySelector(".value")
        ?.textContent?.trim(),
    { rootId: id, rowName: name },
  );
}

async function statusText(page) {
  return page.evaluate(() => document.getElementById("status-text")?.textContent?.trim());
}

async function waitForStatus(page, predicate, timeoutMs = 45_000) {
  const started = Date.now();
  let lastValue;
  while (Date.now() - started < timeoutMs) {
    lastValue = await statusText(page);
    if (predicate(lastValue)) return lastValue;
    await page.waitForTimeout(150);
  }
  throw new Error(`Timed out waiting for status; last value=${lastValue}`);
}

async function waitForRow(page, id, name, predicate, timeoutMs = 45_000) {
  const started = Date.now();
  let lastValue;
  while (Date.now() - started < timeoutMs) {
    lastValue = await row(page, id, name);
    if (predicate(lastValue)) return lastValue;
    await page.waitForTimeout(150);
  }
  throw new Error(`Timed out waiting for ${id}:${name}; last value=${lastValue}`);
}

async function clickPreset(page, preset) {
  await page.locator(`button.preset[data-preset="${preset}"]`).click();
}

async function setPenMode(page, enabled) {
  const mode = await page.evaluate(() => ({ drawMode: state.drawMode, drawTool: state.drawTool }));
  if (enabled && !(mode.drawMode && mode.drawTool === "pen")) {
    await page.locator("#draw-toggle").click();
  }
  if (!enabled && mode.drawMode) await page.locator('[data-draw-tool="edit"]').click();
  await page.waitForFunction(
    (wanted) => document.getElementById("draw-toggle").classList.contains("active") === wanted && state.drawMode === wanted && (!wanted || state.drawTool === "pen"),
    enabled,
  );
}

async function clickSolve(page) {
  await page.locator("#solve-boundary").click();
}

async function stableProductionPostCount(page, productionPostCounter) {
  let last = productionPostCounter.count;
  let stableFor = 0;
  while (stableFor < 160) {
    await page.waitForTimeout(40);
    if (productionPostCounter.count === last) {
      stableFor += 40;
    } else {
      last = productionPostCounter.count;
      stableFor = 0;
    }
  }
  return productionPostCounter.count;
}

async function boundaryFramePreservation(page) {
  return page.evaluate(() => {
    const frame = (points) => {
      let x = 0, y = 0;
      for (const p of points) {
        x += p.x;
        y += p.y;
      }
      const center = { x: x / Math.max(1, points.length), y: y / Math.max(1, points.length) };
      let scale = 0;
      for (const p of points) scale = Math.max(scale, Math.hypot(p.x - center.x, p.y - center.y));
      return { center, scale: scale || 1 };
    };
    const raw = frame(state.rawDraw);
    const stored = frame(state.points);
    const boundary = frame(state.solution.boundary);
    return {
      raw,
      stored,
      boundary,
      storedCenterDelta: Math.hypot(stored.center.x - raw.center.x, stored.center.y - raw.center.y),
      boundaryCenterDelta: Math.hypot(boundary.center.x - raw.center.x, boundary.center.y - raw.center.y),
      storedScaleRel: Math.abs(stored.scale / raw.scale - 1),
      boundaryScaleRel: Math.abs(boundary.scale / raw.scale - 1),
    };
  });
}

async function verifyBlankStart(page, productionPostCounter) {
  await waitForRow(page, "metrics", "reference class", (actual) => actual === "blank canvas");
  await waitForRow(page, "metrics", "field source", (actual) => actual === "blank");
  await waitForRow(page, "metrics", "topology", (actual) => actual === "blank");
  await waitForRow(page, "metrics", "pen closure", (actual) => actual === "blank");
  await waitForRow(page, "production-q", "backend", (actual) => actual === "idle");
  await waitForRow(page, "production-q", "reason", (actual) => actual === "no boundary");
  await waitForStatus(page, (actual) => actual === "draw boundary");
  const blank = await page.evaluate(() => ({
    drawMode: state.drawMode,
    penActive: document.getElementById("draw-toggle").classList.contains("active"),
    activePresets: [...document.querySelectorAll(".preset.active")].map((item) => item.dataset.preset),
    points: state.points.length,
    boundary: state.solution?.boundary?.length || 0,
    isEmpty: state.solution?.isEmpty === true,
  }));
  assert(blank.drawMode === true && blank.penActive === true, "blank start should open with pen active");
  assert(blank.activePresets.length === 0, `blank start should not have an active preset: ${blank.activePresets.join(",")}`);
  assert(blank.points === 0 && blank.boundary === 0 && blank.isEmpty === true, `blank start created geometry: ${JSON.stringify(blank)}`);
  await page.waitForTimeout(350);
  assert(productionPostCounter.count === 0, `blank start submitted production Q before drawing: ${productionPostCounter.count}`);
  const pixels = await sampleCanvas(page);
  assert(pixels.nonWhite < 20 && pixels.dark === 0, `blank start canvas is not blank: ${JSON.stringify(pixels)}`);
  return {
    status: await statusText(page),
    backend: await row(page, "production-q", "backend"),
    fieldSource: await row(page, "metrics", "field source"),
    topology: await row(page, "metrics", "topology"),
    penActive: blank.penActive,
    activePresets: blank.activePresets,
    nonWhitePixels: pixels.nonWhite,
  };
}

async function dragCanvasTool(page, tool, options = {}) {
  await page.locator(`[data-draw-tool="${tool}"]`).click();
  if (options.sides) await page.locator("#polygon-sides").selectOption(String(options.sides));
  const box = await page.locator("#stage").boundingBox();
  assert(box, "stage canvas missing");
  const start = {
    x: box.x + box.width * (options.startX ?? 0.48),
    y: box.y + box.height * (options.startY ?? 0.50),
  };
  const end = {
    x: box.x + box.width * (options.endX ?? 0.70),
    y: box.y + box.height * (options.endY ?? 0.34),
  };
  await page.mouse.move(start.x, start.y);
  await page.mouse.down();
  await page.mouse.move(end.x, end.y, { steps: 10 });
  await page.waitForTimeout(40);
  const during = await page.evaluate(() => ({
    drawing: state.drawing,
    drawTool: state.drawTool,
    drawMode: state.drawMode,
    rawDrawLength: state.rawDraw.length,
    pointsLength: state.points.length,
    solutionBoundaryLength: state.solution?.boundary?.length || 0,
    solutionIsEmpty: state.solution?.isEmpty === true,
    fieldSource: [...document.getElementById("metrics").querySelectorAll(".metric")]
      .find((item) => item.querySelector(".name")?.textContent?.trim() === "field source")
      ?.querySelector(".value")
      ?.textContent
      ?.trim() || "",
    status: document.getElementById("status-text")?.textContent?.trim() || "",
    backend: state.backend?.rows?.find((item) => item[0] === "backend")?.[1] || "",
    reason: state.backend?.rows?.find((item) => item[0] === "reason")?.[1] || "",
  }));
  await page.mouse.up();
  return during;
}

async function drawPaletteSquiggle(page) {
  await page.locator('[data-draw-tool="squiggle"]').click();
  const box = await page.locator("#stage").boundingBox();
  assert(box, "stage canvas missing");
  const cx = box.x + box.width * 0.52;
  const cy = box.y + box.height * 0.50;
  const rx = Math.min(box.width, box.height) * 0.20;
  const ry = Math.min(box.width, box.height) * 0.14;
  await page.mouse.move(cx + rx, cy);
  await page.mouse.down();
  let during = null;
  for (let i = 1; i <= 34; i++) {
    const t = (Math.PI * 2 * i) / 34;
    const wobble = 1 + 0.18 * Math.sin(5 * t);
    await page.mouse.move(cx + rx * wobble * Math.cos(t), cy + ry * Math.sin(t), { steps: 2 });
    if (i === 20) {
      during = await page.evaluate(() => ({
        drawing: state.drawing,
        drawTool: state.drawTool,
        drawMode: state.drawMode,
        rawDrawLength: state.rawDraw.length,
        pointsLength: state.points.length,
        solutionBoundaryLength: state.solution?.boundary?.length || 0,
        solutionIsEmpty: state.solution?.isEmpty === true,
        fieldSource: [...document.getElementById("metrics").querySelectorAll(".metric")]
          .find((item) => item.querySelector(".name")?.textContent?.trim() === "field source")
          ?.querySelector(".value")
          ?.textContent
          ?.trim() || "",
        status: document.getElementById("status-text")?.textContent?.trim() || "",
        backend: state.backend?.rows?.find((item) => item[0] === "backend")?.[1] || "",
        reason: state.backend?.rows?.find((item) => item[0] === "reason")?.[1] || "",
      }));
    }
  }
  await page.mouse.up();
  return during;
}

async function verifyDrawingPalette(page, productionPostCounter) {
  const tools = await page.evaluate(() => [...document.querySelectorAll("[data-draw-tool]")].map((item) => item.dataset.drawTool));
  for (const tool of ["pen", "squiggle", "circle", "parallel", "polygon", "arc", "edit"]) {
    assert(tools.includes(tool), `drawing palette missing ${tool}`);
  }
  const cases = [
    { tool: "circle", closure: "circle compass closed" },
    { tool: "parallel", closure: "parallel-line strip closed", endX: 0.73, endY: 0.50 },
    { tool: "polygon", closure: "7-gon closed", sides: 7, geometry: "corner-preserving polygon", corners: "7" },
    { tool: "arc", closure: "arc lens closed", endX: 0.74, endY: 0.42 },
    { tool: "squiggle", closure: "auto-closed squiggle curve", freehand: true },
  ];
  const results = [];
  for (const item of cases) {
    await page.locator("#clear-shape").click();
    await waitForRow(page, "metrics", "reference class", (actual) => actual === "blank canvas");
    const beforePosts = await stableProductionPostCount(page, productionPostCounter);
    const during = item.freehand ? await drawPaletteSquiggle(page) : await dragCanvasTool(page, item.tool, item);
    assert(during?.drawing === true && during.drawMode === true && during.drawTool === item.tool, `${item.tool} did not enter drawing mode: ${JSON.stringify(during)}`);
    assert(during.pointsLength === 0 && during.solutionBoundaryLength === 0 && during.solutionIsEmpty === true, `${item.tool} instantiated a domain before release: ${JSON.stringify(during)}`);
    assert(during.fieldSource === "blank", `${item.tool} showed a field while drawing`);
    assert(during.backend === "idle", `${item.tool} submitted backend while drawing`);
    assert(during.reason === "open pen stroke", `${item.tool} should keep backend idle during open gesture`);
    assert(during.status === "drawing boundary", `${item.tool} status mismatch while drawing`);
    await waitForRow(page, "metrics", "reference class", (actual) => actual === "drawn");
    await waitForRow(page, "metrics", "pen closure", (actual) => actual === item.closure);
    await waitForRow(page, "metrics", "topology", (actual) => actual === "simple closed");
    await waitForRow(page, "metrics", "field source", (actual) => actual === "boundary only");
    await waitForRow(page, "production-q", "reason", (actual) => actual === "awaiting solve");
    await waitForStatus(page, (actual) => actual === "ready to solve");
    const afterClosurePosts = productionPostCounter.count - beforePosts;
    assert(afterClosurePosts === 0, `${item.tool} submitted production before Solve: ${afterClosurePosts}`);
    if (item.geometry) await waitForRow(page, "metrics", "geometry class", (actual) => actual === item.geometry);
    if (item.corners) await waitForRow(page, "metrics", "corner count", (actual) => actual === item.corners);
    await clickSolve(page);
    await waitForRow(page, "metrics", "field source", (actual) => actual === "production Q output");
    await waitForStatus(page, (actual) => actual === "production Q certified");
    assert(await row(page, "production-q", "dense matrix") === "not stored", `${item.tool} stored dense matrix`);
    assert(await row(page, "production-q", "pair table") === "not stored", `${item.tool} stored pair table`);
    results.push({
      tool: item.tool,
      closure: await row(page, "metrics", "pen closure"),
      geometry: await row(page, "metrics", "geometry class"),
      corners: await row(page, "metrics", "corner count"),
      postsBeforeSolve: afterClosurePosts,
      postsAfterSolve: productionPostCounter.count - beforePosts,
      status: await statusText(page),
    });
  }
  await page.locator("#clear-shape").click();
  await waitForRow(page, "metrics", "reference class", (actual) => actual === "blank canvas");
  return results;
}

async function selectPde(page, value) {
  await page.locator("#pde-select").selectOption(value);
  await waitForRow(page, "metrics", "field source", (actual) => actual === "boundary only");
  await waitForStatus(page, (actual) => actual === "ready to solve");
  await clickSolve(page);
  const problem = value === "laplace" ? "laplace_dtn" : value;
  await waitForRow(page, "production-q", "problem", (actual) => actual === problem);
  await waitForRow(page, "metrics", "field source", (actual) => actual === "production Q output");
  await waitForStatus(page, (actual) => actual === "production Q certified");
  const uiBound = Number.parseFloat(await row(page, "production-q", "UI error bound"));
  const visualResidual = Number.parseFloat(await row(page, "production-q", "visual residual"));
  assert(await row(page, "production-q", "backend") === "HarmonicZetaPlanarDomainQJet", `${value} backend mismatch`);
  assert(await row(page, "production-q", "protocol") === "planar_chord_qjet_harmonic_zeta_repaid", `${value} protocol mismatch`);
  assert(await row(page, "production-q", "ledger") === "borrowed_repaid", `${value} ledger was not repaid`);
  assert(await row(page, "production-q", "finite output") === "yes", `${value} output was not finite`);
  assert(await row(page, "production-q", "dense matrix") === "not stored", `${value} stored dense matrix`);
  assert(await row(page, "production-q", "pair table") === "not stored", `${value} stored pair table`);
  assert(await row(page, "production-q", "cert ledger") === "pass: borrowed_repaid", `${value} ledger certificate missing`);
  assert(await row(page, "production-q", "cert dense matrix") === "pass: not stored", `${value} dense matrix certificate missing`);
  assert(await row(page, "production-q", "cert pair table") === "pass: not stored", `${value} pair table certificate missing`);
  assert(await row(page, "production-q", "cert finite output") === "pass: yes", `${value} finite output certificate missing`);
  assert(Number.isFinite(uiBound) && uiBound <= 1e-12, `${value} arithmetic bound is not machine level: ${uiBound}`);
  assert(Number.isFinite(visualResidual), `${value} visual residual missing`);
  return {
    pde: value,
    problem,
    protocol: await row(page, "production-q", "protocol"),
    checksum: await row(page, "production-q", "checksum"),
    fieldSource: await row(page, "metrics", "field source"),
    finite: await row(page, "production-q", "finite output"),
    denseMatrix: await row(page, "production-q", "dense matrix"),
    pairTable: await row(page, "production-q", "pair table"),
    ledger: await row(page, "production-q", "ledger"),
    uiErrorBound: await row(page, "production-q", "UI error bound"),
    visualResidual: await row(page, "production-q", "visual residual"),
    status: await statusText(page),
  };
}

async function verifyExactReferencePresets(page) {
  const cases = [
    {
      preset: "circle",
      reference: "exact circle",
      accuracy: "machine precision disk reference",
    },
    {
      preset: "golden",
      reference: "golden ellipse chart",
      accuracy: "closed-form conic pullback reference",
    },
  ];
  const results = [];
  for (const item of cases) {
    await clickPreset(page, item.preset);
    await waitForRow(page, "metrics", "reference class", (actual) => actual === item.reference);
    await waitForRow(page, "metrics", "accuracy class", (actual) => actual === item.accuracy);
    await waitForRow(page, "metrics", "topology", (actual) => actual === "simple closed");
    await waitForRow(page, "metrics", "field source", (actual) => actual === "boundary only");
    await waitForStatus(page, (actual) => actual === "ready to solve");
    await clickSolve(page);
    await waitForRow(page, "metrics", "field source", (actual) => actual === "production Q output");
    await waitForStatus(page, (actual) => actual === "production Q certified");
    const uiBound = Number.parseFloat(await row(page, "metrics", "UI error bound"));
    assert(Number.isFinite(uiBound) && uiBound <= 1e-12, `${item.preset} UI error bound is not machine level: ${uiBound}`);
    assert(await row(page, "production-q", "backend") === "HarmonicZetaPlanarDomainQJet", `${item.preset} production backend mismatch`);
    assert(await row(page, "production-q", "accuracy class") === item.accuracy, `${item.preset} production accuracy class mismatch`);
    assert(await row(page, "production-q", "dense matrix") === "not stored", `${item.preset} stored dense matrix`);
    assert(await row(page, "production-q", "pair table") === "not stored", `${item.preset} stored pair table`);
    assert(await row(page, "production-q", "finite output") === "yes", `${item.preset} production output not finite`);
    results.push({
      preset: item.preset,
      reference: await row(page, "metrics", "reference class"),
      accuracy: await row(page, "metrics", "accuracy class"),
      uiErrorBound: await row(page, "metrics", "UI error bound"),
      backend: await row(page, "production-q", "backend"),
      finite: await row(page, "production-q", "finite output"),
      denseMatrix: await row(page, "production-q", "dense matrix"),
      pairTable: await row(page, "production-q", "pair table"),
    });
  }
  return results;
}

async function exerciseLiveDrag(page, productionPostCounter) {
  await setPenMode(page, false);
  await clickPreset(page, "lobe");
  await waitForRow(page, "metrics", "field source", (actual) => actual === "boundary only");
  await clickSolve(page);
  await waitForRow(page, "production-q", "backend", (actual) => actual === "HarmonicZetaPlanarDomainQJet");
  await waitForRow(page, "metrics", "field source", (actual) => actual === "production Q output");
  const starts = await page.evaluate(() => {
    const canvas = document.getElementById("stage");
    const rect = canvas.getBoundingClientRect();
    const boundary = state.solution.boundary;
    const candidateIndices = new Set([
      Math.floor(boundary.length * 0.02),
      Math.floor(boundary.length * 0.08),
      Math.floor(boundary.length * 0.16),
      Math.floor(boundary.length * 0.25),
      Math.floor(boundary.length * 0.5),
      Math.floor(boundary.length * 0.75),
    ]);
    let maxX = 0, minX = 0, maxY = 0, minY = 0;
    for (let i = 1; i < boundary.length; i++) {
      if (boundary[i].x > boundary[maxX].x) maxX = i;
      if (boundary[i].x < boundary[minX].x) minX = i;
      if (boundary[i].y > boundary[maxY].y) maxY = i;
      if (boundary[i].y < boundary[minY].y) minY = i;
    }
    [maxX, minX, maxY, minY].forEach((index) => candidateIndices.add(index));
    return [...candidateIndices].map((index) => {
      const point = worldToCanvas(boundary[index]);
      return {
        index,
        x: rect.left + (point.x / canvas.width) * rect.width,
        y: rect.top + (point.y / canvas.height) * rect.height,
      };
    });
  });
  let start = null;
  const beforePosts = await stableProductionPostCount(page, productionPostCounter);
  for (const candidate of starts) {
    await page.mouse.move(candidate.x, candidate.y);
    await page.mouse.down();
    await page.waitForTimeout(60);
    const engaged = await page.evaluate(() => state.interacting && state.dragIndex >= 0);
    if (engaged) {
      start = candidate;
      break;
    }
    await page.mouse.up();
    await page.waitForTimeout(40);
  }
  assert(start, `could not engage boundary drag from candidates: ${JSON.stringify(starts)}`);
  const samples = [];
  for (let index = 1; index <= 18; index++) {
    const t = index / 18;
    await page.mouse.move(start.x + 95 * t, start.y + 36 * Math.sin(Math.PI * t), { steps: 2 });
    await page.waitForTimeout(12);
    const sample = await page.evaluate(() => ({
      interacting: state.interacting,
      liveBoundaryN: state.liveBoundaryN,
      solutionN: state.solution?.n || 0,
      computeMs: state.solution?.computeMs || 0,
      backend: state.backend?.rows?.find((item) => item[0] === "backend")?.[1] || "",
      fieldSource: [...document.getElementById("metrics").querySelectorAll(".metric")]
        .find((item) => item.querySelector(".name")?.textContent?.trim() === "field source")
        ?.querySelector(".value")
        ?.textContent
        ?.trim() || "",
    }));
    samples.push(sample);
  }
  await waitForRow(page, "production-q", "backend", (actual) => actual === "idle", 5_000);
  const duringPosts = productionPostCounter.count;
  const during = await page.evaluate(() => ({
    interacting: state.interacting,
    liveBoundaryN: state.liveBoundaryN,
    solutionN: state.solution?.n || 0,
    backend: state.backend?.rows?.find((item) => item[0] === "backend")?.[1] || "",
    fieldSource: [...document.getElementById("metrics").querySelectorAll(".metric")]
      .find((item) => item.querySelector(".name")?.textContent?.trim() === "field source")
      ?.querySelector(".value")
      ?.textContent
      ?.trim() || "",
  }));
  assert(during.interacting === true, "drag did not enter live interaction mode");
  assert(during.liveBoundaryN === during.solutionN, "drag path remeshed instead of retaining live QJet boundary");
  assert(during.backend === "idle", "drag should reset production to idle");
  assert(during.fieldSource === "boundary only", "drag should show boundary only until Solve is pressed again");
  assert(await statusText(page) === "ready to solve", "header did not return to ready-to-solve during drag");
  assert(duringPosts === beforePosts, `production backend submitted during drag: before=${beforePosts} during=${duringPosts}`);
  await page.mouse.up();
  await waitForRow(page, "metrics", "field source", (actual) => actual === "boundary only");
  await waitForStatus(page, (actual) => actual === "ready to solve");
  await clickSolve(page);
  await waitForRow(page, "metrics", "field source", (actual) => actual === "production Q output");
  await waitForStatus(page, (actual) => actual === "production Q certified");
  const computeMs = samples.map((sample) => Number(sample.computeMs)).filter(Number.isFinite);
  const maxComputeMs = Math.max(...computeMs);
  const meanComputeMs = computeMs.reduce((a, b) => a + b, 0) / Math.max(1, computeMs.length);
  assert(maxComputeMs < 40, `drag compute exceeded frame budget: max=${maxComputeMs}`);
  assert(meanComputeMs < 12, `drag compute average too high: mean=${meanComputeMs}`);
  return {
    samples: samples.length,
    maxComputeMs,
    meanComputeMs,
    backendDuringDrag: during.backend,
    fieldSourceDuringDrag: during.fieldSource,
    productionPostsDuringDrag: duringPosts - beforePosts,
    productionPostsAfterRelease: productionPostCounter.count - beforePosts,
  };
}

async function drawSmoothCurve(page, productionPostCounter) {
  await setPenMode(page, true);
  const box = await page.locator("#stage").boundingBox();
  assert(box, "stage canvas missing");
  const cx = box.x + box.width * 0.5;
  const cy = box.y + box.height * 0.5;
  const rx = Math.min(box.width, box.height) * 0.22;
  const ry = Math.min(box.width, box.height) * 0.16;
  const beforePosts = await stableProductionPostCount(page, productionPostCounter);
  const samples = [];
  await page.mouse.move(cx + rx, cy);
  await page.mouse.down();
  for (let index = 1; index <= 36; index++) {
    const t = (Math.PI * 2 * index) / 36;
    const wobble = 1 + 0.12 * Math.cos(3 * t);
    await page.mouse.move(cx + rx * wobble * Math.cos(t), cy + ry * Math.sin(t), { steps: 2 });
    await page.waitForTimeout(8);
    if (index % 3 === 0) {
      samples.push(await page.evaluate(() => ({
        drawing: state.drawing,
        interacting: state.interacting,
        rawDrawLength: state.rawDraw.length,
        pointsLength: state.points.length,
        solutionBoundaryLength: state.solution?.boundary?.length || 0,
        solutionIsEmpty: state.solution?.isEmpty === true,
        computeMs: state.solution?.computeMs || 0,
        backend: state.backend?.rows?.find((item) => item[0] === "backend")?.[1] || "",
        reason: state.backend?.rows?.find((item) => item[0] === "reason")?.[1] || "",
        fieldSource: [...document.getElementById("metrics").querySelectorAll(".metric")]
          .find((item) => item.querySelector(".name")?.textContent?.trim() === "field source")
          ?.querySelector(".value")
          ?.textContent
          ?.trim() || "",
        status: document.getElementById("status-text")?.textContent?.trim() || "",
      })));
    }
  }
  await waitForRow(page, "production-q", "backend", (actual) => actual === "idle", 5_000);
  const duringPosts = productionPostCounter.count;
  const during = samples[samples.length - 1] || {};
  assert(during.drawing === true && during.interacting === true, "freehand draw did not remain in live interaction mode");
  assert(during.rawDrawLength > 12, `freehand draw did not collect enough points: ${during.rawDrawLength}`);
  assert(during.pointsLength === 0 && during.solutionBoundaryLength === 0 && during.solutionIsEmpty === true, `freehand draw created a domain before closure: ${JSON.stringify(during)}`);
  assert(during.backend === "idle" && during.reason === "open pen stroke", `freehand draw should keep production idle on open stroke: ${JSON.stringify(during)}`);
  assert(during.fieldSource === "blank", "freehand draw should not show a field before closure");
  assert(during.status === "drawing boundary", "header did not show boundary drawing state during freehand draw");
  assert(duringPosts === beforePosts, `production backend submitted during freehand draw: before=${beforePosts} during=${duringPosts}`);
  await page.mouse.up();
  await waitForRow(page, "metrics", "reference class", (actual) => actual === "drawn");
  const penClosure = await waitForRow(page, "metrics", "pen closure", (actual) => actual === "auto-closed pen curve");
  await waitForRow(page, "metrics", "field source", (actual) => actual === "boundary only");
  await waitForRow(page, "production-q", "reason", (actual) => actual === "awaiting solve");
  await waitForStatus(page, (actual) => actual === "ready to solve");
  const postsAfterClosure = productionPostCounter.count - beforePosts;
  assert(postsAfterClosure === 0, `freehand draw submitted production before Solve: ${postsAfterClosure}`);
  const framePreservation = await boundaryFramePreservation(page);
  assert(framePreservation.storedCenterDelta < 0.025 && framePreservation.boundaryCenterDelta < 0.025, `freehand draw recentered on closure: ${JSON.stringify(framePreservation)}`);
  assert(framePreservation.storedScaleRel < 0.04 && framePreservation.boundaryScaleRel < 0.04, `freehand draw rescaled on closure: ${JSON.stringify(framePreservation)}`);
  await clickSolve(page);
  await waitForRow(page, "metrics", "field source", (actual) => actual === "production Q output");
  await waitForStatus(page, (actual) => actual === "production Q certified");
  assert(productionPostCounter.count > beforePosts, "freehand draw did not submit production Q after Solve");
  const computeMs = samples.map((sample) => Number(sample.computeMs)).filter(Number.isFinite);
  const maxComputeMs = Math.max(...computeMs);
  const meanComputeMs = computeMs.reduce((a, b) => a + b, 0) / Math.max(1, computeMs.length);
  assert(maxComputeMs < 24, `freehand draw compute exceeded frame budget: max=${maxComputeMs}`);
  assert(meanComputeMs < 12, `freehand draw compute average too high: mean=${meanComputeMs}`);
  return {
    samples: samples.length,
    rawDrawLength: during.rawDrawLength,
    openStrokeOnly: during.solutionIsEmpty && during.pointsLength === 0 && during.solutionBoundaryLength === 0,
    maxComputeMs,
    meanComputeMs,
    penClosure,
    framePreservation,
    backendAfterRelease: await row(page, "production-q", "backend"),
    productionPostsDuringDraw: duringPosts - beforePosts,
    productionPostsAfterClosure: postsAfterClosure,
    productionPostsAfterSolve: productionPostCounter.count - beforePosts,
  };
}

async function drawSelfCrossingPen(page, productionPostCounter) {
  await setPenMode(page, true);
  const box = await page.locator("#stage").boundingBox();
  assert(box, "stage canvas missing");
  const cx = box.x + box.width * 0.5;
  const cy = box.y + box.height * 0.5;
  const r = Math.min(box.width, box.height) * 0.22;
  const beforePosts = await stableProductionPostCount(page, productionPostCounter);
  const corners = [
    [cx - r, cy - r * 0.72],
    [cx + r, cy + r * 0.72],
    [cx - r, cy + r * 0.72],
    [cx + r, cy - r * 0.72],
    [cx - r * 0.55, cy],
  ];
  await page.mouse.move(corners[0][0], corners[0][1]);
  await page.mouse.down();
  for (let i = 1; i < corners.length; i++) {
    await page.mouse.move(corners[i][0], corners[i][1], { steps: 10 });
    await page.waitForTimeout(16);
  }
  await waitForRow(page, "production-q", "backend", (actual) => actual === "idle", 5_000);
  const during = await page.evaluate(() => ({
    drawing: state.drawing,
    interacting: state.interacting,
    rawDrawLength: state.rawDraw.length,
    pointsLength: state.points.length,
    solutionBoundaryLength: state.solution?.boundary?.length || 0,
    solutionIsEmpty: state.solution?.isEmpty === true,
    backend: state.backend?.rows?.find((item) => item[0] === "backend")?.[1] || "",
    reason: state.backend?.rows?.find((item) => item[0] === "reason")?.[1] || "",
    fieldSource: [...document.getElementById("metrics").querySelectorAll(".metric")]
      .find((item) => item.querySelector(".name")?.textContent?.trim() === "field source")
      ?.querySelector(".value")
      ?.textContent
      ?.trim() || "",
    status: document.getElementById("status-text")?.textContent?.trim() || "",
  }));
  const duringPosts = productionPostCounter.count;
  assert(during.drawing === true && during.interacting === true, "self-crossing pen did not remain live");
  assert(during.rawDrawLength > 20, `self-crossing pen collected too few points: ${during.rawDrawLength}`);
  assert(during.pointsLength === 0 && during.solutionBoundaryLength === 0 && during.solutionIsEmpty === true, `self-crossing pen created a domain before closure: ${JSON.stringify(during)}`);
  assert(during.backend === "idle" && during.reason === "open pen stroke", `self-crossing pen should keep production idle on open stroke: ${JSON.stringify(during)}`);
  assert(during.fieldSource === "blank", "self-crossing pen should not show a field before closure");
  assert(during.status === "drawing boundary", "header did not show boundary drawing state for self-crossing pen");
  assert(duringPosts === beforePosts, `production backend submitted during self-crossing pen draw: before=${beforePosts} during=${duringPosts}`);
  await page.mouse.up();
  const penClosure = await waitForRow(
    page,
    "metrics",
    "pen closure",
    (actual) => typeof actual === "string" && actual.includes("self-intersection repaired"),
  );
  await waitForRow(page, "metrics", "topology", (actual) => actual === "simple closed");
  await waitForRow(page, "metrics", "field source", (actual) => actual === "boundary only");
  await waitForRow(page, "production-q", "reason", (actual) => actual === "awaiting solve");
  await waitForStatus(page, (actual) => actual === "ready to solve");
  const postsAfterClosure = productionPostCounter.count - beforePosts;
  assert(postsAfterClosure === 0, `self-crossing pen submitted production before Solve: ${postsAfterClosure}`);
  const framePreservation = await boundaryFramePreservation(page);
  assert(framePreservation.storedCenterDelta < 0.025 && framePreservation.boundaryCenterDelta < 0.025, `self-crossing repair recentered on closure: ${JSON.stringify(framePreservation)}`);
  assert(framePreservation.storedScaleRel < 0.08 && framePreservation.boundaryScaleRel < 0.08, `self-crossing repair rescaled on closure: ${JSON.stringify(framePreservation)}`);
  await clickSolve(page);
  await waitForRow(page, "metrics", "field source", (actual) => actual === "production Q output");
  await waitForStatus(page, (actual) => actual === "production Q certified");
  assert(productionPostCounter.count > beforePosts, "self-crossing pen did not submit production Q after Solve");
  const topology = await page.evaluate(() => state.solution.topology);
  assert(topology.simple === true && topology.intersections === 0, `repaired pen topology is not simple: ${JSON.stringify(topology)}`);
  return {
    rawDrawLength: during.rawDrawLength,
    openStrokeOnly: during.solutionIsEmpty && during.pointsLength === 0 && during.solutionBoundaryLength === 0,
    penClosure,
    framePreservation,
    topology,
    productionPostsDuringDraw: duringPosts - beforePosts,
    productionPostsAfterClosure: postsAfterClosure,
    productionPostsAfterSolve: productionPostCounter.count - beforePosts,
  };
}

async function injectSelfCrossingBoundary(page) {
  await page.evaluate(() => {
    state.customized = true;
    state.drawRepairStatus = "programmatic invalid boundary";
    state.liveBoundary = null;
    state.liveBoundaryN = 0;
    state.points = [
      { x: -0.82, y: -0.72 },
      { x: 0.82, y: 0.72 },
      { x: -0.82, y: 0.72 },
      { x: 0.82, y: -0.72 },
    ];
    invalidateProduction();
    requestFrame();
  });
  const topology = await waitForRow(
    page,
    "metrics",
    "topology",
    (actual) => typeof actual === "string" && actual.includes("crossings") && !actual.startsWith("0 "),
  );
  await waitForRow(page, "production-q", "backend", (actual) => actual === "not submitted");
  await waitForRow(page, "production-q", "reason", (actual) => actual === "boundary is not simple");
  await waitForStatus(page, (actual) => actual === "invalid boundary");
  assert(await row(page, "production-q", "dense matrix") === "not stored", "invalid boundary stored dense matrix");
  assert(await row(page, "production-q", "pair table") === "not stored", "invalid boundary stored pair table");
  return {
    topology,
    backend: await row(page, "production-q", "backend"),
    reason: await row(page, "production-q", "reason"),
    fieldSource: await row(page, "metrics", "field source"),
  };
}

async function sampleCanvas(page) {
  return page.evaluate(() => {
    const canvas = document.getElementById("stage");
    const ctx = canvas.getContext("2d");
    let sampled = 0;
    let nonWhite = 0;
    let dark = 0;
    let mid = 0;
    const stepX = Math.max(1, Math.floor(canvas.width / 100));
    const stepY = Math.max(1, Math.floor(canvas.height / 70));
    for (let y = 0; y < canvas.height; y += stepY) {
      for (let x = 0; x < canvas.width; x += stepX) {
        const data = ctx.getImageData(x, y, 1, 1).data;
        sampled += 1;
        const white = data[0] > 248 && data[1] > 248 && data[2] > 248;
        if (!white) nonWhite += 1;
        if (data[0] < 55 && data[1] < 55 && data[2] < 55) dark += 1;
        if (Math.abs(data[0] - data[1]) < 5 && Math.abs(data[1] - data[2]) < 5 && data[0] >= 55 && data[0] < 210) {
          mid += 1;
        }
      }
    }
    return { width: canvas.width, height: canvas.height, sampled, nonWhite, dark, mid };
  });
}

async function verifyDomainMaskedField(page) {
  return page.evaluate(() => {
    const canvas = document.getElementById("stage");
    const ctx = canvas.getContext("2d");
    let inside = 0;
    let outside = 0;
    let insideField = 0;
    let outsideField = 0;
    let insideDark = 0;
    let outsideDark = 0;
    const stepX = Math.max(1, Math.floor(canvas.width / 120));
    const stepY = Math.max(1, Math.floor(canvas.height / 90));
    for (let y = 3; y < canvas.height - 3; y += stepY) {
      for (let x = 3; x < canvas.width - 3; x += stepX) {
        const p = canvasToWorld({ x, y });
        const isInside = pointInside(p, state.solution.boundary);
        const data = ctx.getImageData(x, y, 1, 1).data;
        const gray = Math.abs(data[0] - data[1]) < 4 && Math.abs(data[1] - data[2]) < 4;
        const fieldLike = gray && data[0] < 225;
        const dark = data[0] < 65 && data[1] < 65 && data[2] < 65;
        if (isInside) {
          inside += 1;
          if (fieldLike) insideField += 1;
          if (dark) insideDark += 1;
        } else {
          outside += 1;
          if (fieldLike) outsideField += 1;
          if (dark) outsideDark += 1;
        }
      }
    }
    return {
      inside,
      outside,
      insideField,
      outsideField,
      insideFieldRate: insideField / Math.max(1, inside),
      outsideFieldRate: outsideField / Math.max(1, outside),
      insideDark,
      outsideDark,
    };
  });
}

async function verifyResponsiveLayout(page) {
  const viewports = [
    { name: "desktop", width: 1440, height: 920 },
    { name: "tablet", width: 900, height: 920 },
    { name: "phone", width: 390, height: 920 },
  ];
  const results = [];
  for (const viewport of viewports) {
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    await page.waitForTimeout(250);
    const layout = await page.evaluate(() => {
      const rectOf = (selector) => {
        const element = document.querySelector(selector);
        if (!element) return null;
        const rect = element.getBoundingClientRect();
        return {
          left: rect.left,
          right: rect.right,
          top: rect.top,
          bottom: rect.bottom,
          width: rect.width,
          height: rect.height,
        };
      };
      const controls = [...document.querySelectorAll(".btn, select, input")].map((element) => {
        const rect = element.getBoundingClientRect();
        return {
          text: element.textContent?.trim() || element.getAttribute("aria-label") || element.id || element.tagName,
          left: rect.left,
          right: rect.right,
          top: rect.top,
          width: rect.width,
          height: rect.height,
          scrollWidth: element.scrollWidth,
          clientWidth: element.clientWidth,
        };
      });
      const overflowingControls = controls.filter(
        (item) =>
          item.width > 0 &&
          (item.left < -1 ||
            item.right > window.innerWidth + 1 ||
            item.scrollWidth > item.clientWidth + 3),
      );
      const canvas = document.getElementById("stage");
      return {
        innerWidth: window.innerWidth,
        scrollWidth: document.documentElement.scrollWidth,
        topbar: rectOf(".topbar"),
        stage: rectOf(".stage"),
        canvas: rectOf("canvas#stage"),
        leftPanel: rectOf("aside.panel:not(.right-panel)"),
        rightPanel: rectOf(".right-panel"),
        drawButton: rectOf("#draw-toggle"),
        pdeSelect: rectOf("#pde-select"),
        exportButton: rectOf("#export-cert"),
        overflowingControls: overflowingControls.slice(0, 8),
        internalCanvas: { width: canvas.width, height: canvas.height },
      };
    });
    assert(layout.scrollWidth <= layout.innerWidth + 2, `${viewport.name} layout has horizontal overflow ${layout.scrollWidth} > ${layout.innerWidth}`);
    assert(layout.stage?.width > 300 && layout.stage?.height > 500, `${viewport.name} stage collapsed`);
    assert(layout.canvas?.width > 300 && layout.canvas?.height > 450, `${viewport.name} canvas collapsed`);
    assert(layout.drawButton?.width > 30 && layout.pdeSelect?.width > 120 && layout.exportButton?.width > 30, `${viewport.name} key controls collapsed`);
    assert(layout.overflowingControls.length === 0, `${viewport.name} controls overflow: ${JSON.stringify(layout.overflowingControls)}`);
    if (viewport.width <= 820) {
      assert(layout.stage.top < layout.leftPanel.top, `${viewport.name} stage should precede boundary panel`);
      assert(layout.leftPanel.top < layout.rightPanel.top, `${viewport.name} boundary panel should precede operator panel`);
    }
    if (viewport.width > 820 && viewport.width <= 1180) {
      assert(layout.stage.top < layout.leftPanel.top, `${viewport.name} stage should sit above lower panels`);
      assert(Math.abs(layout.leftPanel.top - layout.rightPanel.top) < 20, `${viewport.name} lower panels should align`);
    }
    const pixels = await sampleCanvas(page);
    assert(pixels.nonWhite > 250, `${viewport.name} canvas appears blank after resize`);
    results.push({
      name: viewport.name,
      width: viewport.width,
      height: viewport.height,
      stage: { width: layout.stage.width, height: layout.stage.height },
      canvas: { width: layout.canvas.width, height: layout.canvas.height },
      internalCanvas: layout.internalCanvas,
      nonWhitePixels: pixels.nonWhite,
      darkPixels: pixels.dark,
      midPixels: pixels.mid,
    });
  }
  await page.setViewportSize({ width: 1440, height: 920 });
  await page.waitForTimeout(200);
  return results;
}

async function exportCertificate(page) {
  const downloadPromise = page.waitForEvent("download", { timeout: 15_000 });
  await page.locator("#export-cert").click();
  const download = await downloadPromise;
  const path = await download.path();
  assert(path, "download path unavailable");
  return {
    filename: download.suggestedFilename(),
    payload: JSON.parse(fs.readFileSync(path, "utf8")),
  };
}

async function main() {
  const browser = await chromium.launch({ headless: HEADLESS, executablePath: CHROME_PATH });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 920 },
    acceptDownloads: true,
  });
  const consoleIssues = [];
  const productionPostCounter = { count: 0 };
  page.on("request", (request) => {
    if (request.method() === "POST" && request.url().includes("/api/production-q/verify")) {
      productionPostCounter.count += 1;
    }
  });
  page.on("console", (message) => {
    if (["warning", "error"].includes(message.type()) && !message.text().includes("getImageData")) {
      consoleIssues.push(`${message.type()}: ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => consoleIssues.push(`pageerror: ${error.message}`));

  const target = `${BASE_URL}${BASE_URL.includes("?") ? "&" : "?"}verify=${Date.now()}`;
  await page.goto(target, { waitUntil: "domcontentloaded" });
  const blankStart = await verifyBlankStart(page, productionPostCounter);
  await waitForRow(page, "exact-reference", "certificate", (actual) => actual === "pass");

  const drawingPalette = await verifyDrawingPalette(page, productionPostCounter);

  const exactReferences = await verifyExactReferencePresets(page);

  const liveDrag = await exerciseLiveDrag(page, productionPostCounter);

  await clickPreset(page, "polygon");
  await waitForRow(page, "metrics", "reference class", (actual) => actual === "polygon corner case");
  await waitForRow(page, "metrics", "accuracy class", (actual) => actual === "corner-corrected generated Q");
  await waitForRow(page, "metrics", "geometry class", (actual) => actual === "corner-preserving polygon");
  await waitForRow(page, "metrics", "corner treatment", (actual) => actual === "Mellin/BGK endpoint channel");
  await waitForRow(page, "metrics", "corner count", (actual) => actual === "8");
  await waitForRow(page, "metrics", "topology", (actual) => actual === "simple closed");
  await waitForRow(page, "metrics", "field source", (actual) => actual === "boundary only");
  await waitForStatus(page, (actual) => actual === "ready to solve");
  await clickSolve(page);
  await waitForRow(page, "metrics", "field source", (actual) => actual === "production Q output");
  await waitForStatus(page, (actual) => actual === "production Q certified");
  const polygonMinTurn = Number.parseFloat(await row(page, "metrics", "min turn"));
  const polygon = {
    metrics: (await rows(page, "metrics")).slice(0, 20),
    production: (await rows(page, "production-q")).slice(0, 24),
  };
  assert(Number.isFinite(polygonMinTurn) && polygonMinTurn < 135, `polygon min turn does not show a real corner: ${polygonMinTurn}`);
  assert(await row(page, "production-q", "dense matrix") === "not stored", "polygon stored dense matrix");
  assert(await row(page, "production-q", "pair table") === "not stored", "polygon stored pair table");
  assert(await row(page, "production-q", "Q error") === "cusp_endpoint_channel", "polygon did not expose corner/cusp channel");
  assert(await row(page, "production-q", "accuracy class") === "corner-corrected generated Q", "production rows lost accuracy class");
  const polygonUiBound = Number.parseFloat(await row(page, "production-q", "UI error bound"));
  const polygonVisualResidual = Number.parseFloat(await row(page, "production-q", "visual residual"));
  assert(Number.isFinite(polygonUiBound) && polygonUiBound <= 1e-12, `polygon production arithmetic bound is not machine level: ${polygonUiBound}`);
  assert(Number.isFinite(polygonVisualResidual), "polygon production rows lost visual residual");
  assert(await row(page, "production-q", "geometry class") === "corner-preserving polygon", "production rows lost polygon geometry class");
  assert(await row(page, "production-q", "corner treatment") === "Mellin/BGK endpoint channel", "production rows lost corner treatment");

  const polygonExported = await exportCertificate(page);
  assert(polygonExported.payload.geometry?.class === "corner-preserving polygon", "polygon export lost geometry class");
  assert(polygonExported.payload.geometry?.accuracy_class === "corner-corrected generated Q", "polygon export lost accuracy class");
  assert(polygonExported.payload.geometry?.ui_error_bound <= 1e-12, "polygon export arithmetic bound is not machine level");
  assert(Number.isFinite(polygonExported.payload.geometry?.visual_residual), "polygon export lost visual residual");
  assert(polygonExported.payload.geometry?.corner_count === 8, "polygon export lost corner count");
  assert(polygonExported.payload.geometry?.corner_treatment === "Mellin/BGK endpoint channel", "polygon export lost corner treatment");
  assert(polygonExported.payload.geometry?.pen_closure === "preset", "polygon export lost preset closure status");
  assert(polygonExported.payload.production_certificate?.dense_matrix_stored === false, "polygon production certificate stored dense matrix");
  assert(polygonExported.payload.production_certificate?.pair_weight_table_stored === false, "polygon production certificate stored pair table");
  assertPassCertificate(polygonExported.payload, "ledger", "borrowed_repaid");
  assertPassCertificate(polygonExported.payload, "dense matrix", "not stored");
  assertPassCertificate(polygonExported.payload, "pair table", "not stored");

  const pdeResults = [];
  for (const pde of ["heat", "poisson", "helmholtz", "wave", "laplace"]) {
    pdeResults.push(await selectPde(page, pde));
  }
  for (const item of pdeResults) {
    assert(item.fieldSource === "production Q output", `${item.pde} did not use production field`);
    assert(item.finite === "yes", `${item.pde} output was not finite`);
    assert(item.status === "production Q certified", `${item.pde} was not production certified`);
    assert(item.denseMatrix === "not stored" && item.pairTable === "not stored", `${item.pde} violated matrix-free storage`);
  }

  const selfCrossingPen = await drawSelfCrossingPen(page, productionPostCounter);

  const freehandDraw = await drawSmoothCurve(page, productionPostCounter);
  await waitForRow(page, "metrics", "topology", (actual) => actual === "simple closed");
  const drawn = {
    metrics: (await rows(page, "metrics")).slice(0, 20),
    production: (await rows(page, "production-q")).slice(0, 24),
  };
  assert(await row(page, "metrics", "reference class") === "drawn", "drawn curve did not settle as drawn");
  assert(await row(page, "metrics", "field source") === "production Q output", "drawn curve did not use production field");
  await waitForStatus(page, (actual) => actual === "production Q certified");

  const canvas = await sampleCanvas(page);
  assert(canvas.nonWhite > 300, "canvas appears blank");
  assert(canvas.dark > 10, "boundary/overlay dark pixels missing");
  assert(canvas.mid > 100, "field/reconstruction mid-tone pixels missing");

  const domainMask = await verifyDomainMaskedField(page);
  assert(domainMask.inside > 100 && domainMask.outside > 100, "domain mask sampling did not cover both inside and outside");
  assert(domainMask.insideFieldRate > 0.08, `field is not visible inside domain: ${JSON.stringify(domainMask)}`);
  assert(domainMask.outsideFieldRate < 0.035, `field leaked outside domain: ${JSON.stringify(domainMask)}`);

  const responsiveLayout = await verifyResponsiveLayout(page);

  const exported = await exportCertificate(page);
  const cert = exported.payload;
  assert(cert.production_certificate?.nodes === 64, "export missing 64-node production certificate");
  assert(cert.production_certificate?.points?.length === 64, "export missing production points");
  assert(cert.production_certificate?.output?.length === 64, "export missing production output");
  assert(cert.production_certificate?.finite_output === true, "export has non-finite production output");
  assert(cert.production_certificate?.stats?.protocol === "planar_chord_qjet_harmonic_zeta_repaid", "export has wrong protocol");
  assert(cert.production_certificate?.accuracy_class === "generated QJet bounded", "export production certificate lost drawn accuracy class");
  assert(cert.production_certificate?.ui_error_bound <= 1e-12, "drawn production certificate arithmetic bound is not machine level");
  assert(Number.isFinite(cert.production_certificate?.visual_residual), "drawn production certificate lost visual residual");
  assert(cert.geometry?.topology?.simple === true, "export missing simple-boundary topology certificate");
  assert(cert.geometry?.accuracy_class === "generated QJet bounded", "drawn export lost accuracy class");
  assert(cert.geometry?.ui_error_bound <= 1e-12, "drawn export arithmetic bound is not machine level");
  assert(Number.isFinite(cert.geometry?.visual_residual), "drawn export lost visual residual");
  assert(cert.geometry?.class === "smooth spectral chart", "drawn export should be smooth spectral chart");
  assert(cert.geometry?.pen_closure === "auto-closed pen curve", "drawn export lost pen closure status");
  assert(cert.exact_reference_suite?.exact_rows?.length === 35, "export missing exact reference rows");
  assert(cert.dense_matrix_stored === false, "export says dense matrix stored");
  assert(cert.pair_weight_table_stored === false, "export says pair table stored");
  assert(cert.production_certificate?.dense_matrix_stored === false, "drawn production certificate stored dense matrix");
  assert(cert.production_certificate?.pair_weight_table_stored === false, "drawn production certificate stored pair table");
  assertPassCertificate(cert, "ledger", "borrowed_repaid");
  assertPassCertificate(cert, "dense matrix", "not stored");
  assertPassCertificate(cert, "pair table", "not stored");

  const invalidBoundary = await injectSelfCrossingBoundary(page);

  await browser.close();
  assert(consoleIssues.length === 0, `browser console issues: ${consoleIssues.join("; ")}`);

  const summary = {
    ok: true,
    url: target,
    blankStart,
    drawingPalette,
    exactReferences,
    polygon,
    liveDrag,
    pdeResults,
    selfCrossingPen,
    freehandDraw,
    drawn,
    invalidBoundary,
    canvas,
    domainMask,
    responsiveLayout,
    export: {
      filename: exported.filename,
      polygonFilename: polygonExported.filename,
      productionNodes: cert.production_certificate.nodes,
      exactRows: cert.exact_reference_suite.exact_rows.length,
      protocol: cert.production_certificate.stats.protocol,
    },
  };
  console.log(JSON.stringify(summary, null, 2));
}

main().catch(async (error) => {
  console.error(error.stack || String(error));
  process.exit(1);
});
