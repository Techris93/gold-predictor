const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function loadTradeBrainTestApi() {
  const elements = new Map();
  const document = {
    activeElement: null,
    getElementById(id) {
      if (!elements.has(id)) {
        elements.set(id, {
          id,
          value: "",
          placeholder: "",
          textContent: "",
          dataset: {},
          className: "",
          checked: false,
          disabled: false,
          appendChild() {},
          addEventListener() {},
        });
      }
      return elements.get(id);
    },
    querySelectorAll() {
      return [];
    },
  };

  const window = {
    document,
    localStorage: {
      getItem() {
        return null;
      },
      setItem() {},
      removeItem() {},
    },
    crypto: {
      randomUUID() {
        return "trade-brain-test-user";
      },
    },
  };

  const context = vm.createContext({
    console,
    document,
    fetch: async () => {
      throw new Error("fetch should not be called in unit tests");
    },
    setTimeout,
    clearTimeout,
    window,
  });

  const source = fs.readFileSync(
    path.join(__dirname, "..", "static", "trade_brain.js"),
    "utf8",
  );
  vm.runInContext(source, context, { filename: "trade_brain.js" });
  return context.window.TradeBrainUI.__test;
}

test("blocked no-trade plans never qualify for auto-track", () => {
  const api = loadTradeBrainTestApi();
  const prediction = {
    verdict: "Bearish",
    ExecutionQuality: {
      direction: "Long",
      grade: "No Trade",
      status: "blocked",
      score: 58,
      setup: "VWAP Pullback Long >=2.5R",
      stopLoss: { price: 4688.35 },
    },
    StableDecision: {
      direction: "Long",
      decision_state: "SCANNING",
    },
  };

  const plan = api.resolvePredictionTradePlan(prediction, {
    price: 4696.6,
    atrDollar: 5.0,
  });

  assert.equal(plan.direction, "LONG");
  assert.equal(plan.stopLoss, null);
  assert.equal(plan.qualified, false);
  assert.equal(api.isPredictionQualifiedForAutoTrack(prediction), false);
});

test("final execution direction overrides raw verdict in the client plan", () => {
  const api = loadTradeBrainTestApi();
  const prediction = {
    verdict: "Bearish",
    ExecutionQuality: {
      direction: "Long",
      grade: "B",
      status: "ready",
      score: 82,
      setup: "VWAP Pullback Long >=2.5R",
      stopLoss: { price: 4688.35 },
    },
    StableDecision: {
      direction: "Long",
      decision_state: "CONFIRMED",
    },
  };

  const plan = api.resolvePredictionTradePlan(prediction, {
    price: 4696.6,
    atrDollar: 5.0,
  });

  assert.equal(api.derivePredictionDirection(prediction), "LONG");
  assert.equal(plan.direction, "LONG");
  assert.equal(plan.qualified, true);
  assert.equal(plan.stopLoss, 4688.35);
});

test("fallback stop only appears for actionable directional plans", () => {
  const api = loadTradeBrainTestApi();
  const prediction = {
    verdict: "Bullish",
    ExecutionQuality: {
      direction: "Long",
      grade: "B",
      status: "ready",
      score: 80,
      setup: "VWAP Reclaim Long",
      stopLoss: { price: null },
    },
    StableDecision: {
      direction: "Long",
      decision_state: "CONFIRMED",
    },
  };

  const plan = api.resolvePredictionTradePlan(prediction, {
    price: 100,
    atrDollar: 2,
  });

  assert.equal(plan.stopLoss, 97.5);
  assert.equal(
    api.predictionStopLoss(prediction, { price: 100, atrDollar: 2 }),
    97.5,
  );
});