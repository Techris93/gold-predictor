(function () {
  const STORAGE_KEY = "gold_predictor_trade_brain_user_id";
  const AUTO_TRACK_KEY = "gold_predictor_trade_brain_auto_track";
  const DEFAULT_AUTO_TRACK_ENABLED = true;
  const state = {
    dashboard: null,
    latestPrediction: null,
    socket: null,
    userId: null,
    initialized: false,
    autoTrackEnabled: DEFAULT_AUTO_TRACK_ENABLED,
    autoTrackInFlight: false,
    lastAutoTradeSignature: null,
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function safeText(value, fallback = "---") {
    if (typeof value === "string") {
      const trimmed = value.trim();
      return trimmed || fallback;
    }
    if (value === null || value === undefined) {
      return fallback;
    }
    return String(value);
  }

  function humanizeToken(value, fallback = "---") {
    const text = safeText(value, fallback);
    if (!text || text === fallback) {
      return fallback;
    }
    return text
      .replace(/[_-]+/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .replace(/\b\w/g, (match) => match.toUpperCase());
  }

  function asNumber(value, fallback = null) {
    if (value === null || value === undefined || value === "") {
      return fallback;
    }
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : fallback;
  }

  function formatPrice(value) {
    const numeric = asNumber(value, null);
    return numeric === null ? "---" : `$${numeric.toFixed(2)}`;
  }

  function formatCurrency(value) {
    const numeric = asNumber(value, null);
    return numeric === null
      ? "---"
      : `${numeric >= 0 ? "+" : ""}$${numeric.toFixed(2)}`;
  }

  function formatR(value) {
    const numeric = asNumber(value, null);
    return numeric === null
      ? "---"
      : `${numeric >= 0 ? "+" : ""}${numeric.toFixed(2)}R`;
  }

  function formatPercent(value) {
    const numeric = asNumber(value, null);
    return numeric === null ? "---" : `${numeric.toFixed(2)}%`;
  }

  function formatMultiplier(value) {
    const numeric = asNumber(value, null);
    return numeric === null ? "---" : `${numeric.toFixed(2)}x`;
  }

  function formatDateTime(value) {
    if (!value) {
      return "---";
    }
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime())
      ? safeText(value)
      : parsed.toLocaleString();
  }

  function getUserId() {
    if (state.userId) {
      return state.userId;
    }
    try {
      const existing = window.localStorage.getItem(STORAGE_KEY);
      if (existing) {
        state.userId = existing;
        return state.userId;
      }
      const generated =
        typeof window.crypto?.randomUUID === "function"
          ? window.crypto.randomUUID()
          : `tb-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
      window.localStorage.setItem(STORAGE_KEY, generated);
      state.userId = generated;
      return state.userId;
    } catch (_) {
      state.userId = "anonymous";
      return state.userId;
    }
  }

  function loadAutoTrackEnabled() {
    try {
      const stored = window.localStorage.getItem(AUTO_TRACK_KEY);
      if (stored === "1") {
        return true;
      }
      if (stored === "0") {
        return false;
      }
      window.localStorage.setItem(
        AUTO_TRACK_KEY,
        DEFAULT_AUTO_TRACK_ENABLED ? "1" : "0",
      );
    } catch (_) {
      return DEFAULT_AUTO_TRACK_ENABLED;
    }
    return DEFAULT_AUTO_TRACK_ENABLED;
  }

  function setAutoTrackEnabled(enabled) {
    state.autoTrackEnabled = Boolean(enabled);
    try {
      window.localStorage.setItem(
        AUTO_TRACK_KEY,
        state.autoTrackEnabled ? "1" : "0",
      );
    } catch (_) {
      // Ignore storage failures and keep the in-memory setting.
    }
    updateAutoTrackUi();
  }

  function buildHeaders(extraHeaders) {
    return {
      "x-user-id": getUserId(),
      ...(extraHeaders || {}),
    };
  }

  async function fetchJson(url, options) {
    const nextOptions = { ...(options || {}) };
    nextOptions.headers = buildHeaders(nextOptions.headers);
    if (nextOptions.body && typeof nextOptions.body !== "string") {
      nextOptions.body = JSON.stringify(nextOptions.body);
      nextOptions.headers["Content-Type"] = "application/json";
    }
    const response = await fetch(url, nextOptions);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(
        data.message || `Request failed with status ${response.status}`,
      );
    }
    return data;
  }

  function currentDashboard() {
    return state.dashboard && typeof state.dashboard === "object"
      ? state.dashboard
      : {};
  }

  function currentActiveTrade() {
    return currentDashboard().activeTrade || null;
  }

  function currentLearning() {
    return currentDashboard().learning &&
      typeof currentDashboard().learning === "object"
      ? currentDashboard().learning
      : {};
  }

  function currentLearningAdjustment() {
    return currentLearning().currentAdjustment &&
      typeof currentLearning().currentAdjustment === "object"
      ? currentLearning().currentAdjustment
      : {};
  }

  function currentRankedSetups() {
    const learning = currentLearning();
    if (Array.isArray(learning.rankedSetups)) {
      return learning.rankedSetups;
    }
    const adjustment = currentLearningAdjustment();
    return Array.isArray(adjustment.rankedSetups)
      ? adjustment.rankedSetups
      : [];
  }

  function currentTopSetup() {
    const adjustment = currentLearningAdjustment();
    if (adjustment.topSetup && typeof adjustment.topSetup === "object") {
      return adjustment.topSetup;
    }
    const ranked = currentRankedSetups();
    return ranked.length ? ranked[0] : null;
  }

  function currentSizingGuidance() {
    const learning = currentLearning();
    if (learning.sizing && typeof learning.sizing === "object") {
      return learning.sizing;
    }
    const adjustment = currentLearningAdjustment();
    return adjustment.sizing && typeof adjustment.sizing === "object"
      ? adjustment.sizing
      : {};
  }

  function setFeedback(message, tone) {
    const node = byId("tb-feedback");
    if (!node) {
      return;
    }
    node.textContent = safeText(message, "");
    node.dataset.tone = tone || "";
  }

  function currentSocketUserId(payload) {
    if (!payload || typeof payload !== "object") {
      return "";
    }
    const payloadUserId = safeText(payload.userId, "");
    if (payloadUserId) {
      return payloadUserId;
    }
    const tradeUserId = safeText(payload.trade?.userId, "");
    if (tradeUserId) {
      return tradeUserId;
    }
    return "";
  }

  function payloadMatchesCurrentUser(payload) {
    const payloadUserId = currentSocketUserId(payload);
    if (!payloadUserId) {
      return false;
    }
    return payloadUserId === getUserId();
  }

  function notifyTradeBrainLifecycle(title, body, notificationTag) {
    const notify = window.GoldPredictorNotifications?.notify;
    if (typeof notify !== "function") {
      return;
    }
    notify({
      title,
      body,
      notificationTag,
      ignoreBackgroundPushGate: true,
    });
  }

  function buildTradeCreatedNotification(trade) {
    const nextTrade = trade && typeof trade === "object" ? trade : {};
    const direction = safeText(nextTrade.direction, "Trade");
    const trigger = safeText(nextTrade.entry?.trigger, "Manual entry");
    return {
      title: `Trade Brain opened: ${direction}`,
      body: `${trigger} at ${formatPrice(nextTrade.entry?.price)}`,
      notificationTag: `trade-brain:created:${safeText(nextTrade.id, "unknown")}`,
    };
  }

  function buildTradeClosedNotification(trade) {
    const nextTrade = trade && typeof trade === "object" ? trade : {};
    const direction = safeText(nextTrade.direction, "Trade");
    const exitReason = humanizeToken(nextTrade.exit?.reason, "Closed");
    return {
      title: `Trade Brain closed: ${direction}`,
      body: `${formatR(nextTrade.exit?.finalR)} · ${exitReason}`,
      notificationTag: `trade-brain:closed:${safeText(nextTrade.id, "unknown")}`,
    };
  }

  function setText(id, value, fallback = "---") {
    const node = byId(id);
    if (!node) {
      return;
    }
    node.textContent = safeText(value, fallback);
  }

  function updateTradeBrainTogglePreview(dashboard = currentDashboard()) {
    const node = byId("tb-toggle-preview");
    if (!node) {
      return;
    }

    const nextDashboard =
      dashboard && typeof dashboard === "object" ? dashboard : {};
    const activeTrade =
      nextDashboard.activeTrade && typeof nextDashboard.activeTrade === "object"
        ? nextDashboard.activeTrade
        : null;
    const stats =
      nextDashboard.stats && typeof nextDashboard.stats === "object"
        ? nextDashboard.stats
        : {};
    const learning =
      nextDashboard.learning && typeof nextDashboard.learning === "object"
        ? nextDashboard.learning
        : {};
    const closedTrades = asNumber(stats.closedTrades, 0) || 0;
    const learnedContexts = asNumber(learning.learnedContexts, 0) || 0;

    let text = `No active trade · ${closedTrades} closed trades · ${learnedContexts} learned contexts`;
    let tone = "preview-watch";

    if (activeTrade) {
      const direction = safeText(activeTrade.direction, "Trade");
      const status = safeText(activeTrade.status, "ACTIVE");
      const trigger = safeText(activeTrade.entry?.trigger, "Manual entry");
      text = `${direction} ${status} · ${trigger} · ${formatR(activeTrade.live?.unrealizedR)}`;
      tone =
        direction === "LONG"
          ? "preview-bullish"
          : direction === "SHORT"
            ? "preview-bearish"
            : "preview-watch";
    } else if (closedTrades > 0) {
      const bestSetup = safeText(stats.bestSetup, "");
      text = `${closedTrades} closed trades · ${learnedContexts} learned contexts${bestSetup ? ` · Best ${bestSetup}` : ""}`;
    }

    node.className = `section-toggle-preview ${tone}`;
    node.textContent = text;
  }

  function renderList(id, items, emptyText, buildLine) {
    const node = byId(id);
    if (!node) {
      return;
    }
    node.textContent = "";
    if (!Array.isArray(items) || items.length === 0) {
      const item = document.createElement("li");
      item.textContent = emptyText;
      node.appendChild(item);
      return;
    }
    items.forEach((item) => {
      const child = document.createElement("li");
      child.textContent = buildLine(item);
      node.appendChild(child);
    });
  }

  function renderChips(id, items, emptyText, formatter) {
    const node = byId(id);
    if (!node) {
      return;
    }
    node.textContent = "";
    if (!Array.isArray(items) || items.length === 0) {
      const chip = document.createElement("span");
      chip.className = "trade-brain-chip";
      chip.textContent = emptyText;
      node.appendChild(chip);
      return;
    }
    items.slice(0, 4).forEach((item) => {
      const chip = document.createElement("span");
      chip.className = "trade-brain-chip";
      chip.textContent = formatter(item);
      node.appendChild(chip);
    });
  }

  function deriveMarketDataFromPrediction(prediction) {
    if (
      prediction &&
      prediction.TradeBrain &&
      prediction.TradeBrain.marketData
    ) {
      return prediction.TradeBrain.marketData;
    }
    const ta =
      prediction &&
      prediction.TechnicalAnalysis &&
      typeof prediction.TechnicalAnalysis === "object"
        ? prediction.TechnicalAnalysis
        : {};
    const priceAction =
      ta.price_action && typeof ta.price_action === "object"
        ? ta.price_action
        : {};
    return {
      price: asNumber(ta.current_price, null),
      adx: asNumber(ta.adx_14, 0),
      vwap: asNumber(ta.session_vwap || ta.vwap, null),
      atrDollar: asNumber(ta.atr_14, 0),
      atrPercent: asNumber(ta.atr_percent, 0),
      structure: safeText(priceAction.structure, "neutral"),
      regime: safeText(ta.volatility_regime?.market_regime, "normal"),
    };
  }

  function currentExecutionQuality(prediction = state.latestPrediction) {
    return prediction && typeof prediction.ExecutionQuality === "object"
      ? prediction.ExecutionQuality
      : {};
  }

  function currentStableDecision(prediction = state.latestPrediction) {
    return prediction && typeof prediction.StableDecision === "object"
      ? prediction.StableDecision
      : {};
  }

  function currentDecisionStatus(prediction = state.latestPrediction) {
    return prediction && typeof prediction.DecisionStatus === "object"
      ? prediction.DecisionStatus
      : {};
  }

  function normalizePredictionDirection(value) {
    const direction = safeText(value, "").toLowerCase();
    if (["long", "buy", "bullish"].includes(direction)) {
      return "LONG";
    }
    if (["short", "sell", "bearish"].includes(direction)) {
      return "SHORT";
    }
    return null;
  }

  function fallbackPredictionDirection(prediction) {
    if (!prediction || typeof prediction !== "object") {
      return null;
    }
    const verdict = safeText(prediction.verdict, "").toLowerCase();
    if (verdict.startsWith("bull")) {
      return "LONG";
    }
    if (verdict.startsWith("bear")) {
      return "SHORT";
    }
    const guidance =
      prediction.TradeGuidance && typeof prediction.TradeGuidance === "object"
        ? prediction.TradeGuidance
        : {};
    const buyLevel = safeText(guidance.buyLevel, "").toLowerCase();
    const sellLevel = safeText(guidance.sellLevel, "").toLowerCase();
    if (buyLevel === "strong" && sellLevel !== "strong") {
      return "LONG";
    }
    if (sellLevel === "strong" && buyLevel !== "strong") {
      return "SHORT";
    }
    return null;
  }

  function resolvePredictionSetup(prediction = state.latestPrediction) {
    const executionQuality = currentExecutionQuality(prediction);
    const setup = safeText(executionQuality.setup, "");
    if (setup && !/no clean entry|avoid/i.test(setup)) {
      return setup;
    }
    return safeText(currentTopSetup()?.setup, "Prediction sync");
  }

  function resolvePredictionTradePlan(
    prediction = state.latestPrediction,
    marketData,
  ) {
    const executionQuality = currentExecutionQuality(prediction);
    const stableDecision = currentStableDecision(prediction);
    const decisionStatus = currentDecisionStatus(prediction);
    const marketSnapshot =
      marketData || deriveMarketDataFromPrediction(prediction) || {};
    const direction =
      normalizePredictionDirection(stableDecision.direction) ||
      normalizePredictionDirection(executionQuality.direction) ||
      normalizePredictionDirection(decisionStatus.status) ||
      fallbackPredictionDirection(prediction);
    const grade = safeText(executionQuality.grade, "").toUpperCase();
    const status = safeText(executionQuality.status, "blocked").toLowerCase();
    const decisionState = safeText(
      stableDecision.decision_state || stableDecision.decisionState,
      "",
    ).toUpperCase();
    const price = asNumber(
      marketSnapshot?.price,
      asNumber(prediction?.TechnicalAnalysis?.current_price, null),
    );
    const atrDollar = asNumber(marketSnapshot?.atrDollar, null);
    const blocked =
      grade === "NO TRADE" ||
      status === "blocked" ||
      ["SCANNING", "INVALIDATED", "EXITED"].includes(decisionState);
    let stopLoss = asNumber(executionQuality.stopLoss?.price, null);
    if (blocked) {
      stopLoss = null;
    } else if (
      stopLoss === null &&
      price !== null &&
      atrDollar !== null &&
      direction
    ) {
      stopLoss =
        direction === "SHORT"
          ? price + atrDollar * 1.25
          : price - atrDollar * 1.25;
    }

    const qualified = Boolean(
      direction &&
      (grade.startsWith("A") || grade.startsWith("B")) &&
      status === "ready" &&
      !blocked &&
      price !== null &&
      stopLoss !== null &&
      Math.abs(price - stopLoss) > 0.0001,
    );

    return {
      direction,
      grade,
      status,
      decisionState,
      blocked,
      price,
      stopLoss,
      score: asNumber(executionQuality.score, null),
      setup: resolvePredictionSetup(prediction),
      qualified,
      marketData: marketSnapshot,
    };
  }

  function derivePredictionDirection(prediction) {
    return resolvePredictionTradePlan(prediction).direction;
  }

  function predictionSetupLabel(prediction = state.latestPrediction) {
    return resolvePredictionSetup(prediction);
  }

  function predictionStopLoss(prediction, marketData, direction) {
    const plan = resolvePredictionTradePlan(prediction, marketData);
    if (plan.direction && direction && plan.direction !== direction) {
      return null;
    }
    return plan.stopLoss;
  }

  function isPredictionQualifiedForAutoTrack(prediction) {
    return resolvePredictionTradePlan(prediction).qualified;
  }

  function syncFieldValue(node, nextValue) {
    if (!node) {
      return;
    }
    const activeElement =
      typeof document !== "undefined" ? document.activeElement : null;
    if (activeElement === node) {
      return;
    }
    const nextText =
      nextValue === null || nextValue === undefined ? "" : String(nextValue);
    if (node.value !== nextText) {
      node.value = nextText;
    }
  }

  function updateAutoTrackUi(prediction = state.latestPrediction) {
    const plan = resolvePredictionTradePlan(prediction);
    const toggle = byId("tb-auto-track");
    const summary = byId("tb-auto-track-summary");
    const note = byId("tb-auto-track-note");
    const createButton = byId("tb-create-btn");
    const activeTrade = currentActiveTrade();
    const direction = plan.direction;
    const setup = plan.setup;

    if (toggle) {
      toggle.checked = Boolean(state.autoTrackEnabled);
    }
    if (createButton) {
      createButton.textContent = state.autoTrackEnabled
        ? "Track Manually Now"
        : "Open Trade";
    }
    if (!summary || !note) {
      return;
    }

    note.textContent =
      "Auto-track creates an internal Trade Brain record only. It does not send broker orders.";

    if (!state.autoTrackEnabled) {
      summary.textContent =
        "Manual mode. Predictions pre-fill the form; tracking starts when you open a trade.";
      return;
    }

    if (state.autoTrackInFlight) {
      summary.textContent =
        "Auto mode is opening the current qualified prediction now.";
      return;
    }

    if (activeTrade) {
      summary.textContent =
        "Auto mode is on. A trade is already being tracked, so new predictions will not auto-open yet.";
      return;
    }

    if (plan.qualified) {
      summary.textContent = `${safeText(direction, "Trade")} setup qualified. Auto-track will open ${safeText(setup, "the next setup")} automatically.`;
      return;
    }

    summary.textContent =
      "Auto mode is on. Predictions still pre-fill the form until a qualified A/B setup appears.";
  }

  function maybePrefillForm(marketData) {
    const activeTrade = currentActiveTrade();
    if (activeTrade) {
      updateNoteHints();
      return;
    }
    const plan = resolvePredictionTradePlan(state.latestPrediction, marketData);
    const directionNode = byId("tb-direction");
    const entryNode = byId("tb-entry-price");
    const stopNode = byId("tb-stop-loss");
    const triggerNode = byId("tb-trigger");
    const positionNode = byId("tb-position-size");
    const sizing = currentSizingGuidance();
    if (directionNode && plan.direction) {
      syncFieldValue(directionNode, plan.direction);
    }
    if (entryNode && !entryNode.value) {
      const price = asNumber(marketData?.price, null);
      if (price !== null) {
        entryNode.value = price.toFixed(2);
      }
    }
    if (stopNode) {
      syncFieldValue(
        stopNode,
        plan.stopLoss !== null ? Math.max(0, plan.stopLoss).toFixed(2) : "",
      );
    }
    if (triggerNode && !triggerNode.value) {
      triggerNode.placeholder = plan.setup
        ? `Suggested: ${safeText(plan.setup, "VWAP reclaim")}`
        : "VWAP reclaim";
    }
    if (positionNode && !positionNode.value) {
      const scale = asNumber(sizing.positionScale, null);
      positionNode.placeholder =
        scale !== null && scale !== 1
          ? `Use ${formatMultiplier(scale)} of your base size`
          : "Use your base size";
    }
    updateNoteHints();
  }

  function buildAutoContextNote(triggerOverride) {
    const marketData =
      currentDashboard().marketData ||
      deriveMarketDataFromPrediction(state.latestPrediction) ||
      {};
    const plan = resolvePredictionTradePlan(state.latestPrediction, marketData);
    const direction = safeText(
      plan.direction || byId("tb-direction")?.value,
      "LONG",
    );
    const rawTrigger = safeText(byId("tb-trigger")?.value, "");
    const topSetup = currentTopSetup();
    const sizing = currentSizingGuidance();
    const adjustment = currentLearningAdjustment();
    const trigger =
      safeText(triggerOverride, "") ||
      rawTrigger ||
      safeText(topSetup?.setup, "Manual entry");
    const structureLabel = humanizeToken(marketData.structure, "Neutral");
    const regimeLabel = humanizeToken(marketData.regime, "Normal");
    const parts = [
      safeText(marketData.session, "Unknown session"),
      `${structureLabel} / ${regimeLabel}`,
      `${direction} bias`,
      trigger ? `trigger ${trigger}` : null,
    ];
    if (adjustment.active) {
      const delta = asNumber(adjustment.confidenceDelta, 0) || 0;
      parts.push(`${delta >= 0 ? "+" : ""}${delta} confidence`);
    }
    if (sizing.active) {
      parts.push(`${formatMultiplier(sizing.positionScale)} base size`);
    }
    return parts.filter(Boolean).join(" · ");
  }

  function buildAutoTradePayload(prediction) {
    const marketData =
      currentDashboard().marketData ||
      deriveMarketDataFromPrediction(prediction);
    const plan = resolvePredictionTradePlan(prediction, marketData);
    const sizing = currentSizingGuidance();
    const direction = plan.direction;
    const trigger = plan.setup;
    const price = asNumber(
      marketData?.price,
      asNumber(prediction?.TechnicalAnalysis?.current_price, null),
    );
    const stopLoss = plan.stopLoss;
    const riskDollar = asNumber(byId("tb-risk-dollar")?.value, 100);
    const positionSize = asNumber(byId("tb-position-size")?.value, 0);

    return {
      direction,
      price,
      stopLoss,
      riskDollar,
      positionSize,
      trigger,
      context: {
        ...marketData,
        trigger,
        confidence:
          asNumber(prediction?.confidence, null) !== null
            ? Number(prediction.confidence) / 100
            : 0.7,
        note: `${buildAutoContextNote(trigger)} · auto-tracked from prediction`,
        source: "auto_prediction",
        recommendedPositionScale: asNumber(sizing.positionScale, null),
        suggestedRiskPercent: asNumber(sizing.suggestedRiskPercent, null),
      },
    };
  }

  function buildAutoTrackSignature(prediction, payload) {
    const plan = resolvePredictionTradePlan(prediction);
    return JSON.stringify({
      direction: payload.direction,
      trigger: payload.trigger,
      price: asNumber(payload.price, null),
      stopLoss: asNumber(payload.stopLoss, null),
      grade: plan.grade,
      status: plan.status,
      decisionState: plan.decisionState,
      score: plan.score,
    });
  }

  async function maybeAutoTrackPrediction(prediction) {
    updateAutoTrackUi(prediction);
    if (
      !state.autoTrackEnabled ||
      state.autoTrackInFlight ||
      currentActiveTrade() ||
      !isPredictionQualifiedForAutoTrack(prediction)
    ) {
      return;
    }

    const payload = buildAutoTradePayload(prediction);
    if (
      payload.price === null ||
      payload.stopLoss === null ||
      Math.abs(payload.price - payload.stopLoss) <= 0.0001
    ) {
      updateAutoTrackUi(prediction);
      return;
    }

    const signature = buildAutoTrackSignature(prediction, payload);
    if (signature === state.lastAutoTradeSignature) {
      return;
    }

    state.autoTrackInFlight = true;
    updateAutoTrackUi(prediction);
    try {
      const data = await fetchJson("/api/trades", {
        method: "POST",
        body: payload,
      });
      state.lastAutoTradeSignature = signature;
      renderDashboard(data.dashboard || {});
      setFeedback("Qualified prediction auto-tracked.", "good");
    } catch (error) {
      const message = safeText(error?.message, "");
      if (!/active trade already exists/i.test(message)) {
        setFeedback(message || "Could not auto-track the prediction.", "bad");
      }
    } finally {
      state.autoTrackInFlight = false;
      updateAutoTrackUi(prediction);
    }
  }

  function buildAutoEmotionNote(emotion) {
    const activeTrade = currentActiveTrade();
    const marketData =
      currentDashboard().marketData ||
      deriveMarketDataFromPrediction(state.latestPrediction) ||
      {};
    const live = activeTrade?.live || {};
    const trailing = live.trailing?.active
      ? "trailing active"
      : "trailing inactive";
    return [
      safeText(emotion, "focused"),
      formatR(live.unrealizedR),
      safeText(live.status || activeTrade?.status, "stand by").toLowerCase(),
      trailing,
      formatPrice(live.currentPrice || marketData.price),
    ].join(" · ");
  }

  function updateNoteHints(previewEmotion = "focused") {
    const contextHint = byId("tb-context-note-hint");
    const contextField = byId("tb-context-note");
    if (contextHint) {
      contextHint.textContent = contextField?.value?.trim()
        ? "Manual note will be stored instead of the auto-generated context capture."
        : `Auto if blank: ${buildAutoContextNote()}`;
    }

    const emotionHint = byId("tb-emotion-note-hint");
    const emotionField = byId("tb-emotion-note");
    const activeTrade = currentActiveTrade();
    if (emotionHint) {
      emotionHint.textContent = emotionField?.value?.trim()
        ? "Manual note will be stored instead of the auto-generated emotion capture."
        : activeTrade
          ? `Auto if blank: ${buildAutoEmotionNote(previewEmotion)}`
          : "Auto if blank: the app will log the emotion, price, unrealized R, and trailing state.";
    }
  }

  function renderStats(stats) {
    setText("tb-stat-total", safeText(stats.totalTrades, "0"), "0");
    setText("tb-stat-win-rate", formatPercent(stats.winRate), "0%");
    setText("tb-stat-avg-r", formatR(stats.avgR), "0.00R");
    setText("tb-stat-best-setup", safeText(stats.bestSetup, "---"));
  }

  function renderActiveTrade(activeTrade, marketData) {
    const closeButton = byId("tb-close-btn");
    const snapshotButton = byId("tb-snapshot-btn");
    const statusPill = byId("tb-status-pill");
    const summaryNode = byId("tb-summary");
    const emotionButtons = document.querySelectorAll(
      "#tb-emotion-actions button",
    );
    const hasActiveTrade = Boolean(
      activeTrade && typeof activeTrade === "object",
    );

    if (!hasActiveTrade) {
      setText("tb-active-title", "No active trade", "No active trade");
      setText("tb-active-status", "Stand by");
      setText("tb-active-pnl", "---");
      setText("tb-active-price", formatPrice(marketData?.price));
      setText("tb-active-stop", "---");
      setText("tb-active-targets", "---");
      setText("tb-active-trailing", "Inactive");
      setText("tb-active-session", safeText(marketData?.session, "---"));
      setText(
        "tb-active-context",
        `${humanizeToken(marketData?.structure, "Neutral")} / ${humanizeToken(marketData?.regime, "Normal")}`,
      );
      renderList(
        "tb-decision-log",
        [],
        "Waiting for an active trade.",
        () => "",
      );
      renderList("tb-stop-history", [], "No stop updates yet.", () => "");
      if (summaryNode) {
        summaryNode.textContent =
          "Track one live XAUUSD trade with stop automation, decision memory, and post-trade review.";
      }
      if (statusPill) {
        statusPill.textContent = "No Active Trade";
        statusPill.dataset.tone = "watch";
      }
      if (closeButton) closeButton.disabled = true;
      if (snapshotButton) snapshotButton.disabled = true;
      emotionButtons.forEach((button) => {
        button.disabled = true;
      });
      return;
    }

    const live = activeTrade.live || {};
    const plan = activeTrade.plan || {};
    const entry = activeTrade.entry || {};
    const trailing = live.trailing || {};
    if (summaryNode) {
      summaryNode.textContent =
        `${activeTrade.direction} trade ${safeText(entry.trigger, "Manual entry")} in ${safeText(entry.session, "Unknown session")}. ` +
        `Current regime ${humanizeToken(entry.context?.regime, "Normal")}.`;
    }
    if (statusPill) {
      statusPill.textContent = `${safeText(activeTrade.direction)} ${safeText(activeTrade.status)}`;
      statusPill.dataset.tone = "active";
    }
    if (closeButton) closeButton.disabled = false;
    if (snapshotButton) snapshotButton.disabled = false;
    emotionButtons.forEach((button) => {
      button.disabled = !state.socket;
    });
    setText(
      "tb-active-title",
      `${safeText(activeTrade.direction)} @ ${formatPrice(entry.price)}`,
    );
    setText(
      "tb-active-status",
      safeText(live.status || activeTrade.status, "ACTIVE"),
    );
    setText(
      "tb-active-pnl",
      `${formatCurrency(live.unrealizedPnL)} · ${formatR(live.unrealizedR)}`,
    );
    setText(
      "tb-active-price",
      formatPrice(live.currentPrice || marketData?.price),
    );
    setText("tb-active-stop", formatPrice(live.stopLoss?.current));
    setText(
      "tb-active-targets",
      `${formatPrice(plan.tp1)} / ${formatPrice(plan.tp2)}`,
    );
    setText(
      "tb-active-trailing",
      trailing.active
        ? `${safeText(trailing.method, "Trail")} · buffer ${formatPrice(trailing.buffer)}`
        : "Inactive",
    );
    setText("tb-active-session", safeText(entry.session, "---"));
    setText(
      "tb-active-context",
      `${humanizeToken(entry.context?.structure, "Neutral")} / ${humanizeToken(entry.context?.regime, "Normal")}`,
    );

    renderList(
      "tb-decision-log",
      Array.isArray(live.decisions) ? live.decisions.slice(-4).reverse() : [],
      "No decisions recorded yet.",
      (decision) =>
        `${safeText(decision.action)} · ${safeText(decision.reasoning)} · ${formatPrice(decision.price)}`,
    );
    renderList(
      "tb-stop-history",
      Array.isArray(live.stopLoss?.history)
        ? live.stopLoss.history.slice(-4).reverse()
        : [],
      "No stop updates yet.",
      (stop) =>
        `${safeText(stop.type)} · ${formatPrice(stop.price)} · ${safeText(stop.reason)}`,
    );
  }

  function renderRecentTrades(recentTrades) {
    const node = byId("tb-recent-trades");
    if (!node) {
      return;
    }
    node.textContent = "";
    if (!Array.isArray(recentTrades) || recentTrades.length === 0) {
      const empty = document.createElement("div");
      empty.className = "trade-brain-empty";
      empty.textContent =
        "Closed trades will appear here after the first review cycle.";
      node.appendChild(empty);
      return;
    }

    recentTrades.slice(0, 6).forEach((trade) => {
      const item = document.createElement("div");
      item.className = "trade-brain-recent-item";

      const head = document.createElement("div");
      head.className = "trade-brain-recent-head";
      head.textContent = `${safeText(trade.direction)} · ${safeText(trade.entry?.trigger, "Manual entry")}`;
      item.appendChild(head);

      const sub = document.createElement("div");
      sub.className = "trade-brain-recent-sub";
      const resultText = trade.exit
        ? `${formatR(trade.exit.finalR)} · ${safeText(trade.exit.reason, "Closed")}`
        : `${safeText(trade.status, "ACTIVE")} · ${formatPrice(trade.live?.currentPrice)}`;
      sub.textContent = `${resultText} · ${formatDateTime(trade.updatedAt || trade.createdAt)}`;
      item.appendChild(sub);

      node.appendChild(item);
    });
  }

  function renderReview(recentTrades) {
    const reviewNode = byId("tb-review-grade");
    const lessonsNode = byId("tb-review-lessons");
    if (!reviewNode || !lessonsNode) {
      return;
    }
    lessonsNode.textContent = "";

    const latestReviewedTrade = Array.isArray(recentTrades)
      ? recentTrades.find(
          (trade) => trade && trade.status === "CLOSED" && trade.review,
        )
      : null;
    if (!latestReviewedTrade) {
      reviewNode.textContent =
        "No closed trades yet. The latest review grade and comparison will appear here.";
      const item = document.createElement("li");
      item.textContent =
        "Post-trade lessons will populate after the first closed trade.";
      lessonsNode.appendChild(item);
      return;
    }

    const review = latestReviewedTrade.review || {};
    reviewNode.textContent = `Grade ${safeText(review.grade, "C")} · ${formatR(review.comparison?.yourR)} vs avg ${formatR(review.comparison?.avgSimilarR)}`;
    const lessons = Array.isArray(review.lessons) ? review.lessons : [];
    lessons.forEach((lesson) => {
      const item = document.createElement("li");
      item.textContent = lesson;
      lessonsNode.appendChild(item);
    });
  }

  function renderLearning(learning) {
    const statusNode = byId("tb-learning-status");
    if (!statusNode) {
      return;
    }

    const payload = learning && typeof learning === "object" ? learning : {};
    const adjustment =
      payload.currentAdjustment && typeof payload.currentAdjustment === "object"
        ? payload.currentAdjustment
        : {};
    const closedSamples = asNumber(payload.closedSamples, 0) || 0;
    const learnedContexts = asNumber(payload.learnedContexts, 0) || 0;
    const baseText = `${humanizeToken(payload.mode, "Contextual Bandit")} · ${closedSamples} closed trades · ${learnedContexts} learned contexts`;

    if (adjustment.active) {
      const delta = asNumber(adjustment.confidenceDelta, 0) || 0;
      statusNode.textContent = `${baseText} · ${delta >= 0 ? "+" : ""}${delta} confidence · ${safeText(adjustment.message, "Learning signal active.")}`;
    } else {
      statusNode.textContent = `${baseText} · ${safeText(adjustment.message, "Learning engine warming up.")}`;
    }

    const currentMatches = Array.isArray(adjustment.matchedContexts)
      ? adjustment.matchedContexts
      : [];
    const topEdges = Array.isArray(payload.topEdges) ? payload.topEdges : [];
    const topRisks = Array.isArray(payload.topRisks) ? payload.topRisks : [];
    const rankedSetups = Array.isArray(payload.rankedSetups)
      ? payload.rankedSetups
      : Array.isArray(adjustment.rankedSetups)
        ? adjustment.rankedSetups
        : [];
    const topSetup =
      payload.topSetup && typeof payload.topSetup === "object"
        ? payload.topSetup
        : adjustment.topSetup && typeof adjustment.topSetup === "object"
          ? adjustment.topSetup
          : rankedSetups[0] || null;
    const sizing =
      payload.sizing && typeof payload.sizing === "object"
        ? payload.sizing
        : adjustment.sizing && typeof adjustment.sizing === "object"
          ? adjustment.sizing
          : {};
    const chipItems = topEdges.length
      ? topEdges
      : topRisks.length
        ? topRisks
        : currentMatches;

    renderChips(
      "tb-learning-contexts",
      chipItems,
      "Learning contexts will appear here",
      (item) => {
        const label = safeText(item.label || item.contextLabel, "Context");
        const samples = safeText(item.samples, 0);
        const reward = formatR(
          item.avgReward !== undefined ? item.avgReward : item.qValue,
        );
        return `${label} · ${samples}t · ${reward}`;
      },
    );

    const setupNode = byId("tb-learning-setup");
    if (setupNode) {
      if (topSetup?.setup) {
        const recommendation = asNumber(topSetup.recommendationScore, 0) || 0;
        const tone = recommendation < 0 ? "Avoid" : "Prefer";
        setupNode.textContent = `${tone} ${safeText(topSetup.setup)} · ${safeText(topSetup.samples, 0)}t · ${formatR(topSetup.avgReward)}`;
      } else {
        setupNode.textContent =
          "Setup ranking will appear after a few closed trades.";
      }
    }

    const sizingNode = byId("tb-learning-sizing");
    if (sizingNode) {
      sizingNode.textContent = safeText(
        sizing.summary,
        "Use base size until reinforcement memory has enough matching trades.",
      );
    }

    renderChips(
      "tb-learning-setups",
      rankedSetups,
      "No ranked setups yet",
      (item) =>
        `${safeText(item.setup)} · ${safeText(item.samples, 0)}t · ${formatR(item.avgReward)}`,
    );
    updateNoteHints();
  }

  function renderDashboard(dashboard) {
    state.dashboard =
      dashboard && typeof dashboard === "object" ? dashboard : {};
    const nextDashboard = currentDashboard();
    renderStats(nextDashboard.stats || {});
    renderActiveTrade(
      nextDashboard.activeTrade,
      nextDashboard.marketData ||
        deriveMarketDataFromPrediction(state.latestPrediction),
    );
    renderRecentTrades(nextDashboard.recentTrades || []);
    renderReview(nextDashboard.recentTrades || []);
    renderLearning(nextDashboard.learning || {});
    renderChips(
      "tb-analytics-setups",
      nextDashboard.analytics?.setups || [],
      "Setups will appear here",
      (item) =>
        `${safeText(item.setup)} · ${formatR(item.avgR)} · ${safeText(item.trades, 0)}t`,
    );
    renderChips(
      "tb-analytics-sessions",
      nextDashboard.analytics?.sessions || [],
      "Sessions will appear here",
      (item) => `${safeText(item.session)} · ${formatR(item.avgR)}`,
    );
    renderChips(
      "tb-analytics-emotions",
      nextDashboard.analytics?.emotions || [],
      "Emotion tags will appear here",
      (item) => `${safeText(item.emotion)} · ${safeText(item.count, 0)}`,
    );
    updateTradeBrainTogglePreview(nextDashboard);
    maybePrefillForm(
      nextDashboard.marketData ||
        deriveMarketDataFromPrediction(state.latestPrediction),
    );
    updateAutoTrackUi();
  }

  function dashboardHasTrackedData(dashboard) {
    if (!dashboard || typeof dashboard !== "object") {
      return false;
    }
    const stats =
      dashboard.stats && typeof dashboard.stats === "object"
        ? dashboard.stats
        : {};
    return (
      Boolean(dashboard.activeTrade) ||
      (asNumber(stats.totalTrades, 0) || 0) > 0
    );
  }

  function maybeEvaluateActiveTrade(prediction = state.latestPrediction) {
    const activeTrade = currentActiveTrade();
    if (!activeTrade || !state.socket) {
      return;
    }

    const derivedMarketData =
      currentDashboard().marketData ||
      deriveMarketDataFromPrediction(prediction) ||
      {};
    const price = asNumber(
      derivedMarketData.price,
      asNumber(prediction?.TechnicalAnalysis?.current_price, null),
    );
    if (price === null) {
      return;
    }

    state.socket.emit("trade:evaluate", {
      tradeId: activeTrade.id,
      price,
      marketData: {
        ...derivedMarketData,
        price,
      },
      userId: getUserId(),
    });
  }

  function buildCreatePayload() {
    const direction = safeText(byId("tb-direction")?.value, "LONG");
    const price = asNumber(byId("tb-entry-price")?.value, null);
    const stopLoss = asNumber(byId("tb-stop-loss")?.value, null);
    const riskDollar = asNumber(byId("tb-risk-dollar")?.value, 0);
    const positionSize = asNumber(byId("tb-position-size")?.value, 0);
    const trigger =
      safeText(byId("tb-trigger")?.value, "") ||
      safeText(currentTopSetup()?.setup, "Manual entry");
    const contextNote = safeText(
      byId("tb-context-note")?.value,
      buildAutoContextNote(trigger),
    );
    const sizing = currentSizingGuidance();
    const marketData =
      currentDashboard().marketData ||
      deriveMarketDataFromPrediction(state.latestPrediction) ||
      {};
    return {
      direction,
      price,
      stopLoss,
      riskDollar,
      positionSize,
      trigger,
      context: {
        ...marketData,
        trigger,
        confidence:
          asNumber(state.latestPrediction?.confidence, null) !== null
            ? Number(state.latestPrediction.confidence) / 100
            : 0.7,
        note: contextNote,
        source: "manual",
        recommendedPositionScale: asNumber(sizing.positionScale, null),
        suggestedRiskPercent: asNumber(sizing.suggestedRiskPercent, null),
      },
    };
  }

  async function refreshDashboard() {
    try {
      const data = await fetchJson("/api/trades/active");
      renderDashboard(data.dashboard || {});
    } catch (error) {
      setFeedback(error.message || "Trade Brain is unavailable.", "bad");
    }
  }

  async function handleCreateTrade(event) {
    event.preventDefault();
    try {
      const payload = buildCreatePayload();
      const data = await fetchJson("/api/trades", {
        method: "POST",
        body: payload,
      });
      renderDashboard(data.dashboard || {});
      if (data && data.trade) {
        const notification = buildTradeCreatedNotification(data.trade);
        notifyTradeBrainLifecycle(
          notification.title,
          notification.body,
          notification.notificationTag,
        );
      }
      setFeedback("Trade opened and linked to the live monitor.", "good");
    } catch (error) {
      setFeedback(error.message || "Could not open the trade.", "bad");
    }
  }

  async function handleCloseTrade() {
    const activeTrade = currentActiveTrade();
    if (!activeTrade) {
      setFeedback("There is no active trade to close.", "warn");
      return;
    }
    const exitPrice =
      asNumber(currentDashboard().marketData?.price, null) ??
      asNumber(activeTrade.live?.currentPrice, null);
    if (exitPrice === null) {
      setFeedback("No live price is available for a manual close.", "bad");
      return;
    }
    try {
      const data = await fetchJson(
        `/api/trades/${encodeURIComponent(activeTrade.id)}/close`,
        {
          method: "POST",
          body: {
            exitPrice,
            reason: "MANUAL_EXIT",
            reasoning: "Closed from the integrated dashboard.",
            emotion: "calm",
          },
        },
      );
      renderDashboard(data.dashboard || {});
      if (data && data.trade) {
        const notification = buildTradeClosedNotification(data.trade);
        notifyTradeBrainLifecycle(
          notification.title,
          notification.body,
          notification.notificationTag,
        );
      }
      setFeedback("Active trade closed.", "good");
    } catch (error) {
      setFeedback(error.message || "Could not close the active trade.", "bad");
    }
  }

  async function handleSnapshot() {
    const activeTrade = currentActiveTrade();
    if (!activeTrade) {
      setFeedback("There is no active trade to snapshot.", "warn");
      return;
    }
    try {
      await fetchJson(
        `/api/trades/${encodeURIComponent(activeTrade.id)}/snapshot`,
        {
          method: "POST",
          body:
            currentDashboard().marketData ||
            deriveMarketDataFromPrediction(state.latestPrediction) ||
            {},
        },
      );
      setFeedback("Current market snapshot saved.", "good");
    } catch (error) {
      setFeedback(error.message || "Could not save the trade snapshot.", "bad");
    }
  }

  function handleEmotionTag(emotion) {
    const activeTrade = currentActiveTrade();
    if (!activeTrade) {
      setFeedback("Open a trade before tagging emotion.", "warn");
      return;
    }
    if (!state.socket) {
      setFeedback("Emotion tagging needs the live socket connection.", "bad");
      return;
    }
    state.socket.emit("emotion:tag", {
      tradeId: activeTrade.id,
      emotion,
      note: safeText(
        byId("tb-emotion-note")?.value,
        buildAutoEmotionNote(emotion),
      ),
      price: asNumber(
        currentDashboard().marketData?.price,
        activeTrade.live?.currentPrice,
      ),
      unrealizedR: asNumber(activeTrade.live?.unrealizedR, 0),
      userId: getUserId(),
    });
    setFeedback(`Emotion tagged: ${emotion}.`, "warn");
  }

  function bindControls() {
    const form = byId("tb-create-form");
    if (form && form.dataset.bound !== "1") {
      form.dataset.bound = "1";
      form.addEventListener("submit", handleCreateTrade);
    }
    const closeButton = byId("tb-close-btn");
    if (closeButton && closeButton.dataset.bound !== "1") {
      closeButton.dataset.bound = "1";
      closeButton.addEventListener("click", handleCloseTrade);
    }
    const snapshotButton = byId("tb-snapshot-btn");
    if (snapshotButton && snapshotButton.dataset.bound !== "1") {
      snapshotButton.dataset.bound = "1";
      snapshotButton.addEventListener("click", handleSnapshot);
    }
    const directionNode = byId("tb-direction");
    if (directionNode && directionNode.dataset.bound !== "1") {
      directionNode.dataset.bound = "1";
      directionNode.addEventListener("change", () => {
        maybePrefillForm(
          currentDashboard().marketData ||
            deriveMarketDataFromPrediction(state.latestPrediction),
        );
        updateNoteHints();
      });
    }
    const autoTrackNode = byId("tb-auto-track");
    if (autoTrackNode && autoTrackNode.dataset.bound !== "1") {
      autoTrackNode.dataset.bound = "1";
      autoTrackNode.addEventListener("change", () => {
        setAutoTrackEnabled(autoTrackNode.checked);
        maybeAutoTrackPrediction(state.latestPrediction);
      });
    }
    const triggerNode = byId("tb-trigger");
    if (triggerNode && triggerNode.dataset.bound !== "1") {
      triggerNode.dataset.bound = "1";
      triggerNode.addEventListener("input", () => updateNoteHints());
    }
    const contextNoteNode = byId("tb-context-note");
    if (contextNoteNode && contextNoteNode.dataset.bound !== "1") {
      contextNoteNode.dataset.bound = "1";
      contextNoteNode.addEventListener("input", () => updateNoteHints());
    }
    const emotionNoteNode = byId("tb-emotion-note");
    if (emotionNoteNode && emotionNoteNode.dataset.bound !== "1") {
      emotionNoteNode.dataset.bound = "1";
      emotionNoteNode.addEventListener("input", () => updateNoteHints());
    }
    const emotionActions = byId("tb-emotion-actions");
    if (emotionActions && emotionActions.dataset.bound !== "1") {
      emotionActions.dataset.bound = "1";
      emotionActions.addEventListener("click", (event) => {
        const button = event.target.closest("button[data-emotion]");
        if (!button) {
          return;
        }
        updateNoteHints(button.dataset.emotion);
        handleEmotionTag(button.dataset.emotion);
      });
    }
  }

  function attachSocket(socket) {
    if (!socket) {
      return;
    }
    state.socket = socket;
    if (socket.__tradeBrainBound) {
      return;
    }
    socket.__tradeBrainBound = true;
    socket.on("trade:created", (payload) => {
      if (!payloadMatchesCurrentUser(payload)) {
        return;
      }
      if (payload && payload.dashboard) {
        renderDashboard(payload.dashboard);
      }
      if (payload && payload.trade) {
        const notification = buildTradeCreatedNotification(payload.trade);
        notifyTradeBrainLifecycle(
          notification.title,
          notification.body,
          notification.notificationTag,
        );
      }
      setFeedback("Trade opened and synced to the live feed.", "good");
    });
    socket.on("trade:updated", (payload) => {
      if (!payloadMatchesCurrentUser(payload)) {
        return;
      }
      if (payload && payload.dashboard) {
        renderDashboard(payload.dashboard);
      }
    });
    socket.on("trade:closed", (payload) => {
      if (!payloadMatchesCurrentUser(payload)) {
        return;
      }
      if (payload && payload.dashboard) {
        renderDashboard(payload.dashboard);
      }
      if (payload && payload.trade) {
        const notification = buildTradeClosedNotification(payload.trade);
        notifyTradeBrainLifecycle(
          notification.title,
          notification.body,
          notification.notificationTag,
        );
      }
      setFeedback("Trade closed and reviewed.", "good");
    });
    socket.on("trade:alert", (payload) => {
      if (!payloadMatchesCurrentUser(payload)) {
        return;
      }
      if (payload && payload.message) {
        setFeedback(payload.message, "warn");
      }
      refreshDashboard();
    });
    socket.on("emotion:updated", (payload) => {
      if (!payloadMatchesCurrentUser(payload)) {
        return;
      }
      if (payload && payload.message) {
        setFeedback(payload.message, "good");
      }
      refreshDashboard();
    });
    socket.on("stats:update", (payload) => {
      if (!payloadMatchesCurrentUser(payload)) {
        return;
      }
      if (payload && payload.dashboard) {
        renderDashboard(payload.dashboard);
      }
    });
    socket.on("trade:error", (payload) => {
      setFeedback(
        payload && payload.message
          ? payload.message
          : "Trade Brain socket error.",
        "bad",
      );
    });
  }

  function syncFromPrediction(prediction) {
    state.latestPrediction = prediction || null;
    const embeddedDashboard =
      prediction &&
      prediction.TradeBrain &&
      typeof prediction.TradeBrain === "object"
        ? prediction.TradeBrain
        : null;
    const existingHasTrackedData = dashboardHasTrackedData(currentDashboard());
    const embeddedHasTrackedData = dashboardHasTrackedData(embeddedDashboard);

    if (
      embeddedDashboard &&
      (!existingHasTrackedData || embeddedHasTrackedData)
    ) {
      renderDashboard(embeddedDashboard);
      maybeAutoTrackPrediction(prediction);
      maybeEvaluateActiveTrade(prediction);
      return;
    }
    maybePrefillForm(deriveMarketDataFromPrediction(prediction));
    updateAutoTrackUi(prediction);
    maybeAutoTrackPrediction(prediction);
    maybeEvaluateActiveTrade(prediction);
  }

  function init() {
    if (state.initialized) {
      return;
    }
    state.initialized = true;
    getUserId();
    state.autoTrackEnabled = loadAutoTrackEnabled();
    bindControls();
    updateTradeBrainTogglePreview();
    updateAutoTrackUi();
    refreshDashboard();
  }

  window.TradeBrainUI = {
    init,
    attachSocket,
    refreshDashboard,
    syncFromPrediction,
    getUserId,
    __test: {
      derivePredictionDirection,
      predictionStopLoss,
      isPredictionQualifiedForAutoTrack,
      resolvePredictionTradePlan,
    },
  };
})();
