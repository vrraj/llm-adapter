async function fetchModels(mergeCustomRegistry = false) {
  const url = mergeCustomRegistry ? "/api/models?merge_custom_registry=true" : "/api/models";
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error("Failed to fetch models: " + res.status);
  }
  return res.json();
}

function setErrorBanner(message, details) {
  const banner = document.getElementById("error_banner");
  if (!message) {
    banner.hidden = true;
    banner.textContent = "";
    return;
  }
  banner.hidden = false;
  banner.textContent = message + (details ? "\n" + details : "");
}

function appendLog(entry) {
  const log = document.getElementById("log_output");
  const now = new Date().toISOString();
  log.textContent += `[${now}] ${entry}\n`;
  log.scrollTop = log.scrollHeight;
}

async function init() {
  const modelKeySelect = document.getElementById("model_key_select");
  const sendBtn = document.getElementById("send_btn");
  const embedSendBtn = document.getElementById("embed_send_btn");
  const promptInput = document.getElementById("prompt_input");
  const reasoningOutput = document.getElementById("reasoning_output");
  const answerOutput = document.getElementById("answer_output");

  const embedSection = document.getElementById("embed_section");
  const embedInput = document.getElementById("embed_input");
  const normalizeEmbeddingInput = document.getElementById("normalize_embedding_input");
  const outputDimensionInput = document.getElementById("output_dimension_input");
  const embedResponseSection = document.getElementById("embed_response_section");
  const embeddingOutput = document.getElementById("embedding_output");

  const chatPromptSection = document.getElementById("chat_prompt_section");
  const chatResponseSection = document.getElementById("chat_response_section");

  const modelParamsOutput = document.getElementById("model_params_output");

  const rawApiOutput = document.getElementById("raw_api_output");
  const normalizedOutput = document.getElementById("normalized_output");

  const responsesRequestOutput = document.getElementById("responses_request_output");
  const formatRequestOutput = document.getElementById("format_request_output");
  const responsesRequestOutputEmbed = document.getElementById("responses_request_output_embed");
  const formatRequestOutputEmbed = document.getElementById("format_request_output_embed");

  const temperatureInput = document.getElementById("temperature_input");
  const topPInput = document.getElementById("top_p_input");
  const reasoningEffortSelect = document.getElementById("reasoning_effort_select");
  const maxTokensInput = document.getElementById("max_tokens_input");
  const processingStatus = document.getElementById("processing_status");
  const mergeCustomRegistryInput = document.getElementById("merge_custom_registry_input");

  let modelsByKey = {};

  function setProcessing(isProcessing, statusText = null) {
    if (!processingStatus) return;
    if (isProcessing) {
      processingStatus.textContent = "Processing...";
      processingStatus.style.color = "#059669"; // green
      processingStatus.style.fontWeight = "700";
    } else {
      processingStatus.textContent = statusText || "";
      processingStatus.style.color = statusText ? "#6b7280" : ""; // gray for status
      processingStatus.style.fontWeight = statusText ? "700" : "";
    }
  }

  function setSendingDisabled(disabled, embedMode) {
    try {
      // Disable both to prevent double submits and avoid relying on UI mode detection.
      sendBtn.disabled = !!disabled;
      embedSendBtn.disabled = !!disabled;
    } catch (e) {
      // ignore
    }
  }

  function getSelectedModelInfo() {
    const key = modelKeySelect.value;
    return modelsByKey[key] || null;
  }

  function isEmbeddingModel(mi) {
    if (!mi) return false;
    const ep = String(mi.endpoint || "").toLowerCase();
    return ep === "embeddings" || ep === "embed_content";
  }

  function splitThoughts(text) {
    const s = String(text || "");
    const open = s.indexOf("<thought>");
    const close = s.indexOf("</thought>");
    if (open === -1 || close === -1 || close <= open) {
      return { reasoning: "", answer: s };
    }
    const reasoning = s.slice(open + "<thought>".length, close).trim();
    const answer = (s.slice(close + "</thought>".length) || "").trim();
    return { reasoning, answer };
  }

  function syncUiForSelectedModel() {
    const mi = getSelectedModelInfo();
    const embedMode = isEmbeddingModel(mi);

    if (embedSection) embedSection.hidden = !embedMode;
    if (embedResponseSection) embedResponseSection.hidden = !embedMode;

    if (chatPromptSection) chatPromptSection.hidden = embedMode;
    if (chatResponseSection) chatResponseSection.hidden = embedMode;

    if (temperatureInput) temperatureInput.disabled = embedMode;
    if (topPInput) topPInput.disabled = embedMode;
    if (reasoningEffortSelect) reasoningEffortSelect.disabled = embedMode;
    if (maxTokensInput) maxTokensInput.disabled = embedMode;

    // Clear outputs when switching mode
    if (reasoningOutput) reasoningOutput.textContent = "";
    if (answerOutput) answerOutput.textContent = "";
    if (embeddingOutput) embeddingOutput.textContent = "";
    if (rawApiOutput) rawApiOutput.textContent = "";
    if (normalizedOutput) normalizedOutput.textContent = "";
    if (responsesRequestOutput) responsesRequestOutput.textContent = "";
    if (formatRequestOutput) formatRequestOutput.textContent = "";
    if (responsesRequestOutputEmbed) responsesRequestOutputEmbed.textContent = "";
    if (formatRequestOutputEmbed) formatRequestOutputEmbed.textContent = "";

    syncModelParamsPanel();
  }

  function buildResolvedRequestPreview() {
    const mi = getSelectedModelInfo();
    const embedMode = isEmbeddingModel(mi);

    const temperatureRaw = temperatureInput.value.trim();
    const topPRaw = topPInput.value.trim();
    const maxTokensRaw = maxTokensInput.value.trim();
    const reasoningEffortRaw = reasoningEffortSelect.value;

    if (!embedMode) {
      const payload = {
        model_key: modelKeySelect.value,
        prompt: (promptInput.value || "").trim(),
      };
      if (temperatureRaw) payload.temperature = Number(temperatureRaw);
      if (topPRaw) payload.top_p = Number(topPRaw);
      if (maxTokensRaw) payload.max_output_tokens = Number(maxTokensRaw);
      if (reasoningEffortRaw) payload.reasoning_effort = reasoningEffortRaw;
      return { endpoint: "/api/chat", request_model: payload };
    }

    return {
      endpoint: "/api/embed",
      request_model: {
        model_key: modelKeySelect.value,
        text: (embedInput.value || "").trim(),
        normalize_embedding: !!normalizeEmbeddingInput.checked,
        output_dimensionality: outputDimensionInput.value ? Number(outputDimensionInput.value) : undefined,
      },
    };
  }

  function syncModelParamsPanel() {
    const mi = getSelectedModelInfo();
    const preview = buildResolvedRequestPreview();

    // Set output dimension placeholder to model default for embeddings
    if (outputDimensionInput && isEmbeddingModel(mi)) {
      const defaultDim = mi?.capabilities?.output_dimensionality || mi?.capabilities?.dimensions;
      outputDimensionInput.placeholder = defaultDim ? String(defaultDim) : "auto (from model)";
    }

    const out = {
      model_registry_entry: mi,
      request_preview: preview,
    };

    modelParamsOutput.textContent = JSON.stringify(out, null, 2);
  }

  async function loadModels() {
    try {
      const data = await fetchModels(mergeCustomRegistryInput.checked);
      const models = data.models || {};
      modelsByKey = models;
      
      // Clear existing options
      while (modelKeySelect.options.length > 0) {
        modelKeySelect.remove(0);
      }
      
      Object.values(models).forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m.key;
        const enabled = !!m.enabled;

        const modelName = m.model || "?";
        const provider = m.provider || "?";
        const endpoint = m.endpoint || "?";
        const label = `${m.key} → ${modelName} (${provider}, ${endpoint})`;
        opt.textContent = enabled ? label : `${label} [disabled]`;
        modelKeySelect.appendChild(opt);
      });

      // Default to a chat-capable enabled model so chat-only controls are not disabled on load.
      try {
        const firstChatEnabled = Object.values(models).find((m) => {
          if (!m || !m.key) return false;
          if (!m.enabled) return false;
          const ep = String(m.endpoint || "").toLowerCase();
          return ep !== "embeddings" && ep !== "embed_content";
        });
        if (firstChatEnabled && firstChatEnabled.key) {
          modelKeySelect.value = firstChatEnabled.key;
        }
      } catch (e) {
        // ignore
      }

      syncUiForSelectedModel();
      syncModelParamsPanel();
    } catch (e) {
      setErrorBanner("Failed to load providers/models", String(e));
      appendLog("Error loading models: " + String(e));
    }
  }

  await loadModels();

  async function send() {
    const model_key = modelKeySelect.value;
    const mi = getSelectedModelInfo();
    const embedMode = isEmbeddingModel(mi);

    setErrorBanner("");
    reasoningOutput.textContent = "";
    answerOutput.textContent = "";
    embeddingOutput.textContent = "";
    rawApiOutput.textContent = "";
    normalizedOutput.textContent = "";
    responsesRequestOutput.textContent = "";
    formatRequestOutput.textContent = "";
    responsesRequestOutputEmbed.textContent = "";
    formatRequestOutputEmbed.textContent = "";
    appendLog(`Request -> model_key=${model_key}`);
    setProcessing(true);
    setSendingDisabled(true, embedMode);

    const temperatureRaw = temperatureInput.value.trim();
    const topPRaw = topPInput.value.trim();
    const maxTokensRaw = maxTokensInput.value.trim();
    const reasoningEffortRaw = reasoningEffortSelect.value;

    let url = "/api/chat";
    let payload = { 
      model_key,
      merge_custom_registry: mergeCustomRegistryInput.checked
    };

    try {
      if (!embedMode) {
        const prompt = promptInput.value.trim();
        if (!prompt) {
          setProcessing(false);
          setSendingDisabled(false, embedMode);
          return;
        }
        payload.prompt = prompt;
        if (temperatureRaw) payload.temperature = Number(temperatureRaw);
        if (topPRaw) payload.top_p = Number(topPRaw);
        if (maxTokensRaw) payload.max_output_tokens = Number(maxTokensRaw);
        if (reasoningEffortRaw) payload.reasoning_effort = reasoningEffortRaw;
      } else {
        const text = (embedInput.value || "").trim();
        if (!text) {
          setProcessing(false);
          setSendingDisabled(false, embedMode);
          return;
        }
        url = "/api/embed";
        payload.text = text;
        payload.normalize_embedding = !!normalizeEmbeddingInput.checked;
        payload.merge_custom_registry = mergeCustomRegistryInput.checked;
      }
    } catch (e) {
      setErrorBanner("Request failed", String(e));
      appendLog("Request build exception: " + String(e));
      setProcessing(false, "Error");
      setSendingDisabled(false, embedMode);
      return;
    }

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const text = await res.text();
        const summary = `HTTP ${res.status} calling ${url}`;
        setErrorBanner(summary, text);
        appendLog(`${summary}: ${text}`);
        setProcessing(false, "Error");
        setSendingDisabled(false, embedMode);
        return;
      }

      const body = await res.json();

      const providerResp = body.provider_response;
      const normalizedResp = body.normalized_result;
      const providerReq = body.provider_request;
      const formatReq = body.format_request;

      if (!embedMode) {
        if (providerReq !== undefined) {
          responsesRequestOutput.textContent = JSON.stringify(providerReq, null, 2);
        }
        if (formatReq !== undefined) {
          formatRequestOutput.textContent = JSON.stringify(formatReq, null, 2);
        }
      } else {
        if (providerReq !== undefined) {
          responsesRequestOutputEmbed.textContent = JSON.stringify(providerReq, null, 2);
        }
        if (formatReq !== undefined) {
          formatRequestOutputEmbed.textContent = JSON.stringify(formatReq, null, 2);
        }
      }

      if (providerResp !== undefined) {
        rawApiOutput.textContent = JSON.stringify(providerResp, null, 2);
      }
      if (normalizedResp !== undefined) {
        normalizedOutput.textContent = JSON.stringify(normalizedResp, null, 2);
      }

      if (body.ok) {
        if (!embedMode) {
          const backendReasoning = body.reasoning_text || "";
          const backendAnswer = body.answer_text || "";
          if (!backendReasoning && backendAnswer.includes("<thought>") && backendAnswer.includes("</thought>")) {
            const parts = splitThoughts(backendAnswer);
            reasoningOutput.textContent = parts.reasoning || "";
            answerOutput.textContent = parts.answer || "";
          } else {
            reasoningOutput.textContent = backendReasoning;
            answerOutput.textContent = backendAnswer;
          }
          appendLog("Response ok. Usage: " + JSON.stringify(body.raw_usage || {}, null, 2));
          
          // Show status and finish reason from the response
          const status = normalizedResp?.status || body.status || "unknown";
          const finishReason = normalizedResp?.finish_reason || body.finish_reason;
          const statusText = finishReason ? `${status} (${finishReason})` : status;
          setProcessing(false, statusText);
          setSendingDisabled(false, embedMode);
        } else {
          embeddingOutput.textContent = JSON.stringify(body, null, 2);
          appendLog("Embedding ok. Usage: " + JSON.stringify(body.raw_usage || {}, null, 2));
          setProcessing(false, "Completed");
          setSendingDisabled(false, embedMode);
        }
      } else {
        const err = body.error || {};

        const providerStr = err.provider || "?";
        const modelStr = err.model || "?";
        const kindStr = err.kind || "unknown";
        const codeStr = err.code || "";
        const retryStr =
          typeof err.retry_after === "number" && !Number.isNaN(err.retry_after)
            ? `${err.retry_after}s`
            : null;

        const summaryParts = [
          `provider=${providerStr}`,
          `model=${modelStr}`,
          `kind=${kindStr}`,
        ];
        if (codeStr) summaryParts.push(`code=${codeStr}`);
        if (retryStr) summaryParts.push(`retry_after=${retryStr}`);

        const summary = "Error from backend: " + summaryParts.join(", ");
        const details = err.message || "";

        setErrorBanner(summary, details);
        appendLog("Error payload: " + JSON.stringify(err, null, 2));
        setProcessing(false, "Error");
        setSendingDisabled(false, embedMode);
      }

    } catch (e) {
      setErrorBanner("Request failed", String(e));
      appendLog("Request exception: " + String(e));
      setProcessing(false, "Error");
      setSendingDisabled(false, embedMode);
    }
  }

  sendBtn.addEventListener("click", send);
  embedSendBtn.addEventListener("click", send);
  modelKeySelect.addEventListener("change", syncUiForSelectedModel);
  modelKeySelect.addEventListener("change", syncModelParamsPanel);
  mergeCustomRegistryInput.addEventListener("change", loadModels);
  temperatureInput.addEventListener("input", syncModelParamsPanel);
  topPInput.addEventListener("input", syncModelParamsPanel);
  reasoningEffortSelect.addEventListener("change", syncModelParamsPanel);
  maxTokensInput.addEventListener("input", syncModelParamsPanel);
  promptInput.addEventListener("input", syncModelParamsPanel);
  promptInput.addEventListener("keydown", (evt) => {
    if (evt.key === "Enter" && (evt.metaKey || evt.ctrlKey)) {
      evt.preventDefault();
      send();
    }
  });

  normalizeEmbeddingInput.addEventListener("change", syncModelParamsPanel);
  outputDimensionInput.addEventListener("input", syncModelParamsPanel);
  embedInput.addEventListener("input", syncModelParamsPanel);
  embedInput.addEventListener("keydown", (evt) => {
    if (evt.key === "Enter" && (evt.metaKey || evt.ctrlKey)) {
      evt.preventDefault();
      send();
    }
  });
}

window.addEventListener("DOMContentLoaded", init);
