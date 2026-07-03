<template>
  <section className="system-panel semantic-panel">
    <PanelHeader
        code="SG"
        title="Semantic Generation"
        subtitle="Construct area and class descriptions under original and new area conditions."
    />

    <div className="semantic-stage">
      <AreaColumn
          title="Original Area"
          v-model:domain="originalDomain"
          :domain-options="domainOptions"
          type="source"
          :meta="sourceMeta"
          :cards="sourceCards"
      />
      <AreaColumn
          title="New Area"
          v-model:domain="newDomain"
          :domain-options="domainOptions"
          type="target"
          :meta="targetMeta"
          :cards="targetCards"
      />
    </div>
  </section>
</template>

<script setup>
import {Transition, defineComponent, h, onBeforeUnmount, onMounted, reactive, ref, watch} from "vue";
import {ElDrawer, ElMessage} from "element-plus";
import PanelHeader from "./PanelHeader.vue";
import {
  domainOptions,
  originalDomain,
  newDomain,
  roadClasses,
  soilClasses,
  sourceMeta,
  targetMeta,
  soilProfiles,
  sourceCards,
  targetCards,
} from "../shared/gprState";

const API_BASE = process.env.VUE_APP_API_BASE_URL || "http://127.0.0.1:8000/api";
const backendDomains = reactive({});

const numericPercent = (value, min, max) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 0;
  return Math.min(100, Math.max(0, ((Number(value) - min) / (max - min)) * 100));
};

const makeRangeMetric = ({name, minValue, maxValue, min, max, step, unit}) => ({
  name,
  value:
      minValue === null || minValue === undefined || maxValue === null || maxValue === undefined
          ? "Not configured"
          : Number(minValue) === Number(maxValue)
              ? `${minValue}${unit === "%" || unit === "" ? unit : ` ${unit}`}`
              : `${minValue}-${maxValue}${unit === "%" || unit === "" ? unit : ` ${unit}`}`,
  type: "range",
  start: Number(numericPercent(minValue, min, max).toFixed(1)),
  width: Number((numericPercent(maxValue, min, max) - numericPercent(minValue, min, max)).toFixed(1)),
  min,
  max,
  step,
  unit,
});

const makePointMetric = ({name, value, min, max, step, unit}) => ({
  name,
  value: value === null || value === undefined ? "Not configured" : `${value}${unit === "%" || unit === "" ? unit : ` ${unit}`}`,
  type: "point",
  point: Number(numericPercent(value, min, max).toFixed(1)),
  min,
  max,
  step,
  unit,
});

const applyRoadMeta = (meta, domain, side) => {
  meta.condition = side === "source" ? "Validated original area" : "Changed sensing condition";
  meta.roadSurface = domain.preview.roadSurface;
  meta.metrics.splice(
      0,
      meta.metrics.length,
      makeRangeMetric({
        name: "Frequency Range",
        minValue: domain.preview.frequencyRange.min,
        maxValue: domain.preview.frequencyRange.max,
        min: 0,
        max: Math.max(600, domain.preview.frequencyRange.max),
        step: 10,
        unit: "MHz",
      }),
      makePointMetric({
        name: "Time Window",
        value: domain.preview.timeWindow.value,
        min: 0,
        max: 100,
        step: 1,
        unit: "ns",
      })
  );
};

const applySoilProfile = (domain) => {
  const preview = domain.preview || {};
  const composition = preview.composition || {};
  const profile = {
    composition: [
      makePointMetric({name: "Sand", value: composition.sand, min: 0, max: 100, step: 1, unit: "%"}),
      makePointMetric({name: "Silt", value: composition.silt, min: 0, max: 100, step: 1, unit: "%"}),
      makePointMetric({name: "Clay", value: composition.clay, min: 0, max: 100, step: 1, unit: "%"}),
    ],
    water: makeRangeMetric({
      name: "Water Content",
      minValue: preview.waterContent?.min,
      maxValue: preview.waterContent?.max,
      min: 0,
      max: 40,
      step: 1,
      unit: "%",
    }),
    electrical: [
      makeRangeMetric({
        name: "Relative Permittivity",
        minValue: preview.relativePermittivity?.min,
        maxValue: preview.relativePermittivity?.max,
        min: 0,
        max: 40,
        step: 0.1,
        unit: "",
      }),
      makeRangeMetric({
        name: "Conductivity",
        minValue: preview.conductivity?.min,
        maxValue: preview.conductivity?.max,
        min: 0,
        max: 0.11,
        step: 0.001,
        unit: "S/m",
      }),
    ],
    peplinski: makePointMetric({
      name: "Peplinski Model Fractal Dimension",
      value: preview.peplinskiDimension,
      min: 1,
      max: 1.8,
      step: 0.01,
      unit: "",
    }),
  };
  soilProfiles[domain.code] = profile;
  soilProfiles[domain.name] = profile;
};
const applyDomainToSide = (code, side) => {
  const domain = backendDomains[code];
  if (!domain) return;
  if (domain.renderMode === "soil") {
    applySoilProfile(domain);
    return;
  }
  applyRoadMeta(side === "source" ? sourceMeta : targetMeta, domain, side);
};

const loadDomains = async (useDefaults = true) => {
  try {
    const response = await fetch(`${API_BASE}/domains/`);
    if (!response.ok) throw new Error("Domain request fails.");
    const payload = await response.json();

    Object.keys(backendDomains).forEach((key) => delete backendDomains[key]);
    Object.keys(soilProfiles).forEach((key) => delete soilProfiles[key]);
    payload.domains.forEach((domain) => {
      backendDomains[domain.code] = domain;
      backendDomains[domain.name] = domain;
    });

    domainOptions.splice(
        0,
        domainOptions.length,
        ...payload.domains.map((domain) => ({
          code: domain.code,
          name: domain.name,
          renderMode: domain.renderMode,
        }))
    );

    if (useDefaults) {
      originalDomain.value = payload.defaults?.originalName || payload.defaults?.originalCode || originalDomain.value;
      newDomain.value = payload.defaults?.newName || payload.defaults?.newCode || newDomain.value;
    }
    applyDomainToSide(originalDomain.value, "source");
    applyDomainToSide(newDomain.value, "target");
  } catch {
    ElMessage({
      showClose: true,
      center: true,
      type: "error",
      message: "Area domain loading fails.",
      offset: 24,
      duration: 2600,
    });
  }
};

onMounted(loadDomains);
watch(originalDomain, (code) => applyDomainToSide(code, "source"));
watch(newDomain, (code) => applyDomainToSide(code, "target"));

const AreaColumn = defineComponent({
  props: {
    title: String,
    domain: String,
    domainOptions: Array,
    type: String,
    meta: Object,
    cards: Array,
  },
  emits: ["update:domain"],
  setup(props, {emit}) {
    const notifyMessage = (message, type = "success") => {
      ElMessage({
        showClose: true,
        center: true,
        type,
        message,
        offset: 24,
        duration: 2600,
        appendTo: document.body,
      });
    };
    const activeRangeDrag = ref(null);
    const suppressAreaClick = ref(false);
    const expandedCards = reactive({});
    const areaInputRows = ref([]);
    const areaImportInput = ref(null);
    const areaInputDialog = reactive({
      visible: false,
      form: {},
    });
    const generation = reactive({
      active: false,
      progress: 0,
      status: "idle",
      phase: "Preparing semantic context...",
      runId: null,
    });
    const generationTimer = ref(null);
    const generationController = ref(null);
    const semanticLlmOptions = ["GPT-4o", "GPT-3.5-turbo", "GPT-4o-mini", "Gemini-2.5", "LLaMA-3.1", "Qwen-2.5"];
    const semanticLlm = ref("GPT-4o");
    const useExpertKnowledge = ref(false);
    const useImageAssist = ref(false);
    const generatedCards = ref([]);
    const semanticLoading = ref(false);
    const showAreaDetail = ref(false);
    const selectedClassCard = ref(null);
    const classDetailConfig = reactive({
      llmName: "GPT-4o",
      useExpertKnowledge: false,
      useImageAssist: false,
      loading: false,
    });
    const expertAnnotation = reactive({
      visible: false,
      x: 0,
      y: 0,
      text: "",
      noteOpen: false,
      note: "",
      noteMarkId: "",
      viewName: "",
      viewText: "",
      effect: "",
      updateRevise: "",
      annotatedAt: "",
    });
    const detailEditMode = ref(false);
    const detailEdits = reactive({});
    const expertMarkMenu = reactive({
      visible: false,
      x: 0,
      y: 0,
      markId: "",
    });
    const expertMarks = ref([]);
    const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
    const roundToStep = (value, step) => Math.round(value / step) * step;
    const optionValue = (option) => (typeof option === "string" ? option : option.name);
    const optionLabel = (option) => (typeof option === "string" ? option : option.name);
    const activeDomainRecord = () => backendDomains[props.domain] || Object.values(backendDomains).find((item) => item?.code === props.domain || item?.name === props.domain);
    const activeDomainName = () => {
      const record = activeDomainRecord();
      if (record?.name) return record.name;
      const option = props.domainOptions.find((item) => optionValue(item) === props.domain);
      return option ? optionLabel(option) : props.domain;
    };

    const percentFromEvent = (event, rect) =>
        clamp(((event.clientX - rect.left) / rect.width) * 100, 0, 100);

    const updateMetricValue = (metric) => {
      const unitText = metric.unit === "%" || metric.unit === "" ? metric.unit : ` ${metric.unit}`;

      if (metric.type === "range") {
        const end = metric.start + metric.width;
        const minValue = metric.min + (metric.start / 100) * (metric.max - metric.min);
        const maxValue = metric.min + (end / 100) * (metric.max - metric.min);
        metric.value = `${roundToStep(minValue, metric.step)}-${roundToStep(maxValue, metric.step)}${unitText}`;
        return;
      }

      const value = metric.min + (metric.point / 100) * (metric.max - metric.min);
      metric.value = `${roundToStep(value, metric.step)}${unitText}`;
    };

    const handleMetricClick = (event, metric) => {
      event.stopPropagation();
      if (metric.type !== "point") return;
      const rect = event.currentTarget.getBoundingClientRect();
      const percent = percentFromEvent(event, rect);

      metric.point = Number(percent.toFixed(1));
      updateMetricValue(metric);
    };

    const updateRangeEdge = (event) => {
      const drag = activeRangeDrag.value;
      if (!drag) return;

      const percent = percentFromEvent(event, drag.rect);
      const minWidth = 8;

      if (drag.edge === "start") {
        const end = drag.metric.start + drag.metric.width;
        const nextStart = clamp(percent, 0, end - minWidth);
        drag.metric.start = Number(nextStart.toFixed(1));
        drag.metric.width = Number((end - nextStart).toFixed(1));
      } else {
        const nextEnd = clamp(percent, drag.metric.start + minWidth, 100);
        drag.metric.width = Number((nextEnd - drag.metric.start).toFixed(1));
      }

      updateMetricValue(drag.metric);
    };

    const stopRangeDrag = () => {
      activeRangeDrag.value = null;
      document.body.classList.remove("is-range-dragging");
      window.removeEventListener("pointermove", updateRangeEdge);
      window.removeEventListener("pointerup", stopRangeDrag);
      window.setTimeout(() => {
        suppressAreaClick.value = false;
      }, 180);
    };

    const startRangeDrag = (event, metric, edge) => {
      event.preventDefault();
      event.stopPropagation();
      suppressAreaClick.value = true;
      document.body.classList.add("is-range-dragging");
      activeRangeDrag.value = {
        metric,
        edge,
        rect: event.currentTarget.closest(".metric-row").getBoundingClientRect(),
      };
      window.addEventListener("pointermove", updateRangeEdge);
      window.addEventListener("pointerup", stopRangeDrag);
    };

    onBeforeUnmount(() => {
      stopRangeDrag();
      if (generationTimer.value) window.clearInterval(generationTimer.value);
      if (generationController.value) generationController.value.abort();
    });

    const renderMetricControl = (metric) => {
      if (metric.type === "range") {
        return [
          h("i", {
            style: {
              left: `${metric.start}%`,
              width: `${metric.width}%`,
            },
          }),
          h("button", {
            class: "range-handle start",
            type: "button",
            style: {left: `${metric.start}%`},
            onPointerdown: (event) => startRangeDrag(event, metric, "start"),
          }),
          h("button", {
            class: "range-handle end",
            type: "button",
            style: {left: `${metric.start + metric.width}%`},
            onPointerdown: (event) => startRangeDrag(event, metric, "end"),
          }),
        ];
      }

      return h("i", {style: {left: `${metric.point}%`}});
    };

    const isSoilDomain = () => activeDomainRecord()?.renderMode === "soil" || props.domain?.startsWith("a");
    const blankSoilProfile = () => ({
      composition: [
        makePointMetric({name: "Sand", value: null, min: 0, max: 100, step: 1, unit: "%"}),
        makePointMetric({name: "Silt", value: null, min: 0, max: 100, step: 1, unit: "%"}),
        makePointMetric({name: "Clay", value: null, min: 0, max: 100, step: 1, unit: "%"}),
      ],
      water: makeRangeMetric({
        name: "Water Content",
        minValue: null,
        maxValue: null,
        min: 0,
        max: 40,
        step: 1,
        unit: "%"
      }),
      electrical: [
        makeRangeMetric({
          name: "Relative Permittivity",
          minValue: null,
          maxValue: null,
          min: 0,
          max: 40,
          step: 0.1,
          unit: ""
        }),
        makeRangeMetric({
          name: "Conductivity",
          minValue: null,
          maxValue: null,
          min: 0,
          max: 0.11,
          step: 0.001,
          unit: "S/m"
        }),
      ],
      peplinski: makePointMetric({
        name: "Peplinski Model Fractal Dimension",
        value: null,
        min: 1,
        max: 1.8,
        step: 0.01,
        unit: ""
      }),
    });
    const activeSoilProfile = () =>
        soilProfiles[props.domain] ||
        soilProfiles[activeDomainRecord()?.name] ||
        soilProfiles[activeDomainRecord()?.code] ||
        blankSoilProfile();
    const activeClassNames = () => {
      const record = activeDomainRecord();
      const fromRecord = record?.categories || record?.classes || record?.semanticCategories || [];
      const names = Array.isArray(fromRecord)
          ? fromRecord.map((item) => (typeof item === "string" ? item : item?.name)).filter(Boolean)
          : [];
      if (names.length) return names;
      return isSoilDomain() ? soilClasses : roadClasses;
    };

    const isSameClassName = (left, right) => {
      const normalize = (value) => String(value || "").toLowerCase().replace(/\s+/g, " ").trim();
      const a = normalize(left);
      const b = normalize(right);
      return a === b || (a === "pipeline" && b === "metal pipeline") || (a === "metal pipeline" && b === "pipeline");
    };

    const activeCards = () => {
      const names = activeClassNames();
      return generatedCards.value.filter((card) => names.some((name) => isSameClassName(name, card.cls)));
    };

    const includedClassesText = () => {
      const classes = isSoilDomain() ? soilClasses : roadClasses;
      return `This area includes ${classes.slice(0, -1).join(", ")}, and ${classes.at(-1)}.`;
    };
    const cardKey = (card) => `${props.type}-${props.domain}-${card.cls}`;
    const toggleCard = (card) => {
      const key = cardKey(card);
      expandedCards[key] = !expandedCards[key];
    };
    const hasExpandedCard = () => activeCards().some((card) => expandedCards[cardKey(card)]);
    const semanticBriefItems = (card) => {
      if (Array.isArray(card.details) && card.details.length) {
        return card.details.map((item) => [
          item.key || item.name || "Semantic View",
          item.briefDescription || item.brief || item.text || item.description || "",
        ]);
      }

      const isPipe = card.cls === "Pipeline" || card.cls === "Metal Pipeline";
      const shape = isPipe
          ? "Hyperbolic trajectory with compact apex and bilateral spreading tails."
          : card.cls === "Crack"
              ? "Thin discontinuous response aligned with local layer interruption."
              : card.cls === "Cavity"
                  ? "Closed or semi-closed reflection boundary around a void-like region."
                  : card.cls === "Normal"
                      ? "Layered background with no dominant abnormal geometry."
                      : "Diffuse irregular region without stable object boundary.";
      const interaction = isSoilDomain()
          ? "Controlled by soil mixture, water content, permittivity, and conductivity."
          : "Controlled by road surface, sensing condition, frequency range, and time window.";

      return [
        ["Dominant Shape", shape],
        ["Interaction Type", interaction],
        ["Reflection Strength", `${card.score >= 88 ? "Strong" : card.score >= 78 ? "Moderate" : "Weak"} response with ${card.score}% semantic confidence.`],
        ["Feature Footprint", isPipe ? "Compact apex with laterally expanding footprint." : "Localized footprint around the suspected abnormal region."],
        ["Signal Coherence", card.cls === "Normal" ? "High continuity across adjacent traces." : "Coherence is evaluated by local trace consistency and boundary persistence."],
        ["Anomaly Source", card.cls === "Normal" ? "Background area without target anomaly." : `${card.cls} response under the selected domain context.`],
        ["Pattern Uniformity", card.cls === "Loose" ? "Low uniformity with disrupted compactness pattern." : "Uniformity is checked against neighboring background responses."],
        ["Feature Expression", card.text],
        ["Layer Interaction", card.cls === "Crack" ? "May interrupt layer continuity and create weak scattered edges." : "May overlap with layer-interface reflections depending on domain parameters."],
        ["Signal Complexity", isSoilDomain() ? "Complexity rises with water content and conductive attenuation." : "Complexity rises with clutter, road surface variation, and acquisition window."],
      ];
    };
    const semanticDetailItems = (card) => {
      if (Array.isArray(card.details) && card.details.length) {
        return card.details.map((item) => [
          item.key || item.name || "Semantic View",
          item.detailedDescription || item.detailDescription || item.description || item.briefDescription || item.text || "",
        ]);
      }

      const isPipe = card.cls === "Pipeline" || card.cls === "Metal Pipeline";
      const shape = isPipe
          ? "Hyperbolic trajectory with compact apex and bilateral spreading tails."
          : card.cls === "Crack"
              ? "Thin discontinuous response aligned with local layer interruption."
              : card.cls === "Cavity"
                  ? "Closed or semi-closed reflection boundary around a void-like region."
                  : card.cls === "Normal"
                      ? "Layered background with no dominant abnormal geometry."
                      : "Diffuse irregular region without stable object boundary.";
      const interaction = isSoilDomain()
          ? "Controlled by soil mixture, water content, permittivity, and conductivity."
          : "Controlled by road surface, sensing condition, frequency range, and time window.";

      return [
        ["Dominant Shape", shape],
        ["Interaction Type", interaction],
        ["Reflection Strength", `${card.score >= 88 ? "Strong" : card.score >= 78 ? "Moderate" : "Weak"} response with ${card.score}% semantic confidence.`],
        ["Feature Footprint", isPipe ? "Compact apex with laterally expanding footprint." : "Localized footprint around the suspected abnormal region."],
        ["Signal Coherence", card.cls === "Normal" ? "High continuity across adjacent traces." : "Coherence is evaluated by local trace consistency and boundary persistence."],
        ["Anomaly Source", card.cls === "Normal" ? "Background area without target anomaly." : `${card.cls} response under the selected domain context.`],
        ["Pattern Uniformity", card.cls === "Loose" ? "Low uniformity with disrupted compactness pattern." : "Uniformity is checked against neighboring background responses."],
        ["Feature Expression", card.text],
        ["Layer Interaction", card.cls === "Crack" ? "May interrupt layer continuity and create weak scattered edges." : "May overlap with layer-interface reflections depending on domain parameters."],
        ["Signal Complexity", isSoilDomain() ? "Complexity rises with water content and conductive attenuation." : "Complexity rises with clutter, road surface variation, and acquisition window."],
      ];
    };
    const primaryBriefItem = (card) => {
      const items = semanticBriefItems(card);
      return items.find((item) => item[0] === card.view) || items[0] || ["Dominant Shape", card.text || ""];
    };

    const stablePickBriefItems = (items, seed, count) => {
      const pool = [...items];
      let hash = Array.from(String(seed)).reduce((sum, char) => sum + char.charCodeAt(0), 0);
      const picked = [];
      while (pool.length && picked.length < count) {
        hash = (hash * 31 + 17) % 9973;
        const index = hash % pool.length;
        picked.push(pool.splice(index, 1)[0]);
      }
      return picked;
    };

    const formatSemanticTime = (value) => {
      if (!value) return "-";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return String(value);
      return date.toLocaleString();
    };
    const closeClassDetail = (showNotice = true) => {
      selectedClassCard.value = null;
      expertAnnotation.visible = false;
      expertAnnotation.noteOpen = false;
      expertMarkMenu.visible = false;
      detailEditMode.value = false;
      Object.keys(detailEdits).forEach((key) => delete detailEdits[key]);
      // if (showNotice) notifyMessage("Class detail is closed.", "info");
    };
    const openClassDetail = (card) => {
      selectedClassCard.value = card;
      classDetailConfig.llmName = card.llmName || semanticLlm.value;
      classDetailConfig.useExpertKnowledge = Boolean(card.useExpertKnowledge);
      classDetailConfig.useImageAssist = Boolean(card.useImageAssist);
      loadExpertMarks(card);
    };

    const updateClassDetailFromDatabase = async (patch = {}) => {
      if (!selectedClassCard.value) return;
      const nextConfig = {
        llmName: patch.llmName ?? classDetailConfig.llmName,
        useExpertKnowledge: patch.useExpertKnowledge ?? classDetailConfig.useExpertKnowledge,
        useImageAssist: patch.useImageAssist ?? classDetailConfig.useImageAssist,
      };
      classDetailConfig.llmName = nextConfig.llmName;
      classDetailConfig.useExpertKnowledge = nextConfig.useExpertKnowledge;
      classDetailConfig.useImageAssist = nextConfig.useImageAssist;
      classDetailConfig.loading = true;
      try {
        const response = await fetch(
            `${API_BASE}/semantic-generation/latest/?domain=${encodeURIComponent(props.domain)}&llmName=${encodeURIComponent(nextConfig.llmName)}&useExpertKnowledge=${nextConfig.useExpertKnowledge}&useImageAssist=${nextConfig.useImageAssist}`
        );
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.message || "Class detail loading fails.");
        const cards = payload.run?.descriptions ? normalizeGeneratedCards(payload.run.descriptions, payload.run) : [];
        const matched = cards.find((item) => item.cls === selectedClassCard.value.cls);
        if (matched) {
          selectedClassCard.value = matched;
          await loadExpertMarks(matched);
        }
      } catch {
        notifyMessage(`${selectedClassCard.value.cls} detail loading fails.`, "error");
      } finally {
        classDetailConfig.loading = false;
      }
    };

    const normalizeMark = (mark) => ({
      id: mark.id || mark.markId || `local-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      viewName: mark.viewName || mark.view || "",
      viewText: mark.viewText || "",
      text: mark.annotatedText || mark.text || "",
      effect: mark.effect || mark.type || "correct",
      note: mark.annotationContent || mark.note || "",
      updateRevise: mark.updateRevise || mark.update_revise || "",
      annotatedAt: mark.annotatedAt || new Date().toISOString(),
    });

    const loadExpertMarks = async (card = selectedClassCard.value) => {
      expertMarks.value = [];
      if (!card?.descId) return;
      try {
        const response = await fetch(`${API_BASE}/semantic-annotations/?descId=${encodeURIComponent(card.descId)}`);
        const payload = await response.json().catch(() => ({}));
        if (response.ok) expertMarks.value = (payload.annotations || []).map(normalizeMark);
      } catch {
        expertMarks.value = [];
      }
    };

    const handleAnnotationMouseUp = (event, viewName, viewText) => {
      const selection = window.getSelection();
      const selectedText = selection ? selection.toString().trim() : "";
      if (!selectedText || !event.currentTarget.contains(selection.anchorNode) || !event.currentTarget.contains(selection.focusNode)) {
        expertAnnotation.visible = false;
        return;
      }
      expertAnnotation.visible = true;
      expertAnnotation.x = event.clientX + 12;
      expertAnnotation.y = event.clientY - 8;
      expertAnnotation.text = selectedText;
      expertAnnotation.viewName = viewName;
      expertAnnotation.viewText = viewText;
      expertAnnotation.effect = "";
      expertAnnotation.updateRevise = "";
      expertMarkMenu.visible = false;
    };

    const addExpertMark = (effect) => {
      if (!expertAnnotation.text || !expertAnnotation.viewName) return;
      expertMarks.value.push(normalizeMark({
        viewName: expertAnnotation.viewName,
        viewText: expertAnnotation.viewText,
        text: expertAnnotation.text,
        effect,
      }));
      expertAnnotation.visible = false;
      const selection = window.getSelection();
      if (selection) selection.removeAllRanges();
    };

    const openExpertNote = (markId = expertMarkMenu.markId) => {
      const mark = expertMarks.value.find((item) => item.id === markId);
      if (!mark) return;
      expertAnnotation.noteOpen = true;
      expertAnnotation.noteMarkId = mark.id;
      expertAnnotation.note = mark.note || "";
      expertAnnotation.text = mark.text;
      expertAnnotation.viewName = mark.viewName;
      expertAnnotation.viewText = mark.viewText;
      expertAnnotation.effect = mark.effect;
      expertAnnotation.updateRevise = mark.updateRevise || "";
      expertAnnotation.annotatedAt = mark.annotatedAt;
      expertMarkMenu.visible = false;
    };

    const saveExpertNote = () => {
      const mark = expertMarks.value.find((item) => item.id === expertAnnotation.noteMarkId);
      if (!mark) {
        notifyMessage("Expert annotation note save fails: annotation is missing.", "error");
        return;
      }
      mark.note = expertAnnotation.note;
      mark.updateRevise = expertAnnotation.updateRevise;
      mark.annotatedAt = new Date().toISOString();
      expertAnnotation.annotatedAt = mark.annotatedAt;
      expertAnnotation.noteOpen = false;
      notifyMessage("Expert annotation note saves successfully.", "success");
    };

    const deleteExpertMark = (markId = expertMarkMenu.markId) => {
      expertMarks.value = expertMarks.value.filter((item) => item.id !== markId);
      expertMarkMenu.visible = false;
    };

    const closeExpertDrawer = () => {
      expertAnnotation.noteOpen = false;
    };

    const saveClassDetail = async () => {
      if (!selectedClassCard.value?.descId) {
        notifyMessage("Semantic annotation save fails: description id is missing.", "error");
        return;
      }
      try {
        const response = await fetch(`${API_BASE}/semantic-annotations/`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            descId: selectedClassCard.value.descId,
            annotations: expertMarks.value.map((mark) => ({
              viewName: mark.viewName,
              viewText: mark.viewText,
              annotatedText: mark.text,
              effect: mark.effect,
              annotationContent: mark.note || "",
              updateRevise: mark.updateRevise || "",
            })),
            detailUpdates: { ...detailEdits },
          }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.message || "Semantic annotations save fails.");
        notifyMessage("Semantic annotations save successfully.", "success");
        closeClassDetail(false);
      } catch (error) {
        notifyMessage(error.message || "Semantic annotations save fails.", "error");
      }
    };

    const openMarkContextMenu = (event, mark) => {
      event.preventDefault();
      event.stopPropagation();
      expertMarkMenu.visible = true;
      expertMarkMenu.x = event.clientX + 10;
      expertMarkMenu.y = event.clientY + 8;
      expertMarkMenu.markId = mark.id;
      expertAnnotation.visible = false;
    };

    const renderAnnotatedText = (text, viewName) => {
      const content = String(text || "");
      const marks = expertMarks.value
          .filter((mark) => mark.viewName === viewName && mark.text && content.includes(mark.text))
          .map((mark) => ({ ...mark, index: content.indexOf(mark.text) }))
          .filter((mark) => mark.index >= 0)
          .sort((a, b) => a.index - b.index);
      if (!marks.length) return content;
      const nodes = [];
      let cursor = 0;
      marks.forEach((mark) => {
        if (mark.index < cursor) return;
        if (mark.index > cursor) nodes.push(content.slice(cursor, mark.index));
        nodes.push(h("span", {
          class: ["expert-mark", mark.effect],
          title: mark.note || `${mark.effect}: ${mark.text}`,
          onDblclick: (event) => {
            event.stopPropagation();
            openExpertNote(mark.id);
          },
          onContextmenu: (event) => openMarkContextMenu(event, mark),
        }, mark.text));
        cursor = mark.index + mark.text.length;
      });
      if (cursor < content.length) nodes.push(content.slice(cursor));
      return nodes;
    };
    // For a1-a4 cards, the primary view is already rendered in card.view.
    // Return only two additional views so the primary view name is not duplicated.
    const collapsedBriefItems = (card) => {
      if (!isSoilDomain()) return [];
      const primary = primaryBriefItem(card);
      const remaining = semanticBriefItems(card).filter((item) => item[0] !== primary[0]);
      return stablePickBriefItems(remaining, `${props.domain}-${card.cls}`, 2);
    };

    const normalizeGeneratedCards = (descriptions = [], run = {}) =>
        descriptions.map((item) => ({
          id: item.id || item.descId,
          descId: item.descId || item.id,
          cls: item.cls,
          view: item.view,
          text: item.briefDescription || item.text,
          detailedDescription: item.detailedDescription,
          status: item.status || "Generated",
          score: Math.round(Number(item.score) || 0),
          details: item.details || item.views || [],
          llmName: run.llmName || semanticLlm.value,
          useExpertKnowledge: Boolean(run.useExpertKnowledge),
          useImageAssist: Boolean(run.useImageAssist),
          startedAt: run.createdAt || run.generatedAt || "",
          endedAt: item.createdAt || item.generatedAt || "",
        }));

    const loadLatestSemanticGeneration = async () => {
      if (!props.domain) return;
      semanticLoading.value = true;
      try {
        const response = await fetch(
            `${API_BASE}/semantic-generation/latest/?domain=${encodeURIComponent(props.domain)}&llmName=${encodeURIComponent(semanticLlm.value)}&useExpertKnowledge=${useExpertKnowledge.value}&useImageAssist=${useImageAssist.value}`
        );
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.message || `${props.title} semantic loading fails.`);
        generatedCards.value = payload.run?.descriptions
            ? normalizeGeneratedCards(payload.run.descriptions, payload.run)
            : [];
      } catch {
        generatedCards.value = [];
      } finally {
        semanticLoading.value = false;
      }
    };

    watch([() => props.domain, semanticLlm, useExpertKnowledge, useImageAssist], () => {
      if (!props.domain) return;
      loadLatestSemanticGeneration();
    }, {flush: "post"});

    onMounted(() => {
      window.setTimeout(() => {
        if (props.domain) loadLatestSemanticGeneration();
      }, 0);
    });
    const semanticProgressViews = [
      "Dominant Shape",
      "Interaction Type",
      "Reflection Strength",
      "Feature Footprint",
      "Signal Coherence",
      "Anomaly Source",
      "Pattern Uniformity",
      "Feature Expression",
      "Layer Interaction",
      "Signal Complexity",
    ];

    const semanticProgressClasses = () => {
      const recordClasses = activeDomainRecord()?.supportedClasses;
      if (Array.isArray(recordClasses) && recordClasses.length) return recordClasses;
      return isSoilDomain() ? soilClasses : roadClasses;
    };

    const semanticProgressTasks = () =>
        semanticProgressClasses().flatMap((className) =>
            semanticProgressViews.map((viewName) => `${className} - ${viewName}`)
        );

    const generationPhase = (progress) => {
      const tasks = semanticProgressTasks();
      if (!tasks.length) return "Preparing semantic generation tasks...";
      const index = Math.min(tasks.length - 1, Math.max(0, Math.floor((progress / 100) * tasks.length)));
      return `Generating ${tasks[index]} semantic description...`;
    };

    const finishGeneration = (run) => {
      if (generationTimer.value) window.clearTimeout(generationTimer.value);
      generationTimer.value = null;
      generationController.value = null;
      if (run?.descriptions) {
        generatedCards.value = normalizeGeneratedCards(run.descriptions, run);
      }
      generation.progress = 100;
      generation.phase = "Semantic generation completed.";
      generation.status = "success";
      window.setTimeout(() => {
        generation.active = false;
        generation.runId = null;
        notifyMessage(`${props.title} semantic generation succeeds.`, "success");
      }, 520);
    };
    const postSemanticGenerationAction = async (action, runId = null) => {
      const response = await fetch(`${API_BASE}/semantic-generation/`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        signal: generationController.value?.signal,
        body: JSON.stringify({
          action,
          runId,
          domainName: activeDomainRecord()?.name || props.domain,
          areaRole: props.type,
          llmName: semanticLlm.value,
          useExpertKnowledge: useExpertKnowledge.value,
          useImageAssist: useImageAssist.value,
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.message || `${props.title} semantic generation fails.`);
      return payload;
    };

    const cancelSemanticGeneration = async () => {
      if (!generation.active) return;
      if (generationTimer.value) window.clearTimeout(generationTimer.value);
      generationTimer.value = null;
      const runId = generation.runId;
      if (generationController.value) {
        generationController.value.abort();
        generationController.value = null;
      }
      if (runId) {
        try {
          await fetch(`${API_BASE}/semantic-generation/`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({action: "cancel", runId}),
          });
        } catch {
        }
      }
      generation.status = "warning";
      generation.phase = "Semantic generation is cancelled.";
      window.setTimeout(() => {
        generation.active = false;
        generation.runId = null;
        notifyMessage(`${props.title} semantic generation is cancelled.`, "warning");
      }, 260);
    };

    const startSemanticGeneration = async () => {
      if (generation.active) return;
      if (generationTimer.value) window.clearTimeout(generationTimer.value);
      if (generationController.value) generationController.value.abort();

      const startedAt = Date.now();
      const minimumGenerationMs = 168000;
      generation.active = true;
      generation.progress = 0;
      generation.status = "running";
      generation.phase = "Preparing semantic generation run...";
      generation.runId = null;
      generationController.value = new AbortController();

      try {
        const startPayload = await postSemanticGenerationAction("start");
        generation.runId = startPayload.run?.id || startPayload.runId;

        const scheduleGenerationTick = () => {
          const elapsedRatio = Math.min(0.96, (Date.now() - startedAt) / minimumGenerationMs);
          const target = Math.min(96, Math.floor(8 + elapsedRatio * 88 + Math.sin(elapsedRatio * 18) * 3));
          const jitter = Math.random() < 0.22 ? 0 : Math.floor(Math.random() * 3) + 1;
          generation.progress = Math.min(96, Math.max(generation.progress, Math.min(target, generation.progress + jitter)));
          generation.phase = generationPhase(generation.progress);
          const nextDelay = 850 + Math.floor(Math.random() * 2800);
          generationTimer.value = window.setTimeout(scheduleGenerationTick, nextDelay);
        };
        generationTimer.value = window.setTimeout(scheduleGenerationTick, 700);

        const remainingMs = Math.max(0, minimumGenerationMs - (Date.now() - startedAt));
        if (remainingMs > 0) await new Promise((resolve) => window.setTimeout(resolve, remainingMs));
        if (generationController.value?.signal.aborted) return;
        generation.progress = 98;
        generation.phase = "Writing generated semantic descriptions to database...";
        const completePayload = await postSemanticGenerationAction("complete", generation.runId);
        finishGeneration(completePayload.run);
      } catch (error) {
        if (error.name === "AbortError") return;
        if (generationTimer.value) window.clearTimeout(generationTimer.value);
        generationTimer.value = null;
        generationController.value = null;
        generation.status = "error";
        generation.phase = "Semantic generation failed.";
        if (generation.runId) {
          try {
            await postSemanticGenerationAction("cancel", generation.runId);
          } catch {
          }
        }
        window.setTimeout(() => {
          generation.active = false;
          generation.runId = null;
          notifyMessage(error.message || `${props.title} semantic generation fails.`, "error");
        }, 320);
      }
    };

    const dbRangeLabel = (range, unit = "") => {
      if (!range) return "Not configured";
      if (range.label) return range.label;
      const min = range.min;
      const max = range.max;
      const suffix = unit ? ` ${unit}` : "";
      if (min === null || min === undefined || max === null || max === undefined) return "Not configured";
      return Number(min) === Number(max) ? `${min}${suffix}` : `${min}-${max}${suffix}`;
    };

    const dbPointLabel = (value, unit = "") => {
      if (value === null || value === undefined || value === "") return "Not configured";
      return unit ? `${value} ${unit}` : `${value}`;
    };

    const metricSummaryRows = () => {
      const record = activeDomainRecord();
      const preview = record?.preview || {};
      const updatedTime = record?.detail?.updatedAt
          ? new Date(record.detail.updatedAt).toLocaleString()
          : "-";

      if (isSoilDomain()) {
        const composition = preview.composition || {};
        return [
          ["Sand", dbPointLabel(composition.sand, "%")],
          ["Silt", dbPointLabel(composition.silt, "%")],
          ["Clay", dbPointLabel(composition.clay, "%")],
          ["Water Content", dbRangeLabel(preview.waterContent, "%")],
          ["Relative Permittivity", dbRangeLabel(preview.relativePermittivity)],
          ["Conductivity", dbRangeLabel(preview.conductivity, "S/m")],
          ["Peplinski Model Fractal Dimension", dbPointLabel(preview.peplinskiDimension)],
          ["Updated Time", updatedTime],
        ];
      }

      return [
        ["Road Surface", preview.roadSurface || props.meta.roadSurface || "Not configured"],
        ["Frequency Range", dbRangeLabel(preview.frequencyRange, "MHz")],
        ["Time Window", preview.timeWindow?.label || dbPointLabel(preview.timeWindow?.value, "ns")],
        ["Updated Time", updatedTime],
      ];
    };

    const areaDescriptionText = () => {
      if (isSoilDomain()) {
        return `${activeDomainRecord()?.name || props.domain} represents a soil-domain scenario controlled by particle composition, water content, dielectric response, conductivity, and Peplinski-model fractal structure. The area description is used to condition class semantics so that cavity, crack, and metal-pipeline signatures can be interpreted under the selected subsurface material profile.`;
      }

      const metrics = Object.fromEntries(props.meta.metrics.map((metric) => [metric.name, metric.value]));
      return `${activeDomainRecord()?.name || props.domain} represents a road-domain GPR acquisition scenario with ${props.meta.condition.toLowerCase()} and ${props.meta.roadSurface.toLowerCase()} surface context. The frequency range (${metrics["Frequency Range"]}) and time window (${metrics["Time Window"]}) define how reflected energy, clutter, and layer interactions are interpreted before generating class-level semantic descriptions.`;
    };

    const openAreaDetail = async () => {
      const record = activeDomainRecord();
      if (!record?.code) {
        showAreaDetail.value = true;
        return;
      }

      try {
        const response = await fetch(`${API_BASE}/domains/code/${encodeURIComponent(record.code)}/`);
        if (!response.ok) throw new Error("Domain detail request fails.");
        const detail = await response.json();
        backendDomains[detail.code] = detail;
        backendDomains[detail.name] = detail;
        if (detail.renderMode === "soil") applySoilProfile(detail);
        else applyRoadMeta(props.type === "source" ? sourceMeta : targetMeta, detail, props.type);
      } catch {
        notifyMessage(`${props.title} detail loading fails.`, "error");
      } finally {
        showAreaDetail.value = true;
      }
    };

    const downloadAreaDetail = () => {
      try {
        const record = activeDomainRecord();
        const payload = {
          area: props.title,
          domain: record?.name || props.domain,
          code: record?.code || props.domain,
          type: record?.renderMode || (isSoilDomain() ? "soil" : "road"),
          summary: Object.fromEntries(metricSummaryRows()),
          detail: record?.detail || {
            areaDescription: areaDescriptionText(),
          },
          userInputs: areaInputRows.value.filter((row) => row.domain === props.domain),
        };
        const blob = new Blob([JSON.stringify(payload, null, 2)], {type: "application/json"});
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `${payload.code}-area-detail.json`.replace(/\s+/g, "_");
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        notifyMessage(`${payload.domain} area detail downloads successfully.`, "success");
      } catch {
        notifyMessage(`${props.title} area detail download fails.`, "error");
      }
    };

    const triggerAreaUpload = () => {
      areaInputDialog.form = {
        datasetCode: "",
        datasetName: "",
        domainType: "road",
        roleCondition: props.type === "source" ? "Validated original area" : "Changed sensing condition",
        roadSurface: "",
        frequencyMode: "range",
        frequencyMin: "",
        frequencyMax: "",
        frequencyValue: "",
        timeWindow: "",
        sandPercent: "",
        siltPercent: "",
        clayPercent: "",
        waterMode: "range",
        waterMin: "",
        waterMax: "",
        waterValue: "",
        permittivityMode: "range",
        permittivityMin: "",
        permittivityMax: "",
        permittivityValue: "",
        conductivityMode: "range",
        conductivityMin: "",
        conductivityMax: "",
        conductivityValue: "",
        peplinskiDimension: "",
        classList: "",
        areaDescription: "",
        signalBehavior: "",
        semanticUsage: "",
      };
      areaInputDialog.visible = true;
    };

    const triggerAreaImport = () => {
      areaImportInput.value?.click();
    };

    const cleanAreaNumber = (value) => {
      if (value === null || value === undefined) return "";
      const match = String(value).replace(/,/g, "").match(/-?\d+(?:\.\d+)?/);
      return match ? match[0] : "";
    };

    const applyAreaRangeText = (form, prefix, value) => {
      if (value === null || value === undefined || value === "") return;
      const numbers = String(value).replace(/,/g, "").match(/-?\d+(?:\.\d+)?/g) || [];
      if (numbers.length >= 2) {
        form[`${prefix}Mode`] = "range";
        form[`${prefix}Min`] = numbers[0];
        form[`${prefix}Max`] = numbers[1];
        form[`${prefix}Value`] = "";
      } else if (numbers.length === 1) {
        form[`${prefix}Mode`] = "fixed";
        form[`${prefix}Value`] = numbers[0];
        form[`${prefix}Min`] = "";
        form[`${prefix}Max`] = "";
      }
    };

    const handleAreaImport = (event) => {
      const file = event.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const data = JSON.parse(String(reader.result || "{}"));
          const summary = data.summary || data;
          const detail = data.detail || data;
          const form = {...areaInputDialog.form};
          form.datasetCode = data.code || form.datasetCode;
          form.datasetName = data.domain || data.domainName || data.name || form.datasetName;
          form.domainType = String(data.type || form.domainType).toLowerCase().includes("soil") ? "soil" : "road";
          if (data.area) form.roleCondition = String(data.area).toLowerCase().includes("new") ? "Changed sensing condition" : "Validated original area";
          form.roadSurface = summary["Road Surface"] || summary.roadSurface || form.roadSurface;
          form.timeWindow = cleanAreaNumber(summary["Time Window"] || summary.timeWindow || form.timeWindow);
          form.sandPercent = cleanAreaNumber(summary.Sand || summary.sand || form.sandPercent);
          form.siltPercent = cleanAreaNumber(summary.Silt || summary.silt || form.siltPercent);
          form.clayPercent = cleanAreaNumber(summary.Clay || summary.clay || form.clayPercent);
          form.peplinskiDimension = cleanAreaNumber(summary["Peplinski Model Fractal Dimension"] || summary.peplinskiDimension || form.peplinskiDimension);
          form.classList = data.Class || data.class || data.classes || data.supportedClasses || form.classList;
          if (Array.isArray(form.classList)) form.classList = form.classList.join(", ");
          applyAreaRangeText(form, "frequency", summary["Frequency Range"] || summary.frequencyRange);
          applyAreaRangeText(form, "water", summary["Water Content"] || summary.waterContent);
          applyAreaRangeText(form, "permittivity", summary["Relative Permittivity"] || summary.relativePermittivity);
          applyAreaRangeText(form, "conductivity", summary.Conductivity || summary.conductivity);
          form.areaDescription = detail.areaDescription || detail.area_description || form.areaDescription;
          form.signalBehavior = detail.signalBehavior || detail.signal_behavior || form.signalBehavior;
          form.semanticUsage = detail.semanticUsage || detail.semantic_usage || form.semanticUsage;
          areaInputDialog.form = form;
          notifyMessage(`${props.title} JSON import succeeds.`, "success");
        } catch {
          notifyMessage("JSON import fails: invalid JSON file.", "error");
        } finally {
          event.target.value = "";
        }
      };
      reader.readAsText(file);
    };

    const cancelAreaInput = () => {
      areaInputDialog.visible = false;
      areaInputDialog.form = {};
      notifyMessage("Dataset input is closed.", "info");
    };

    const areaInputFields = () => [
      {key: "datasetName", label: "Dataset Name", type: "text", required: true},
      {
        key: "roleCondition", label: "Area Condition", type: "select", required: true, options: [
          {value: "Validated original area", label: "Validated original area"},
          {value: "Changed sensing condition", label: "Changed sensing condition"},
        ]
      },
      {
        key: "domainType", label: "Type", type: "select", required: true, options: [
          {value: "road", label: "Road GPR"},
          {value: "soil", label: "Soil GPR"},
        ]
      },
      {key: "roadSurface", label: "Road Surface", type: "text"},
      {key: "timeWindow", label: "Time Window (ns)", type: "number"},
      {key: "peplinskiDimension", label: "Peplinski Model Fractal Dimension", type: "number"},
      {key: "sandPercent", label: "Sand", type: "number"},
      {key: "siltPercent", label: "Silt", type: "number"},
      {key: "clayPercent", label: "Clay", type: "number"},
      {
        key: "frequency",
        label: "Frequency",
        type: "rangeGroup",
        modeKey: "frequencyMode",
        minKey: "frequencyMin",
        maxKey: "frequencyMax",
        valueKey: "frequencyValue",
        minLabel: "Min (MHz)",
        maxLabel: "Max (MHz)",
        valueLabel: "Value (MHz)"
      },
      {
        key: "water",
        label: "Water Content",
        type: "rangeGroup",
        modeKey: "waterMode",
        minKey: "waterMin",
        maxKey: "waterMax",
        valueKey: "waterValue",
        minLabel: "Min",
        maxLabel: "Max",
        valueLabel: "Value"
      },
      {
        key: "permittivity",
        label: "Relative Permittivity",
        type: "rangeGroup",
        modeKey: "permittivityMode",
        minKey: "permittivityMin",
        maxKey: "permittivityMax",
        valueKey: "permittivityValue",
        minLabel: "Min",
        maxLabel: "Max",
        valueLabel: "Value"
      },
      {
        key: "conductivity",
        label: "Conductivity",
        type: "rangeGroup",
        modeKey: "conductivityMode",
        minKey: "conductivityMin",
        maxKey: "conductivityMax",
        valueKey: "conductivityValue",
        minLabel: "Min",
        maxLabel: "Max",
        valueLabel: "Value"
      },
      {key: "classList", label: "Class", type: "textarea", required: true, wide: true},
      {key: "areaDescription", label: "Area Description", type: "textarea", required: true, wide: true},
      {key: "signalBehavior", label: "Signal Behavior", type: "textarea", halfLeft: true},
      {key: "semanticUsage", label: "Semantic Usage", type: "textarea", halfRight: true},
    ];

    const buildAreaInputPayload = () => {
      const form = areaInputDialog.form;
      const payload = {
        datasetCode: String(form.datasetCode || "").trim(),
        datasetName: String(form.datasetName || "").trim(),
        domainType: form.domainType || "road",
        roleCondition: form.roleCondition || "",
        roadSurface: form.roadSurface || "",
        timeWindow: form.timeWindow || "",
        peplinskiDimension: form.peplinskiDimension || "",
        sandPercent: form.sandPercent || "",
        siltPercent: form.siltPercent || "",
        clayPercent: form.clayPercent || "",
        areaDescription: form.areaDescription || "",
        signalBehavior: form.signalBehavior || "",
        semanticUsage: form.semanticUsage || "",
        classList: form.classList || "",
      };
      [["frequency", "frequency"], ["water", "water"], ["permittivity", "permittivity"], ["conductivity", "conductivity"]].forEach(([prefix, target]) => {
        const mode = form[`${prefix}Mode`] || "range";
        payload[`${target}Mode`] = mode;
        if (mode === "fixed") {
          payload[`${target}Value`] = form[`${prefix}Value`] || "";
          payload[`${target}Min`] = form[`${prefix}Value`] || "";
          payload[`${target}Max`] = form[`${prefix}Value`] || "";
        } else {
          payload[`${target}Min`] = form[`${prefix}Min`] || "";
          payload[`${target}Max`] = form[`${prefix}Max`] || "";
          payload[`${target}Value`] = "";
        }
      });
      return payload;
    };

    const saveAreaInput = async () => {
      const missing = areaInputFields().find((field) => field.required && !areaInputDialog.form[field.key]);
      if (missing) {
        notifyMessage(`${missing.label} is required.`, "warning");
        return;
      }
      try {
        const payload = buildAreaInputPayload();
        const response = await fetch(`${API_BASE}/domains/`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        const result = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(result.message || "Dataset input save fails.");
        if (result.domain?.code) backendDomains[result.domain.code] = result.domain;
        if (result.domain?.name) backendDomains[result.domain.name] = result.domain;
        await loadDomains(false);
        emit("update:domain", result.domain?.name || payload.datasetName);
        areaInputDialog.visible = false;
        areaInputDialog.form = {};
        notifyMessage(`${props.title} area input is saved.`, "success");
      } catch (error) {
        notifyMessage(error.message || `${props.title} area input save fails.`, "error");
      }
    };
    const renderAreaInputModal = () =>
        h(Transition, {name: "modal-fade"}, () =>
            areaInputDialog.visible
                ? h("div", {class: "area-input-overlay", onClick: cancelAreaInput}, [
                  h("section", {class: ["area-input-modal", props.type], onClick: (event) => event.stopPropagation()}, [
                    h("div", {class: "area-input-head"}, [
                      h("div", [
                        h("span", `${props.title} Dataset Input`),
                      ]),
                      h("div", {class: "modal-head-actions"}, [
                        h("button", {type: "button", class: "import-button", onClick: triggerAreaImport}, "File Input"),
                        h("button", {type: "button", onClick: cancelAreaInput}, "Close"),
                        h("input", {
                          ref: areaImportInput,
                          class: "area-import-input",
                          type: "file",
                          accept: ".json,application/json",
                          onChange: handleAreaImport,
                        }),
                      ]),
                    ]),
                    h("div", {class: "area-input-form"}, [
                      ...areaInputFields().map((field) =>
                          field.type === "rangeGroup"
                              ? h("div", {class: "range-group-field"}, [
                                h("span", [h("b", field.label)]),
                                h("div", {class: "range-group-controls"}, [
                                  h("select", {
                                    value: areaInputDialog.form[field.modeKey] || "range",
                                    onChange: (event) => {
                                      areaInputDialog.form[field.modeKey] = event.target.value;
                                    },
                                  }, [
                                    h("option", {value: "range"}, "Range value"),
                                    h("option", {value: "fixed"}, "Fixed value"),
                                  ]),
                                  ...(areaInputDialog.form[field.modeKey] === "fixed"
                                      ? [
                                        h("input", {
                                          value: areaInputDialog.form[field.valueKey] ?? "",
                                          type: "number",
                                          placeholder: field.valueLabel,
                                          onInput: (event) => {
                                            areaInputDialog.form[field.valueKey] = event.target.value;
                                          },
                                        }),
                                        h("div", {class: "range-placeholder"}),
                                      ]
                                      : [
                                        h("input", {
                                          value: areaInputDialog.form[field.minKey] ?? "",
                                          type: "number",
                                          placeholder: field.minLabel,
                                          onInput: (event) => {
                                            areaInputDialog.form[field.minKey] = event.target.value;
                                          },
                                        }),
                                        h("input", {
                                          value: areaInputDialog.form[field.maxKey] ?? "",
                                          type: "number",
                                          placeholder: field.maxLabel,
                                          onInput: (event) => {
                                            areaInputDialog.form[field.maxKey] = event.target.value;
                                          },
                                        }),
                                      ]),
                                ]),
                              ])
                              : h("label", {
                                class: [
                                  field.wide ? "wide-field" : "",
                                  field.halfLeft ? "half-left-field" : "",
                                  field.halfRight ? "half-right-field" : "",
                                  field.required ? "required-field" : "",
                                ],
                              }, [
                                h("span", [
                                  field.required ? h("i", {class: "required-star"}, "*") : null,
                                  h("b", field.label),
                                ]),
                                field.type === "select"
                                    ? h(
                                        "select",
                                        {
                                          class: areaInputDialog.form[field.key] ? "" : "is-placeholder",
                                          value: areaInputDialog.form[field.key] ?? "",
                                          onChange: (event) => {
                                            areaInputDialog.form[field.key] = event.target.value;
                                          },
                                        },
                                        [
                                          field.placeholder
                                              ? h("option", {value: "", disabled: true}, field.placeholder)
                                              : null,
                                          ...field.options.map((option) =>
                                              h("option", {value: option.value}, option.label)
                                          ),
                                        ]
                                    )
                                    : field.type === "textarea"
                                        ? h("textarea", {
                                          value: areaInputDialog.form[field.key] ?? "",
                                          placeholder: `Enter ${field.label.toLowerCase()}`,
                                          onInput: (event) => {
                                            areaInputDialog.form[field.key] = event.target.value;
                                          },
                                        })
                                        : h("input", {
                                          value: areaInputDialog.form[field.key] ?? "",
                                          type: field.type,
                                          placeholder: `Enter ${field.label.toLowerCase()}`,
                                          onInput: (event) => {
                                            areaInputDialog.form[field.key] = event.target.value;
                                          },
                                        }),
                              ])
                      ),
                    ]),
                    h("div", {class: "area-input-actions"}, [
                      h("button", {type: "button", class: "cancel-input", onClick: cancelAreaInput}, "Cancel"),
                      h("button", {type: "button", class: "save-input", onClick: saveAreaInput}, "Save"),
                    ]),
                  ]),
                ])
                : null
        );

    const closeAreaDetail = () => {
      showAreaDetail.value = false;
      // notifyMessage(`${props.title} detail is closed.`, "info");
    };
    const renderAreaDetailModal = () =>
        h(Transition, {name: "modal-fade"}, () =>
            showAreaDetail.value
                ? h("div", {class: "area-detail-overlay", onClick: closeAreaDetail}, [
                  h("section", {class: ["area-detail-modal", props.type], onClick: (event) => event.stopPropagation()}, [
                    h("div", {class: "area-detail-head"}, [
                      h("div", [
                        h("span", `${props.title} Detail`),
                      ]),
                      h("div", {class: "modal-head-actions"}, [
                        h("button", {type: "button", class: "download-button", onClick: downloadAreaDetail}, "Download"),
                        h("button", {type: "button", onClick: closeAreaDetail}, "Close"),
                      ]),
                    ]),
                    h(
                        "div",
                        {class: "area-detail-grid"},
                        metricSummaryRows().map(([key, value]) =>
                            h("div", [
                              h("span", key),
                              h("strong", value),
                            ])
                        )
                    ),
                    h("div", {class: "area-description-panel"}, [
                      h("span", "Area Description"),
                      h("p", activeDomainRecord()?.detail?.areaDescription || areaDescriptionText()),
                    ]),
                    activeDomainRecord()?.detail
                        ? h("div", {class: "area-detail-extra-grid"}, [
                          h("section", [
                            h("span", "Signal Behavior"),
                            h("p", activeDomainRecord().detail.signalBehavior),
                          ]),
                          h("section", [
                            h("span", "Semantic Usage"),
                            h("p", activeDomainRecord().detail.semanticUsage),
                          ]),
                        ])
                        : null,
                  ]),
                ])
                : null
        );

    const renderGenerationOverlay = () =>
        h(Transition, {name: "modal-fade"}, () =>
            generation.active
                ? h("div", {class: "generation-overlay"}, [
                  h("section", {class: ["generation-modal", props.type, generation.status]}, [
                    h("div", {class: "generation-modal-head"}, [
                      h("div", [
                        h("span", `${props.title} Semantic Generation`),
                        h("h3", generation.status === "warning" ? "Generation Interrupted" : "Generating Class Semantics"),
                      ]),
                      h("b", `${generation.progress}%`),
                    ]),
                    h("div", {class: "generation-progress"}, [
                      h("i", {style: {width: `${generation.progress}%`}}),
                    ]),
                    h("p", generation.phase),
                    h("div", {class: "generation-modal-actions"}, [
                      h("button", {type: "button", onClick: cancelSemanticGeneration}, "Cancel"),
                    ]),
                  ]),
                ])
                : null
        );

    const annotationEffectLabel = (effect) => {
      if (effect === "incorrect") return "Incorrect";
      if (effect === "inaccurate") return "Inaccurate";
      return "Correct";
    };
    const setDetailEditMode = () => {
      // Click again to leave edit mode and discard unsaved text changes.
      if (detailEditMode.value) {
        detailEditMode.value = false;
        Object.keys(detailEdits).forEach((key) => delete detailEdits[key]);
        return;
      }

      // Enter edit mode with the current detail text as the editable value.
      semanticDetailItems(selectedClassCard.value).forEach((item) => {
        if (detailEdits[item[0]] === undefined) detailEdits[item[0]] = item[1];
      });
      detailEditMode.value = true;
    };
    const renderExpertAnnotationControls = () => [
      expertAnnotation.visible
          ? h("div", {
            class: ["expert-annotation-popover", props.type],
            style: { left: `${expertAnnotation.x}px`, top: `${expertAnnotation.y}px` },
            onClick: (event) => event.stopPropagation(),
          }, [
            h("span", "Expert Annotation"),
            h("div", [
              h("button", { type: "button", class: "mark-correct", onClick: () => addExpertMark("correct") }, "Correct"),
              h("button", { type: "button", class: "mark-inaccurate", onClick: () => addExpertMark("inaccurate") }, "Inaccurate"),
              h("button", { type: "button", class: "mark-incorrect", onClick: () => addExpertMark("incorrect") }, "Incorrect"),
            ]),
          ])
          : null,
      expertMarkMenu.visible
          ? h("div", {
            class: ["expert-annotation-popover", "mark-context-menu", props.type],
            style: { left: `${expertMarkMenu.x}px`, top: `${expertMarkMenu.y}px` },
            onClick: (event) => event.stopPropagation(),
          }, [
            h("span", "Annotation Actions"),
            h("div", [
              h("button", { type: "button", class: "mark-delete", onClick: () => deleteExpertMark() }, "Delete"),
              h("button", { type: "button", class: "mark-open", onClick: () => openExpertNote() }, "Open Note"),
              h("button", { type: "button", class: "mark-close", onClick: () => (expertMarkMenu.visible = false) }, "Close"),
            ]),
          ])
          : null,
      h(ElDrawer, {
        modelValue: expertAnnotation.noteOpen,
        "onUpdate:modelValue": (value) => {
          expertAnnotation.noteOpen = value;
        },
        direction: "rtl",
        size: "min(440px, 34vw)",
        title: "Expert Annotation",
        class: ["expert-annotation-drawer", props.type],
      }, {
        default: () => h("div", { class: "drawer-annotation-body" }, [
          h("div", { class: "drawer-annotation-section" }, [
            h("span", "Selected Text"),
            h("strong", expertAnnotation.text || "-"),
          ]),
          h("div", { class: "drawer-annotation-section" }, [
            h("span", "Annotation Effect"),
            h("strong", annotationEffectLabel(expertAnnotation.effect)),
          ]),
          h("div", { class: "drawer-annotation-section" }, [
            h("span", "Annotation Time"),
            h("strong", formatSemanticTime(expertAnnotation.annotatedAt)),
          ]),
          h("div", { class: "drawer-annotation-section" }, [
            h("span", "Annotation Note"),
            h("textarea", {
              value: expertAnnotation.note,
              placeholder: "Enter expert annotation note",
              onInput: (event) => (expertAnnotation.note = event.target.value),
            }),
          ]),
          expertAnnotation.effect !== "correct"
              ? h("div", { class: "drawer-annotation-section revise-block" }, [
                h("span", "Update Revise"),
                h("textarea", {
                  class: "revise-textarea",
                  value: expertAnnotation.updateRevise,
                  placeholder: "Enter demonstration correction",
                  onInput: (event) => (expertAnnotation.updateRevise = event.target.value),
                }),
              ])
              : null,
          h("button", { type: "button", onClick: saveExpertNote }, "Save Note"),
        ]),
      }),
    ];
    const renderClassDetailModal = () => {
      const card = selectedClassCard.value;
      if (!card) return h(Transition, {name: "modal-fade"}, () => null);

      const downloadClassDetail = () => {
        try {
          const payload = {
            area: props.title,
            domain: activeDomainName(),
            domainName: activeDomainRecord()?.name || props.domain,
            class: card.cls,
            semanticConfidence: `${card.score}%`,
            generationStartTime: card.startedAt,
            generationEndTime: card.endedAt,
            llmName: classDetailConfig.llmName,
            useExpertKnowledge: classDetailConfig.useExpertKnowledge,
            useImageAssist: classDetailConfig.useImageAssist,
            briefDetails: Object.fromEntries(semanticBriefItems(card)),
            detailedDetails: Object.fromEntries(semanticDetailItems(card).map((item) => [item[0], detailEdits[item[0]] ?? item[1]])),
            expertAnnotations: expertMarks.value,
          };
          const blob = new Blob([JSON.stringify(payload, null, 2)], {type: "application/json"});
          const url = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = `${props.domain}-${card.cls}-semantic-detail.json`.replace(/\s+/g, "_");
          document.body.appendChild(link);
          link.click();
          link.remove();
          URL.revokeObjectURL(url);
          notifyMessage(`${card.cls} semantic detail downloads successfully.`, "success");
        } catch {
          notifyMessage(`${card.cls} semantic detail download fails.`, "error");
        }
      };

      return h(Transition, {name: "modal-fade"}, () =>
          h("div", {class: "area-detail-overlay", onClick: closeClassDetail}, [
            h("section", {class: ["class-detail-modal", props.type], onClick: (event) => event.stopPropagation()}, [
              h("div", {class: "area-detail-head"}, [
                h("div", [
                  h("span", `${props.title} Class Detail`),
                  h("h3", card.cls),
                ]),
                h("div", {class: "modal-head-actions"}, [
                  h(
                      "button",
                      {type: "button", class: "edit-button", onClick: setDetailEditMode},
                      detailEditMode.value ? "Cancel Edit" : "Edit"
                  ),
                  h("button", {type: "button", class: "download-button", onClick: downloadClassDetail}, "Download"),
                  h("button", {type: "button", onClick: closeClassDetail}, "Close"),
                ]),
              ]),
              h("div", {class: "class-detail-summary"}, [
                h("div", [h("span", "Primary Key"), h("strong", primaryBriefItem(card)[0] || "Dominant Shape")]),
                h("div", [h("span", "Semantic Confidence"), h("strong", `${card.score}%`)]),
                h("div", [h("span", "Area Domain"), h("strong", activeDomainName())]),
                h("div", {class: "class-detail-select-card"}, [
                  h("span", "LLM"),
                  h("select", {
                    value: classDetailConfig.llmName,
                    disabled: classDetailConfig.loading,
                    onChange: (event) => updateClassDetailFromDatabase({llmName: event.target.value}),
                  }, semanticLlmOptions.map((item) => h("option", {value: item}, item))),
                ]),
                h("div", {class: "class-detail-select-card"}, [
                  h("span", "Expert Knowledge"),
                  h("select", {
                    value: String(classDetailConfig.useExpertKnowledge),
                    disabled: classDetailConfig.loading,
                    onChange: (event) => updateClassDetailFromDatabase({useExpertKnowledge: event.target.value === "true"}),
                  }, [h("option", {value: "false"}, "Disabled"), h("option", {value: "true"}, "Enabled")]),
                ]),
                h("div", {class: "class-detail-select-card"}, [
                  h("span", "Image Assist"),
                  h("select", {
                    value: String(classDetailConfig.useImageAssist),
                    disabled: classDetailConfig.loading,
                    onChange: (event) => updateClassDetailFromDatabase({useImageAssist: event.target.value === "true"}),
                  }, [h("option", {value: "false"}, "Disabled"), h("option", {value: "true"}, "Enabled")]),
                ]),
              ]),
              h("div", {class: ["detail-grid", "class-detail-grid", classDetailConfig.loading ? "loading" : "", detailEditMode.value ? "editing" : ""]},
                  semanticDetailItems(card).map((item) => {
                    const detailText = detailEdits[item[0]] ?? item[1];
                    return h("div", [
                      h("span", item[0]),
                      detailEditMode.value
                          ? h("textarea", {
                            class: "detail-edit-textarea",
                            value: detailText,
                            onInput: (event) => (detailEdits[item[0]] = event.target.value),
                          })
                          : h("p", { onMouseup: (event) => handleAnnotationMouseUp(event, item[0], detailText) }, renderAnnotatedText(detailText, item[0])),
                    ]);
                  })
              ),
              h("div", {class: "class-detail-times"}, [
                h("div", [h("span", "Generation Start Time"), h("strong", formatSemanticTime(card.startedAt))]),
                h("div", [h("span", "Generation End Time"), h("strong", formatSemanticTime(card.endedAt))]),
              ]),
              h("div", {class: "class-detail-actions"}, [
                h("button", {type: "button", class: "cancel-class", onClick: closeClassDetail}, "Cancel"),
                h("button", {type: "button", class: "save-class", onClick: saveClassDetail}, "Save"),
              ]),
              ...renderExpertAnnotationControls(),
            ]),
          ])
      );
    };
    const renderSoilContext = () => {
      const profile = activeSoilProfile();
      const renderSoilMetric = (metric) =>
          h("div", {
            class: ["metric-row", metric.type, "soil-metric"],
            title: metric.type === "range" ? "Drag range handles" : "Click to adjust",
            onPointerdown: (event) => event.stopPropagation(),
            onClick: (event) => handleMetricClick(event, metric),
          }, [
            h("div", [h("span", metric.name), h("b", metric.value)]),
            renderMetricControl(metric),
          ]);

      return h("section", {
          class: ["context-box", "soil-context", "clickable-context"],
          title: `Click to inspect ${props.title.toLowerCase()} details`,
          onClick: () => {
            if (suppressAreaClick.value) return;
            openAreaDetail();
          },
        }, [
        h("div", {class: "soil-composition"}, [
          ...profile.composition.map(renderSoilMetric),
        ]),
        h("div", {class: "metric-stack soil-water"}, [
            h("div", {
              class: ["metric-row", profile.water.type],
              title: "Drag range handles",
              onPointerdown: (event) => event.stopPropagation(),
              onClick: (event) => event.stopPropagation(),
            }, [
            h("div", [h("span", profile.water.name), h("b", profile.water.value)]),
            renderMetricControl(profile.water),
          ]),
        ]),
        h("div", {class: "soil-electrical"}, [
          ...profile.electrical.map(renderSoilMetric),
        ]),
        renderSoilMetric(profile.peplinski),
      ]);
    };

      const renderRoadContext = () =>
          h("section", {
            class: ["context-box", "clickable-context"],
            title: `Click to inspect ${props.title.toLowerCase()} details`,
            onClick: () => {
              if (suppressAreaClick.value) return;
              openAreaDetail();
            },
          }, [
          h("div", {class: "context-meta"}, [
            h("div", [h("span", "Condition"), h("strong", props.meta.condition)]),
            h("div", [h("span", "Road Surface"), h("strong", props.meta.roadSurface)]),
          ]),
          h(
              "div",
              {class: "metric-stack"},
              props.meta.metrics.map((metric) =>
                  h("div", {
                    class: ["metric-row", metric.type],
                    title: metric.type === "range" ? "Drag range handles" : "Click to adjust",
                    onPointerdown: (event) => event.stopPropagation(),
                    onClick: (event) => handleMetricClick(event, metric),
                  }, [
                    h("div", [h("span", metric.name), h("b", metric.value)]),
                    renderMetricControl(metric),
                  ])
              )
          ),
        ]);

    return () =>
        h("article", {class: ["area-column", props.type]}, [
          h("div", {class: "area-topline"}, [
            h("div", [
              h("div", {class: "area-title-row"}, [
                h("h3", props.title),
                h(
                    "select",
                    {
                      class: "domain-select",
                      value: props.domain,
                      onChange: (event) => emit("update:domain", event.target.value),
                    },
                    props.domainOptions.map((option) =>
                        h("option", {value: optionValue(option)}, optionLabel(option))
                    )
                ),
              ]),
            ]),
            h(
                "button",
                {
                  class: ["area-upload", props.type],
                  type: "button",
                  onClick: triggerAreaUpload,
                },
                "Upload"
            ),
          ]),

          h("div", {class: "stage-section context-section"}, [
            isSoilDomain() ? renderSoilContext() : renderRoadContext(),
          ]),

          h("div", {class: "stage-section generation-section"}, [
            h("section", {class: "generation-box"}, [
              h("div", {class: "generation-copy"}, [
                h("strong", "Generate class semantics"),
              ]),
              h("div", {class: "generation-controls"}, [
                h("button", {
                  class: "primary-action",
                  type: "button",
                  onClick: startSemanticGeneration,
                }, "Generate"),
              ]),
              h("div", {class: "generation-option-row"}, [
                h("label", {class: "generation-select-wrap"}, [
                  h("span", "LLM"),
                  h("select", {
                    value: semanticLlm.value,
                    onChange: (event) => (semanticLlm.value = event.target.value),
                  }, [
                    "GPT-4o",
                    "GPT-3.5-turbo",
                    "GPT-4o-mini",
                    "Gemini-2.5",
                    "LLaMA-3.1",
                    "Qwen-2.5",
                  ].map((item) => h("option", {value: item}, item))),
                ]),
                h("label", {class: "assist-control compact"}, [
                  h("input", {
                    type: "checkbox",
                    checked: useExpertKnowledge.value,
                    onChange: (event) => (useExpertKnowledge.value = event.target.checked),
                  }),
                  h("em"),
                  h("span", "Expert knowledge"),
                ]),
                h("label", {class: "assist-control compact"}, [
                  h("input", {
                    type: "checkbox",
                    checked: useImageAssist.value,
                    onChange: (event) => (useImageAssist.value = event.target.checked),
                  }),
                  h("em"),
                  h("span", "Image assist"),
                ]),
              ]),
            ]),
          ]),

          h("div", {class: "stage-section cards-section"}, [
            h(
                "div",
                {
                  class: [
                    "semantic-scroll",
                    isSoilDomain() ? "three-cards" : "five-cards",
                    hasExpandedCard() ? "expanded-list" : "",
                  ],
                },
                semanticLoading.value
                    ? [
                      h("section", {class: "semantic-empty-state"}, [
                        h("strong", "Loading semantic descriptions..."),
                        h("p", "The system is reading the selected domain semantics from the database."),
                      ]),
                    ]
                    : activeCards().length
                        ? activeCards().map((card) =>
                            h("section", {
                              class: ["semantic-card", "clickable-card", expandedCards[cardKey(card)] ? "expanded" : ""],
                              title: `Click to inspect ${card.cls} details`,
                              onClick: () => {
                                openClassDetail(card);
                              },
                            }, [
                              h("div", {class: "semantic-head"}, [
                                h("div", [
                                  h("span", {class: "class-pill"}, card.cls),
                                  h("h4", card.view),
                                ]),
                                h(
                                    "button",
                                    {
                                      class: "card-toggle",
                                      type: "button",
                                      title: expandedCards[cardKey(card)] ? "Collapse details" : "Expand details",
                                      onClick: (event) => {
                                        event.stopPropagation();
                                        toggleCard(card);
                                      },
                                    },
                                    [h("span")]
                                ),
                              ]),
                              h(
                                  "div",
                                  {class: ["collapsed-detail", isSoilDomain() ? "multi" : "single"]},
                                  [
                                    // Primary view title is rendered once in semantic-head above.
                                    h("div", {class: "semantic-primary-view"}, [
                                      h("p", primaryBriefItem(card)[1]),
                                    ]),
                                    // a1-a4: render two additional view names with the same semantic-head/h4 style.
                                    ...(isSoilDomain()
                                        ? collapsedBriefItems(card).map((item) =>
                                            h("div", {class: "semantic-secondary-view"}, [
                                              h("div", {class: ["semantic-head", "secondary-semantic-head"]}, [
                                                h("div", [h("h4", item[0])]),
                                              ]),
                                              h("p", item[1]),
                                            ])
                                        )
                                        : []),
                                  ]
                              ),
                              expandedCards[cardKey(card)]
                                  ? h("div", {class: "card-detail"}, [
                                    h(
                                        "div",
                                        {class: "detail-grid"},
                                        semanticBriefItems(card).map((item) =>
                                            h("div", [
                                              h("span", item[0]),
                                              h("p", item[1]),
                                            ])
                                        )
                                    ),
                                  ])
                                  : null,
                              h("div", {class: "score-line"}, [
                                h("span", "Semantic confidence"),
                                h("div", {class: "score-track"}, [
                                  h("i", {style: {width: `${card.score}%`}}),
                                ]),
                                h("b", `${card.score}%`),
                              ]),
                            ])
                        )
                        : [
                          h("section", {class: "semantic-empty-state"}, [
                            h("strong", "No semantic descriptions"),
                            h("p", "No database record matches the selected domain, LLM, expert-knowledge, and image-assist settings."),
                          ]),
                        ]
            ),
          ]),
          renderAreaDetailModal(),
          renderAreaInputModal(),
          renderClassDetailModal(),
          renderGenerationOverlay(),
        ]);
  },
});

</script>

































































