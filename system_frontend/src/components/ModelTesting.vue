<template>
  <section class="system-panel reserved-panel">
    <PanelHeader
        code="EV"
        title="Model Testing and Comparison"
        subtitle="Test new-area samples, compare baselines, and review diagnosis outputs."
    />
    <EvaluationPanel />
  </section>
</template>

<script setup>
import { Transition, defineComponent, h, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import { ElMessage } from "element-plus";
import PanelHeader from "./PanelHeader.vue";
import {
  domainOptions,
  originalDomain,
  newDomain,
  roadClasses,
  soilClasses,
  testingRecords,
} from "../shared/gprState";

const API_BASE = process.env.VUE_APP_API_BASE_URL || "http://127.0.0.1:8000/api";

const EvaluationPanel = defineComponent({
  setup() {
    const testingRows = ref([...testingRecords]);
    const testingTotalPages = ref(1);
    const viewedTestingRecord = ref(null);
    const retrainingRecord = ref(null);
    const testingPage = ref(1);
    const testingPageJump = ref("1");
    const testingSearch = ref("");
    const appliedTestingSearch = ref("");
    const testingModelInputOpen = ref(false);
    const testingModelImportInput = ref(null);
    const testingModelInputForm = reactive({
      modelMethod: "",
      backbone: "",
      original: "",
      newArea: "",
      semanticGenerator: "",
      embeddingModel: "",
      optimizer: "",
      learningRate: "",
      batchSize: "",
      epochs: "",
      modelDescription: "",
    });
    const testingModelTraining = reactive({
      active: false,
      progress: 0,
      phase: "Preparing model testing training...",
    });
    const testingModelTrainingTimer = ref(null);
    const selectedTestingRecord = ref(testingRows.value[0] ? `${testingRows.value[0].method}-${testingRows.value[0].original}-${testingRows.value[0].newArea}-${testingRows.value[0].backbone}` : "");
    const evaluationImageUrl = ref("");
    const evaluationImageInput = ref(null);
    const evaluationImagePending = ref(false);
    const evaluationImageTimer = ref(null);
    const evaluationUploadedImage = reactive({
      imageId: "",
      imageName: "",
      imagePath: "",
    });
    const originalActivationReady = ref(false);
    const oursActivationReady = ref(false);
    const originalActivationImageUrl = ref("");
    const oursActivationImageUrl = ref("");
    const comparisonGenerationTimer = ref(null);
    const comparisonGeneration = reactive({
      active: false,
      method: "",
      progress: 0,
      phase: "Preparing comparison activation pipeline...",
    });
    const selectedOriginalMethod = ref("Original Classifier");
    const selectedOursMethod = ref("Ours");
    const comparisonMethodOptions = [
      "Original Classifier",
      "wDAE",
      "SubReg",
      "VGSE",
      "ICIS",
      "DANN",
      "ADDA",
      "TPDS",
      "G2KD",
      "Ours"
    ];
    const retrainingParams = reactive({
      optimizer: "AdamW",
      learningRate: "1e-5",
      batchSize: "8",
      epochs: "100",
      refinementIterations: "3",
      semanticGenerator: "GPT-4o",
      embeddingModel: "CLIP",
      knowledgeItems: "6",
      backbone: "ResNet-50",
      original: "GPR-SD",
      newArea: "GPR-Road",
      modelDescription: "",
    });
    const retrainingRun = reactive({
      active: false,
      progress: 0,
      phase: "Preparing retraining configuration...",
    });
    const retrainingTimer = ref(null);
    const retrainImportInput = ref(null);
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
    const closeTestingRecordDetail = () => {
      viewedTestingRecord.value = null;
      // notifyMessage("Testing record detail is closed.", "info");
    };
    const closeRetrainingModal = () => {
      retrainingRecord.value = null;
      notifyMessage("Retraining configuration is closed.", "info");
    };
    const normalizeTrainingRun = (row) => {
      const method = row.methodName || row.modelName || "Ours";
      const semanticGenerator = row.semanticGenerator || row.semantic_generator || "";
      const embeddingModel = row.embeddingModel || row.embedding_model || (method === "Ours" || semanticGenerator ? "CLIP" : "");
      return {
        id: row.id,
        method,
        modelName: row.modelName || row.methodName || "Ours",
        original: row.sourceDataset,
        newArea: row.targetDataset,
        backbone: row.backbone,
        semanticGenerator,
        embeddingModel,
        optimizer: row.optimizer || "",
        learningRate: row.learningRate || "",
        knowledgeItems: row.knowledgeItems,
        refinementIterations: row.refinementIterations,
        batchSize: row.batchSize,
        epochs: row.epochs,
        acc: row.acc || `${Number(row.accuracy || 0).toFixed(1)}%`,
        description: row.description || row.modelDescription || "",
        classAccuracy: row.classAccuracy || {},
        methodCheckpointPath: row.methodCheckpointPath || row.method_checkpoint_path || "",
        createdAt: row.createdAt,
        updatedAt: row.updatedAt,
      };
    };
    const loadTestingRuns = async (page = testingPage.value) => {
      try {
        const params = new URLSearchParams({ page: String(page), pageSize: "4" });
        if (appliedTestingSearch.value.trim()) params.set("search", appliedTestingSearch.value.trim());
        const response = await fetch(`${API_BASE}/model-training-runs/?${params.toString()}`);
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.message || "Model testing records loading fails.");
        testingRows.value = (payload.runs || []).map(normalizeTrainingRun);
        testingTotalPages.value = payload.pagination?.pageCount || 1;
        testingPage.value = payload.pagination?.page || page;
        testingPageJump.value = String(testingPage.value);
        if (testingRows.value.length && !testingRows.value.some((row) => recordKeyOf(row) === selectedTestingRecord.value)) {
          selectedTestingRecord.value = recordKeyOf(testingRows.value[0]);
        }
      } catch (error) {
        notifyMessage(error.message || "Model testing records loading fails.", "error");
      }
    };
    const handleTrainingRunCreated = () => {
      loadTestingRuns(1);
      loadDiagnosisSummary();
    };
    const formatDateTime = (value) => {
      if (!value) return "-";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return String(value);
      return date.toLocaleString();
    };
    const recordKeyOf = (row) => row.id ? `training-${row.id}` : `${row.method}-${row.original}-${row.newArea}-${row.backbone}`;
    const testingClassAccuracyRows = (row) => {
      const entries = Object.entries(row.classAccuracy || {});
      return entries.length ? entries : [["Overall Accuracy", Number.parseFloat(row.acc) || 0]];
    };
    const renderInlineAccuracyGrid = (row) => {
      const items = [["Overall Accuracy", Number.parseFloat(row.acc) || 0], ...testingClassAccuracyRows(row)];
      return h("div", { class: ["accuracy-inline-grid", items.length <= 4 ? "soil" : "road"] },
          items.map(([name, value]) => h("div", [h("span", name), h("strong", `${Number(value).toFixed(1)}%`)]))
      );
    };
    const applyTestingRunUpdate = (nextRow) => {
      const normalized = normalizeTrainingRun(nextRow);
      testingRows.value = testingRows.value.map((row) => row.id === normalized.id ? normalized : row);
      if (viewedTestingRecord.value?.id === normalized.id) viewedTestingRecord.value = normalized;
      if (retrainingRecord.value?.id === normalized.id) retrainingRecord.value = normalized;
      return normalized;
    };
    const normalizeDomainName = (value) => {
      if (!value) return "";
      if (typeof value === "string") return value;
      return value.name || value.code || String(value);
    };
    const diagnosisComparison = reactive({
      loading: false,
      methodA: null,
      methodB: null,
    });
    const isSoilLabel = (domain) => normalizeDomainName(domain).startsWith("a");
    const diagnosisClassLabels = () =>
        isSoilLabel(originalDomain.value) || isSoilLabel(newDomain.value) ? soilClasses : roadClasses;
    const toPercentNumber = (value) => {
      const parsed = Number.parseFloat(String(value ?? "").replace("%", ""));
      return Number.isFinite(parsed) ? parsed : 0;
    };
    const normalizeDiagnosisRun = (run, method) => {
      if (!run) return null;
      return {
        method,
        overall: Number(run.accuracy ?? toPercentNumber(run.acc)),
        classes: run.classAccuracy || {},
      };
    };
    const loadDiagnosisSummary = async () => {
      const params = new URLSearchParams({
        original: normalizeDomainName(originalDomain.value),
        new: normalizeDomainName(newDomain.value),
        methodA: selectedOriginalMethod.value,
        methodB: selectedOursMethod.value,
      });
      diagnosisComparison.loading = true;
      try {
        const response = await fetch(`${API_BASE}/model-training-runs/compare/?${params.toString()}`);
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.message || "Diagnosis summary loading fails.");
        diagnosisComparison.methodA = normalizeDiagnosisRun(payload.methodA, selectedOriginalMethod.value);
        diagnosisComparison.methodB = normalizeDiagnosisRun(payload.methodB, selectedOursMethod.value);
      } catch (error) {
        diagnosisComparison.methodA = null;
        diagnosisComparison.methodB = null;
        notifyMessage(error.message || "Diagnosis summary loading fails.", "error");
      } finally {
        diagnosisComparison.loading = false;
      }
    };
    const metricValueForClass = (metrics, label) => {
      if (!metrics) return 0;
      return Number(metrics.classes[label] ?? metrics.classes[label.replace("Metal Pipeline", "Pipeline")] ?? 0);
    };
    const renderAccuracyRows = (slot) => {
      const metrics = slot === "A" ? diagnosisComparison.methodA : diagnosisComparison.methodB;
      if (!metrics) return [h("div", { class: "accuracy-empty" }, "No matched record")];
      const labels = Object.keys(metrics.classes || {}).length ? Object.keys(metrics.classes) : diagnosisClassLabels();
      const rows = labels.map((label) => {
        const value = metricValueForClass(metrics, label);
        return h("div", { class: "class-accuracy-row" }, [
          h("div", [h("span", label), h("b", `${value.toFixed(1)}%`)]),
          h("i", { style: `width: ${value}%;` }),
        ]);
      });
      rows.push(
          h("div", { class: "class-accuracy-row overall" }, [
            h("div", [h("span", "Overall Accuracy"), h("b", `${metrics.overall.toFixed(1)}%`)]),
            h("i", { style: `width: ${metrics.overall}%;` }),
          ])
      );
      return rows;
    };
    const buildAccuracyExport = (slot, method) => {
      const metrics = slot === "A" ? diagnosisComparison.methodA : diagnosisComparison.methodB;
      return {
        method,
        overallAccuracy: metrics ? `${metrics.overall.toFixed(1)}%` : "No matched record",
        perClassAccuracy: metrics
            ? Object.fromEntries(Object.entries(metrics.classes || {}).map(([label, value]) => [label, `${Number(value).toFixed(1)}%`]))
            : {},
      };
    };
    const testingPageSize = 4;
    const filteredTestingRows = () => {
      const query = appliedTestingSearch.value.trim().toLowerCase();
      if (!query) return testingRows.value;

      return testingRows.value.filter((row) =>
          [row.method, row.original, row.newArea, row.backbone, row.acc]
              .some((value) => String(value).toLowerCase().includes(query))
      );
    };
    const testingPageCount = () => testingTotalPages.value;
    const pagedTestingRecords = () => testingRows.value;
    const setTestingPage = (page) => {
      const nextPage = Math.min(testingPageCount(), Math.max(1, page || 1));
      testingPage.value = nextPage;
      testingPageJump.value = String(testingPage.value);
      loadTestingRuns(nextPage);
    };
    const visibleTestingPages = () => {
      const totalPages = testingPageCount();
      if (totalPages <= 4) return Array.from({ length: totalPages }, (_, index) => index + 1);
      return [1, 2, 3, "...", totalPages];
    };
    const runTestingSearch = () => {
      appliedTestingSearch.value = testingSearch.value;
      loadTestingRuns(1);
      notifyMessage(appliedTestingSearch.value.trim() ? "Model testing search is submitted." : "Model testing search is reset.", "success");
    };
    const resetTestingModelInput = () => {
      testingModelInputForm.modelMethod = "";
      testingModelInputForm.backbone = "";
      testingModelInputForm.original = "";
      testingModelInputForm.newArea = "";
      testingModelInputForm.semanticGenerator = "";
      testingModelInputForm.embeddingModel = "";
      testingModelInputForm.optimizer = "";
      testingModelInputForm.learningRate = "";
      testingModelInputForm.batchSize = "";
      testingModelInputForm.epochs = "";
      testingModelInputForm.modelDescription = "";
    };
    const openTestingModelInput = () => {
      resetTestingModelInput();
      testingModelInputOpen.value = true;
    };
    const closeTestingModelInput = () => {
      testingModelInputOpen.value = false;
      notifyMessage("Model testing input is closed.", "info");
    };
    const handleTestingModelImport = (event) => {
      const file = event.target.files?.[0];
      if (!file) {
        notifyMessage("Model import is cancelled.", "warning");
        return;
      }
      if (!file.name.toLowerCase().match(/\.(pth|pt|onnx|json|pkl)$/)) {
        notifyMessage("Model import fails: unsupported file type.", "error");
        event.target.value = "";
        return;
      }
      notifyMessage("Model file input succeeds.", "success");
      event.target.value = "";
    };
    const testingModelTrainingPhase = (progress) => {
      if (progress < 24) return "Preparing transfer direction and backbone...";
      if (progress < 52) return "Running selected baseline training script...";
      if (progress < 82) return "Reading testing metrics and per-class accuracy...";
      return "Writing model testing record to database...";
    };
    const startTestingModelTraining = async () => {
      const requiredFields = [
        "modelMethod",
        "backbone",
        "original",
        "newArea",
        "optimizer",
        "learningRate",
        "batchSize",
        "epochs",
      ];
      if (requiredFields.some((field) => !testingModelInputForm[field])) {
        notifyMessage("Model testing input needs all required fields.", "warning");
        return;
      }
      if (testingModelTraining.active) return;
      if (testingModelTrainingTimer.value) window.clearInterval(testingModelTrainingTimer.value);
      testingModelTraining.active = true;
      testingModelTraining.progress = 0;
      testingModelTraining.phase = "Preparing model testing training...";
      testingModelTrainingTimer.value = window.setInterval(() => {
        const step = Math.floor(Math.random() * 5) + 2;
        testingModelTraining.progress = Math.min(95, testingModelTraining.progress + step);
        testingModelTraining.phase = testingModelTrainingPhase(testingModelTraining.progress);
      }, 520);
      try {
        const response = await fetch(`${API_BASE}/model-training-runs/`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            modelName: testingModelInputForm.modelMethod,
            sourceDataset: testingModelInputForm.original,
            targetDataset: testingModelInputForm.newArea,
            backbone: testingModelInputForm.backbone,
            semanticGenerator: testingModelInputForm.semanticGenerator,
            embeddingModel: testingModelInputForm.embeddingModel,
            optimizer: testingModelInputForm.optimizer,
            learningRate: testingModelInputForm.learningRate,
            batchSize: testingModelInputForm.batchSize,
            epochs: testingModelInputForm.epochs,
            modelDescription: testingModelInputForm.modelDescription,
          }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.message || "Model testing training fails.");
        if (testingModelTrainingTimer.value) window.clearInterval(testingModelTrainingTimer.value);
        testingModelTrainingTimer.value = null;
        testingModelTraining.progress = 100;
        testingModelTraining.phase = "Model testing training completes.";
        window.dispatchEvent(new CustomEvent("model-training-run-created", { detail: payload.run }));
        await loadTestingRuns(1);
        await loadDiagnosisSummary();
        window.setTimeout(() => {
          testingModelTraining.active = false;
          testingModelInputOpen.value = false;
          notifyMessage(payload.message || "Model testing training succeeds.", "success");
        }, 420);
      } catch (error) {
        if (testingModelTrainingTimer.value) window.clearInterval(testingModelTrainingTimer.value);
        testingModelTrainingTimer.value = null;
        testingModelTraining.active = false;
        notifyMessage(error.message || "Model testing training fails.", "error");
      }
    };
    const domainLabelLines = (domain) => {
      const map = {
        "a1 Sandy Loam": ["a1 Sandy", "Loam"],
        "a2 Saturated Silty Clay": ["a2 Saturated", "Silty Clay"],
        "a3 Urban Backfill Soil": ["a3 Urban", "Backfill Soil"],
        "a4 Layered Road Structure": ["a4 Layered", "Road Structure"],
      };

      return map[domain] || [domain];
    };
    const renderDomainLabel = (domain) =>
        h(
            "span",
            { class: "testing-domain" },
            domainLabelLines(domain).map((line) => h("i", line))
        );
    const handleEvaluationImageFocus = () => {
      if (!evaluationImagePending.value) return;
      if (evaluationImageTimer.value) window.clearTimeout(evaluationImageTimer.value);

      evaluationImageTimer.value = window.setTimeout(() => {
        if (!evaluationImagePending.value) return;
        evaluationImagePending.value = false;
        window.removeEventListener("focus", handleEvaluationImageFocus);
        // notifyMessage("New B-scan image input is cancelled.", "warning");
      }, 220);
    };
    const triggerEvaluationImageInput = () => {
      evaluationImagePending.value = true;
      window.addEventListener("focus", handleEvaluationImageFocus);
      evaluationImageInput.value?.click();
    };
    const handleEvaluationImageInput = async (event) => {
      const file = event.target.files?.[0];
      evaluationImagePending.value = false;
      window.removeEventListener("focus", handleEvaluationImageFocus);
      if (evaluationImageTimer.value) window.clearTimeout(evaluationImageTimer.value);

      if (!file) {
        notifyMessage("New B-scan image input is cancelled.", "warning");
        return;
      }

      if (!file.type.startsWith("image/")) {
        notifyMessage("New B-scan image input fails: unsupported file type.", "error");
        event.target.value = "";
        return;
      }

      const formData = new FormData();
      formData.append("image", file);
      try {
        const response = await fetch(`${API_BASE}/feature-image-input/`, {
          method: "POST",
          body: formData,
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.message || "New B-scan image input fails.");
        if (evaluationImageUrl.value) URL.revokeObjectURL(evaluationImageUrl.value);
        evaluationImageUrl.value = URL.createObjectURL(file);
        evaluationUploadedImage.imageId = payload.imageId || "";
        evaluationUploadedImage.imageName = payload.imageName || "";
        evaluationUploadedImage.imagePath = payload.imagePath || "";
        originalActivationReady.value = false;
        oursActivationReady.value = false;
        originalActivationImageUrl.value = "";
        oursActivationImageUrl.value = "";
        notifyMessage(payload.message || "New B-scan image input succeeds.", "success");
      } catch (error) {
        notifyMessage(error.message || "New B-scan image input fails.", "error");
      } finally {
        event.target.value = "";
      }
    };
    const comparisonGenerationPhase = (progress, method) => {
      if (progress < 28) return `Loading new B-scan image for ${method}...`;
      if (progress < 56) return `Extracting ${method} feature responses...`;
      if (progress < 84) return `Projecting ${method} activation intensity...`;
      return `Rendering ${method} activation map...`;
    };
    const selectedTestingRow = () =>
        testingRows.value.find((row) => recordKeyOf(row) === selectedTestingRecord.value) || testingRows.value[0] || null;
    const generateComparisonActivation = async (method) => {
      if (!evaluationImageUrl.value || !evaluationUploadedImage.imageId || !evaluationUploadedImage.imageName) {
        notifyMessage("Comparison activation generation needs an input image.", "warning");
        return;
      }
      const selectedRow = selectedTestingRow();
      if (!selectedRow) {
        notifyMessage("Comparison activation generation needs a selected testing record.", "warning");
        return;
      }

      if (comparisonGeneration.active) return;
      if (comparisonGenerationTimer.value) window.clearInterval(comparisonGenerationTimer.value);

      comparisonGeneration.active = true;
      comparisonGeneration.method = method;
      comparisonGeneration.progress = 0;
      comparisonGeneration.phase = "Preparing comparison activation pipeline...";

      comparisonGenerationTimer.value = window.setInterval(() => {
        const step = Math.floor(Math.random() * 7) + 4;
        comparisonGeneration.progress = Math.min(94, comparisonGeneration.progress + step);
        comparisonGeneration.phase = comparisonGenerationPhase(comparisonGeneration.progress, method);
      }, 420);

      try {
        const response = await fetch(`${API_BASE}/comparison-activation-map/`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            recordId: selectedRow.id,
            methodKind: method === "Original Classifier" ? "sourceonly" : "ours",
            sourceDataset: selectedRow.original,
            targetDataset: selectedRow.newArea,
            backbone: selectedRow.backbone,
            methodCheckpointPath: selectedRow.methodCheckpointPath,
            imageId: evaluationUploadedImage.imageId,
            imageName: evaluationUploadedImage.imageName,
          }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.message || `${method} activation map generation fails.`);
        comparisonGeneration.progress = 100;
        comparisonGeneration.phase = `${method} activation map generation completes.`;
        if (method === "Original Classifier") {
          originalActivationReady.value = true;
          originalActivationImageUrl.value = payload.activationImage || "";
        } else {
          oursActivationReady.value = true;
          oursActivationImageUrl.value = payload.activationImage || "";
        }
        notifyMessage(payload.message || `${method} activation map generation succeeds.`, "success");
      } catch (error) {
        notifyMessage(error.message || `${method} activation map generation fails.`, "error");
      } finally {
        if (comparisonGenerationTimer.value) {
          window.clearInterval(comparisonGenerationTimer.value);
          comparisonGenerationTimer.value = null;
        }
        window.setTimeout(() => {
          comparisonGeneration.active = false;
        }, 420);
      }
    };
    const exportDiagnosisSummary = () => {
      try {
        const payload = {
          module: "Diagnosis Summary",
          originalArea: normalizeDomainName(originalDomain.value),
          newArea: normalizeDomainName(newDomain.value),
          comparison: {
            methodA: buildAccuracyExport("A", selectedOriginalMethod.value),
            methodB: buildAccuracyExport("B", selectedOursMethod.value),
          },
          exportedAt: new Date().toISOString(),
        };
        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `diagnosis-summary-${normalizeDomainName(originalDomain.value)}-to-${normalizeDomainName(newDomain.value)}.json`.replace(/\s+/g, "_");
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        notifyMessage("Diagnosis summary export succeeds.", "success");
      } catch {
        notifyMessage("Diagnosis summary export fails.", "error");
      }
    };
    const deleteTestingRecord = async (row) => {
      if (!row?.id) {
        const key = recordKeyOf(row);
        testingRows.value = testingRows.value.filter((item) => recordKeyOf(item) !== key);
        notifyMessage(`${row.method} testing record is deleted.`, "success");
        return;
      }
      try {
        const response = await fetch(`${API_BASE}/model-training-runs/${row.id}/`, { method: "DELETE" });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.message || "Model testing record delete fails.");
        testingRows.value = testingRows.value.filter((item) => item.id !== row.id);
        if (testingPage.value > testingPageCount()) setTestingPage(testingPageCount());
        if (selectedTestingRecord.value === recordKeyOf(row)) {
          selectedTestingRecord.value = testingRows.value[0] ? recordKeyOf(testingRows.value[0]) : "";
        }
        notifyMessage("Model testing record is deleted.", "success");
      } catch (error) {
        notifyMessage(error.message || "Model testing record delete fails.", "error");
      }
    };
    const usesSemanticConfig = (row) =>
        row?.method === "Ours" || Boolean(row?.semanticGenerator || row?.embeddingModel || row?.knowledgeItems != null || row?.refinementIterations != null);
    const retrainDomainOptions = () => domainOptions.map((domain) => normalizeDomainName(domain));
    const handleRetrainModelImport = (event) => {
      const file = event.target.files?.[0];
      if (!file) {
        notifyMessage("Model import is cancelled.", "warning");
        return;
      }
      if (!file.name.toLowerCase().match(/\.(pth|pt|onnx|json|pkl)$/)) {
        notifyMessage("Model import fails: unsupported file type.", "error");
        event.target.value = "";
        return;
      }
      notifyMessage("Model file input succeeds.", "success");
      event.target.value = "";
    };
    const openRetrainingModal = (row) => {
      retrainingRecord.value = row;
      retrainingParams.optimizer = row.optimizer || "AdamW";
      retrainingParams.learningRate = row.learningRate || "1e-5";
      retrainingParams.batchSize = String(row.batchSize || "8");
      retrainingParams.epochs = String(row.epochs || "100");
      retrainingParams.refinementIterations = String(row.refinementIterations ?? "3");
      retrainingParams.semanticGenerator = row.semanticGenerator || "GPT-4o";
      retrainingParams.embeddingModel = row.embeddingModel || "CLIP";
      retrainingParams.knowledgeItems = String(row.knowledgeItems ?? "6");
      retrainingParams.backbone = row.backbone || "ResNet-50";
      retrainingParams.original = row.original || normalizeDomainName(originalDomain.value);
      retrainingParams.newArea = row.newArea || normalizeDomainName(newDomain.value);
      retrainingParams.modelDescription = row.description || "";
      retrainingParams.backbone = row.backbone || "ResNet-50";
      retrainingParams.original = row.original || normalizeDomainName(originalDomain.value);
      retrainingParams.newArea = row.newArea || normalizeDomainName(newDomain.value);
      retrainingParams.modelDescription = row.description || "";
    };
    const retrainingPhase = (progress) => {
      if (progress < 28) return "Loading selected testing record and backbone...";
      if (progress < 54) return "Resetting optimizer and semantic refinement parameters...";
      if (progress < 82) return "Running new-area retraining schedule...";
      return "Validating retrained testing record...";
    };
    const startRetraining = async () => {
      if (!retrainingRecord.value) {
        notifyMessage("Retraining needs a selected testing record.", "warning");
        return;
      }
      if (retrainingRun.active) return;
      if (retrainingTimer.value) window.clearInterval(retrainingTimer.value);

      retrainingRun.active = true;
      retrainingRun.progress = 0;
      retrainingRun.phase = "Preparing retraining configuration...";

      retrainingTimer.value = window.setInterval(() => {
        const step = Math.floor(Math.random() * 5) + 3;
        retrainingRun.progress = Math.min(95, retrainingRun.progress + step);
        retrainingRun.phase = retrainingPhase(retrainingRun.progress);
      }, 380);

      try {
        const sourceRecord = retrainingRecord.value;
        const response = await fetch(`${API_BASE}/model-training-runs/`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            modelName: sourceRecord.method,
            ...(usesSemanticConfig(sourceRecord) ? {
              semanticGenerator: retrainingParams.semanticGenerator,
              embeddingModel: retrainingParams.embeddingModel,
              knowledgeItems: retrainingParams.knowledgeItems,
              refinementIterations: retrainingParams.refinementIterations,
            } : {}),
            backbone: retrainingParams.backbone,
            sourceDataset: retrainingParams.original,
            targetDataset: retrainingParams.newArea,
            optimizer: retrainingParams.optimizer,
            learningRate: retrainingParams.learningRate,
            batchSize: retrainingParams.batchSize,
            epochs: retrainingParams.epochs,
            modelDescription: retrainingParams.modelDescription,
          }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.message || "Retraining fails.");
        if (retrainingTimer.value) window.clearInterval(retrainingTimer.value);
        retrainingTimer.value = null;
        retrainingRun.progress = 100;
        retrainingRun.phase = "Retraining completes and writes a new testing record.";
        await loadTestingRuns(1);
        await loadDiagnosisSummary();
        window.dispatchEvent(new CustomEvent("model-training-run-created"));
        window.setTimeout(() => {
          retrainingRun.active = false;
          retrainingRecord.value = null;
          notifyMessage(payload.message || "Model retraining succeeds.", "success");
        }, 420);
      } catch (error) {
        if (retrainingTimer.value) window.clearInterval(retrainingTimer.value);
        retrainingTimer.value = null;
        retrainingRun.active = false;
        notifyMessage(error.message || "Retraining fails.", "error");
      }
    };

    watch(
        [originalDomain, newDomain, selectedOriginalMethod, selectedOursMethod],
        () => loadDiagnosisSummary()
    );

    onMounted(() => {
      loadTestingRuns(1);
      loadDiagnosisSummary();
      window.addEventListener("model-training-run-created", handleTrainingRunCreated);
    });

    onBeforeUnmount(() => {
      window.removeEventListener("model-training-run-created", handleTrainingRunCreated);
      if (evaluationImageUrl.value) URL.revokeObjectURL(evaluationImageUrl.value);
      if (retrainingTimer.value) window.clearInterval(retrainingTimer.value);
      if (testingModelTrainingTimer.value) window.clearInterval(testingModelTrainingTimer.value);
      if (evaluationImageTimer.value) window.clearTimeout(evaluationImageTimer.value);
      if (comparisonGenerationTimer.value) window.clearInterval(comparisonGenerationTimer.value);
      window.removeEventListener("focus", handleEvaluationImageFocus);
    });

    const renderTestingRecordModal = () =>
        h(Transition, { name: "modal-fade" }, () =>
            viewedTestingRecord.value
                ? h("div", { class: "area-detail-overlay", onClick: closeTestingRecordDetail }, [
                  h("section", { class: "testing-detail-modal wide", onClick: (event) => event.stopPropagation() }, [
                    h("div", { class: "area-detail-head" }, [
                      h("div", [
                        h("span", "Testing Record Detail"),
                        h("h3", viewedTestingRecord.value.method),
                      ]),
                      h("div", { class: "modal-head-actions" }, [
                        h("button", { type: "button", class: "download-button retrain-detail-button", onClick: () => { openRetrainingModal(viewedTestingRecord.value); viewedTestingRecord.value = null; } }, "Retrain"),
                        h("button", { type: "button", onClick: closeTestingRecordDetail }, "Close"),
                      ]),
                    ]),
                    h("div", { class: "testing-detail-grid run-detail-grid" }, [
                      h("div", [h("span", "Model Method"), h("strong", viewedTestingRecord.value.method)]),
                      h("div", [h("span", "Backbone"), h("strong", viewedTestingRecord.value.backbone)]),
                      h("div", [h("span", "Original Dataset"), h("strong", viewedTestingRecord.value.original)]),
                      h("div", [h("span", "New Dataset"), h("strong", viewedTestingRecord.value.newArea)]),
                      viewedTestingRecord.value.semanticGenerator ? h("div", [h("span", "Semantic Generator"), h("strong", viewedTestingRecord.value.semanticGenerator)]) : null,
                      viewedTestingRecord.value.embeddingModel ? h("div", [h("span", "Embedding Model"), h("strong", viewedTestingRecord.value.embeddingModel)]) : null,
                      viewedTestingRecord.value.knowledgeItems != null ? h("div", [h("span", "Knowledge Items"), h("strong", `${viewedTestingRecord.value.knowledgeItems}`)]) : null,
                      viewedTestingRecord.value.refinementIterations != null ? h("div", [h("span", "Refinement Iterations"), h("strong", `${viewedTestingRecord.value.refinementIterations}`)]) : null,
                      h("div", [h("span", "Optimizer"), h("strong", viewedTestingRecord.value.optimizer || "-")]),
                      h("div", [h("span", "Learning Rate"), h("strong", viewedTestingRecord.value.learningRate || "-")]),
                      h("div", [h("span", "Batch Size"), h("strong", `${viewedTestingRecord.value.batchSize || "-"}`)]),
                      h("div", [h("span", "Epochs"), h("strong", `${viewedTestingRecord.value.epochs || "-"}`)]),
                      h("div", [h("span", "Created Time"), h("strong", formatDateTime(viewedTestingRecord.value.createdAt))]),
                      h("div", [h("span", "Updated Time"), h("strong", formatDateTime(viewedTestingRecord.value.updatedAt))]),
                      h("div", { class: "accuracy-detail-wide" }, [renderInlineAccuracyGrid(viewedTestingRecord.value)]),
                    ].filter(Boolean)),
                    h("div", { class: "testing-detail-notes" }, [
                      h("strong", "Model Description"),
                      h("p", viewedTestingRecord.value.description || "No model description is available."),
                    ]),
                  ]),
                ])
                : null
        );

    const renderRetrainingModal = () =>
        h(Transition, { name: "modal-fade" }, () =>
            retrainingRecord.value
                ? h("div", { class: "area-detail-overlay", onClick: closeRetrainingModal }, [
                  h("section", { class: "retraining-modal testing-retraining-modal", onClick: (event) => event.stopPropagation() }, [
                    h("div", { class: "area-detail-head" }, [
                      h("div", [
                        h("span", "Retraining Configuration"),
                        h("h3", retrainingRecord.value.method),
                      ]),
                      h("div", { class: "modal-head-actions" }, [
                        h("button", { type: "button", class: "download-button retrain-detail-button", onClick: () => retrainImportInput.value?.click() }, "Model Import"),
                        h("input", { ref: retrainImportInput, class: "feature-image-input", type: "file", accept: ".pth,.pt,.onnx,.json,.pkl", onChange: handleRetrainModelImport }),
                        h("button", { type: "button", onClick: closeRetrainingModal }, "Close"),
                      ]),
                    ]),
                    h("div", { class: "retraining-form full retraining-aligned-form" }, [
                      h("div", { class: "retraining-static-field" }, [
                        h("span", "Model Method"),
                        h("strong", retrainingRecord.value.method),
                      ]),
                      h("label", [
                        h("span", "Backbone"),
                        h("select", { value: retrainingParams.backbone, onChange: (event) => (retrainingParams.backbone = event.target.value) }, ["ResNet-50", "ResNet-101", "ViT-S/16"].map((item) => h("option", item))),
                      ]),
                      h("label", [
                        h("span", "Original Dataset"),
                        h("select", { value: retrainingParams.original, onChange: (event) => (retrainingParams.original = event.target.value) }, retrainDomainOptions().map((domain) => h("option", { value: domain }, domain))),
                      ]),
                      h("label", [
                        h("span", "New Dataset"),
                        h("select", { value: retrainingParams.newArea, onChange: (event) => (retrainingParams.newArea = event.target.value) }, retrainDomainOptions().map((domain) => h("option", { value: domain }, domain))),
                      ]),
                      ...(usesSemanticConfig(retrainingRecord.value) ? [
                        h("label", [
                          h("span", "Semantic Generator"),
                          h("select", { value: retrainingParams.semanticGenerator, onChange: (event) => (retrainingParams.semanticGenerator = event.target.value) }, ["GPT-4o", "GPT-3.5-turbo", "GPT-4o-mini", "Gemini-2.5", "LLaMA-3.1", "Qwen-2.5"].map((item) => h("option", item))),
                        ]),
                        h("label", [
                          h("span", "Embedding Model"),
                          h("select", { value: retrainingParams.embeddingModel, onChange: (event) => (retrainingParams.embeddingModel = event.target.value) }, ["CLIP", "SBERT", "LLaMA", "LLAMA", "Qwen"].map((item) => h("option", { value: item }, item))),
                        ]),
                        h("label", [
                          h("span", "Knowledge Items"),
                          h("input", { type: "number", min: "0", max: "10", value: retrainingParams.knowledgeItems, onInput: (event) => (retrainingParams.knowledgeItems = event.target.value) }),
                        ]),
                        h("label", [
                          h("span", "Refinement Iterations"),
                          h("input", { type: "number", min: "0", max: "6", value: retrainingParams.refinementIterations, onInput: (event) => (retrainingParams.refinementIterations = event.target.value) }),
                        ]),
                      ] : []),
                      h("label", [
                        h("span", "Optimizer"),
                        h("select", { value: retrainingParams.optimizer, onChange: (event) => (retrainingParams.optimizer = event.target.value) }, [h("option", "AdamW"), h("option", "Adam"), h("option", "SGD")]),
                      ]),
                      h("label", [
                        h("span", "Learning Rate"),
                        h("select", { value: retrainingParams.learningRate, onChange: (event) => (retrainingParams.learningRate = event.target.value) }, ["1e-6", "5e-6", "1e-5", "5e-5", "1e-4"].map((rate) => h("option", rate))),
                      ]),
                      h("label", [
                        h("span", "Batch Size"),
                        h("select", { value: retrainingParams.batchSize, onChange: (event) => (retrainingParams.batchSize = event.target.value) }, ["4", "6", "8", "10", "12", "14", "16"].map((size) => h("option", size))),
                      ]),
                      h("label", [
                        h("span", "Epochs"),
                        h("input", { type: "number", min: "50", max: "200", step: "10", value: retrainingParams.epochs, onInput: (event) => (retrainingParams.epochs = event.target.value) }),
                      ]),
                    ]),
                    h("label", { class: "retraining-description-field" }, [
                      h("span", "Model Description"),
                      h("textarea", { value: retrainingParams.modelDescription, onInput: (event) => (retrainingParams.modelDescription = event.target.value) }),
                    ]),
                    h("button", { class: "retraining-start", type: "button", onClick: startRetraining }, "Start Retraining"),
                    retrainingRun.active
                        ? h("div", { class: "retraining-progress-box" }, [
                          h("div", { class: "generation-modal-head" }, [
                            h("div", [h("span", "Retraining Progress"), h("h3", "Updating Model Parameters")]),
                            h("b", `${retrainingRun.progress}%`),
                          ]),
                          h("div", { class: "generation-progress" }, [h("i", { style: { width: `${retrainingRun.progress}%` } })]),
                          h("p", retrainingRun.phase),
                        ])
                        : null,
                  ]),
                ])
                : null
        );
    const renderTestingModelInputModal = () =>
        h(Transition, { name: "modal-fade" }, () =>
            testingModelInputOpen.value
                ? h("div", { class: "area-detail-overlay", onClick: closeTestingModelInput }, [
                  h("section", { class: "retraining-modal testing-model-input-modal", onClick: (event) => event.stopPropagation() }, [
                    h("div", { class: "area-detail-head" }, [
                      h("div", [
                        h("span", "Model Testing Input"),
                        h("h3", "Load Model"),
                      ]),
                      h("div", { class: "modal-head-actions" }, [
                        h("button", { type: "button", class: "download-button retrain-detail-button", onClick: () => testingModelImportInput.value?.click() }, "Model Import"),
                        h("input", { ref: testingModelImportInput, class: "feature-image-input", type: "file", accept: ".pth,.pt,.onnx,.json,.pkl", onChange: handleTestingModelImport }),
                        h("button", { type: "button", onClick: closeTestingModelInput }, "Close"),
                      ]),
                    ]),
                    h("div", { class: "retraining-form full retraining-aligned-form testing-model-input-form" }, [
                      h("label", { class: "required-field" }, [
                        h("span", "Model Method"),
                        h("select", { value: testingModelInputForm.modelMethod, onChange: (event) => (testingModelInputForm.modelMethod = event.target.value) }, [
                          h("option", { value: "", disabled: true, hidden: true }, "Select model"),
                          ...comparisonMethodOptions.map((item) => h("option", { value: item }, item)),
                        ]),
                      ]),
                      h("label", { class: "required-field" }, [
                        h("span", "Backbone"),
                        h("select", { value: testingModelInputForm.backbone, onChange: (event) => (testingModelInputForm.backbone = event.target.value) }, [
                          h("option", { value: "", disabled: true, hidden: true }, "Select backbone"),
                          ...["ResNet-50", "ResNet-101", "ViT-S/16"].map((item) => h("option", { value: item }, item)),
                        ]),
                      ]),
                      h("label", { class: "required-field" }, [
                        h("span", "Original Area"),
                        h("select", { value: testingModelInputForm.original, onChange: (event) => (testingModelInputForm.original = event.target.value) }, [
                          h("option", { value: "", disabled: true, hidden: true }, "Select original area"),
                          ...retrainDomainOptions().map((domain) => h("option", { value: domain }, domain)),
                        ]),
                      ]),
                      h("label", { class: "required-field" }, [
                        h("span", "New Area"),
                        h("select", { value: testingModelInputForm.newArea, onChange: (event) => (testingModelInputForm.newArea = event.target.value) }, [
                          h("option", { value: "", disabled: true, hidden: true }, "Select new area"),
                          ...retrainDomainOptions().map((domain) => h("option", { value: domain }, domain)),
                        ]),
                      ]),
                      h("label", [
                        h("span", "Semantic Generator"),
                        h("select", { value: testingModelInputForm.semanticGenerator, onChange: (event) => (testingModelInputForm.semanticGenerator = event.target.value) }, [
                          h("option", { value: "", disabled: true, hidden: true }, "Optional"),
                          ...["GPT-4o", "GPT-3.5-turbo", "GPT-4o-mini", "Gemini-2.5", "LLaMA-3.1", "Qwen-2.5"].map((item) => h("option", { value: item }, item)),
                        ]),
                      ]),
                      h("label", [
                        h("span", "Embedding Model"),
                        h("select", { value: testingModelInputForm.embeddingModel, onChange: (event) => (testingModelInputForm.embeddingModel = event.target.value) }, [
                          h("option", { value: "", disabled: true, hidden: true }, "Optional"),
                          ...["CLIP", "SBERT", "LLaMA", "Qwen"].map((item) => h("option", { value: item }, item)),
                        ]),
                      ]),
                      h("label", { class: "required-field" }, [
                        h("span", "Optimizer"),
                        h("select", { value: testingModelInputForm.optimizer, onChange: (event) => (testingModelInputForm.optimizer = event.target.value) }, [
                          h("option", { value: "", disabled: true, hidden: true }, "Select optimizer"),
                          ...["Adam", "AdamW", "SGD"].map((item) => h("option", { value: item }, item)),
                        ]),
                      ]),
                      h("label", { class: "required-field" }, [
                        h("span", "Learning Rate"),
                        h("select", { value: testingModelInputForm.learningRate, onChange: (event) => (testingModelInputForm.learningRate = event.target.value) }, [
                          h("option", { value: "", disabled: true, hidden: true }, "Select learning rate"),
                          ...["1e-6", "5e-6", "1e-5", "5e-5", "1e-4"].map((item) => h("option", { value: item }, item)),
                        ]),
                      ]),
                      h("label", { class: "required-field" }, [
                        h("span", "Batch Size"),
                        h("input", {
                          type: "number",
                          min: "1",
                          step: "1",
                          value: testingModelInputForm.batchSize,
                          placeholder: "Enter batch size",
                          onInput: (event) => (testingModelInputForm.batchSize = event.target.value),
                        }),
                      ]),
                      h("label", { class: "required-field" }, [
                        h("span", "Epochs"),
                        h("input", {
                          type: "number",
                          min: "1",
                          step: "1",
                          value: testingModelInputForm.epochs,
                          placeholder: "Enter epochs",
                          onInput: (event) => (testingModelInputForm.epochs = event.target.value),
                        }),
                      ]),
                      h("label", { class: "full-row" }, [
                        h("span", "Method Description"),
                        h("textarea", {
                          value: testingModelInputForm.modelDescription,
                          placeholder: "Describe method setting, transfer direction, training behavior, or deployment notes.",
                          onInput: (event) => (testingModelInputForm.modelDescription = event.target.value),
                        }),
                      ]),
                    ]),
                    h("div", { class: "modal-bottom-actions" }, [
                      h("button", { type: "button", class: "cancel-action", onClick: closeTestingModelInput }, "Cancel"),
                      h("button", { type: "button", class: "save-action", onClick: startTestingModelTraining }, "Start Training"),
                    ]),
                    testingModelTraining.active
                        ? h("div", { class: "retraining-progress-box testing-model-training-progress" }, [
                          h("div", { class: "generation-modal-head" }, [
                            h("div", [h("span", "Training Progress"), h("h3", "Running Model Testing Training")]),
                            h("b", `${testingModelTraining.progress}%`),
                          ]),
                          h("div", { class: "generation-progress" }, [
                            h("i", { style: { width: `${testingModelTraining.progress}%` } }),
                          ]),
                          h("p", testingModelTraining.phase),
                        ])
                        : null,
                  ]),
                ])
                : null
        );
    return () =>
        h("div", { class: "evaluation-workspace" }, [
          h("section", { class: "evaluation-card testing-record-card" }, [
            h("div", { class: "evaluation-section-head" }, [
              h("div", [
                h("h3", "Model Testing Records"),
              ]),
              h("button", { type: "button", class: "load-testing-model-button", onClick: openTestingModelInput }, "Load Model"),
              h("div", { class: "testing-search" }, [
                h("input", {
                  value: testingSearch.value,
                  placeholder: "Search model",
                  onInput: (event) => {
                    testingSearch.value = event.target.value;
                  },
                  onKeydown: (event) => {
                    if (event.key === "Enter") runTestingSearch();
                  },
                }),
                h("button", { type: "button", onClick: runTestingSearch }, "Search"),
              ]),
            ]),
            h("div", { class: "eval-table testing-table" }, [
              h("div", { class: "eval-table-head testing-head" }, [
                h("span", ""),
                h("span", "Model"),
                h("span", "Original"),
                h("span", "New"),
                h("span", "Backbone"),
                h("span", "Acc"),
                h("span", "Operations"),
              ]),
              ...pagedTestingRecords().map((row) => {
                const recordKey = recordKeyOf(row);
                return h("div", {
                  class: ["eval-row", "testing-row", row.method === "Ours" ? "best" : "", selectedTestingRecord.value === recordKey ? "selected" : ""],
                  onClick: () => {
                    selectedTestingRecord.value = recordKey;
                  },
                }, [
                  h("button", {
                    type: "button",
                    class: ["testing-check", selectedTestingRecord.value === recordKey ? "active" : ""],
                    "aria-label": `Select ${row.method}`,
                    onClick: (event) => {
                      event.stopPropagation();
                      selectedTestingRecord.value = recordKey;
                    },
                  }, [h("span")]),
                  h(
                      "strong",
                      row.method === "Original Classifier"
                          ? [h("span", "Original"), h("span", "Classifier")]
                          : row.method
                  ),
                  renderDomainLabel(row.original),
                  renderDomainLabel(row.newArea),
                  h("span", row.backbone),
                  h("b", row.acc),
                  h("div", { class: "testing-actions" }, [
                    h("button", {
                      type: "button",
                      class: "record-view",
                      onClick: (event) => {
                        event.stopPropagation();
                        viewedTestingRecord.value = row;
                      },
                    }, "View"),
                    h("button", {
                      type: "button",
                      class: "record-redesign",
                      onClick: (event) => {
                        event.stopPropagation();
                        openRetrainingModal(row);
                      },
                    }, "Retraining"),
                    h("button", {
                      type: "button",
                      class: "record-delete",
                      onClick: (event) => {
                        event.stopPropagation();
                        deleteTestingRecord(row);
                      },
                    }, "Delete"),
                  ]),
                ]);
              }),
              h("div", { class: "testing-pagination" }, [
                h(
                    "button",
                    {
                      type: "button",
                      disabled: testingPage.value === 1,
                      onClick: () => setTestingPage(testingPage.value - 1),
                    },
                    "Prev"
                ),
                ...visibleTestingPages().map((page) =>
                    page === "..."
                        ? h("span", { class: "page-ellipsis" }, "...")
                        : h(
                            "button",
                            {
                              type: "button",
                              class: testingPage.value === page ? "active" : "",
                              onClick: () => setTestingPage(page),
                            },
                            String(page)
                        )
                ),
                h(
                    "button",
                    {
                      type: "button",
                      disabled: testingPage.value === testingPageCount(),
                      onClick: () => setTestingPage(testingPage.value + 1),
                    },
                    "Next"
                ),
                h("input", {
                  class: "page-jump-input",
                  value: testingPageJump.value,
                  onInput: (event) => {
                    testingPageJump.value = event.target.value;
                  },
                  onKeydown: (event) => {
                    if (event.key === "Enter") setTestingPage(Number(testingPageJump.value));
                  },
                }),
                h(
                    "button",
                    {
                      type: "button",
                      onClick: () => setTestingPage(Number(testingPageJump.value)),
                    },
                    "Go"
                ),
              ]),
            ]),
          ]),

          h("section", { class: "evaluation-card comparison-visual-card" }, [
            h("div", { class: "comparison-visual-head" }, [
              h("h3", "Model Comparison Visualization"),
              h("div", { class: "visual-actions comparison-actions" }, [
                h("button", {
                  class: "image-upload",
                  type: "button",
                  onClick: triggerEvaluationImageInput,
                }, [
                  h("span", "Image Input"),
                ]),
                h("input", {
                  ref: evaluationImageInput,
                  class: "feature-image-input",
                  type: "file",
                  accept: "image/*",
                  onChange: handleEvaluationImageInput,
                }),
                h("button", {
                  class: "activation-button original-method",
                  type: "button",
                  onClick: () => generateComparisonActivation("Original Classifier"),
                }, "Original Classifier"),
                h("button", {
                  class: "activation-button ours",
                  type: "button",
                  onClick: () => generateComparisonActivation("Ours"),
                }, "Ours"),
              ]),
            ]),
            h("div", { class: "comparison-frames" }, [
              h("div", { class: "scan-panel" }, [
                h("div", { class: "scan-label" }, "New B-scan Image"),
                h(
                    "div",
                    { class: ["square-frame", "new-scan-frame", evaluationImageUrl.value ? "has-image" : ""] },
                    evaluationImageUrl.value
                        ? [h("img", { src: evaluationImageUrl.value, alt: "New B-scan image" })]
                        : []
                ),
              ]),
              h("div", { class: "scan-panel" }, [
                h("div", { class: "scan-label" }, "Original Classifier Activation Map"),
                h("div", {
                  class: [
                    "square-frame",
                    "original-activation-frame",
                    originalActivationReady.value ? "has-activation" : "",
                  ],
                }, originalActivationReady.value && originalActivationImageUrl.value ? [
                  h("img", { src: originalActivationImageUrl.value, alt: "Original Classifier activation map" }),
                ] : []),
              ]),
              h("div", { class: "scan-panel" }, [
                h("div", { class: "scan-label" }, "Ours Activation Map"),
                h("div", {
                  class: [
                    "square-frame",
                    "ours-activation-frame",
                    oursActivationReady.value ? "has-activation" : "",
                  ],
                }, oursActivationReady.value && oursActivationImageUrl.value ? [
                  h("img", { src: oursActivationImageUrl.value, alt: "Ours activation map" }),
                ] : []),
              ]),
            ]),
            h(Transition, { name: "modal-fade" }, () =>
                comparisonGeneration.active
                    ? h("div", { class: "activation-generation-overlay comparison-generation-overlay" }, [
                      h("section", { class: "activation-generation-modal" }, [
                        h("div", { class: "generation-modal-head" }, [
                          h("div", [
                            h("span", "Model Comparison Visualization"),
                            h("h3", `Generating ${comparisonGeneration.method} Map`),
                          ]),
                          h("b", `${comparisonGeneration.progress}%`),
                        ]),
                        h("div", { class: "generation-progress" }, [
                          h("i", { style: { width: `${comparisonGeneration.progress}%` } }),
                        ]),
                        h("p", comparisonGeneration.phase),
                      ]),
                    ])
                    : null
            ),
          ]),

          h("section", { class: "evaluation-card diagnosis-card" }, [
            h("div", { class: "evaluation-section-head compact" }, [
              h("div", [
                h("h3", "Diagnosis Summary"),
              ]),
              h("button", {
                type: "button",
                class: "summary-export",
                onClick: exportDiagnosisSummary,
              }, "Export"),
            ]),
            h("div", { class: "diagnosis-strip" }, [
              h("label", { class: "diagnosis-select-field" }, [
                h("span", "Original Area"),
                h(
                    "select",
                    {
                      value: normalizeDomainName(originalDomain.value),
                      onChange: (event) => {
                        originalDomain.value = event.target.value;
                        if (event.target.value === "GPR-SD") newDomain.value = "GPR-Road";
                        if (event.target.value === "GPR-Road") newDomain.value = "GPR-SD";
                      },
                    },
                    domainOptions.map((domain) => h("option", { value: normalizeDomainName(domain) }, normalizeDomainName(domain)))
                ),
              ]),
              h("label", { class: "diagnosis-select-field" }, [
                h("span", "New Area"),
                h(
                    "select",
                    {
                      value: normalizeDomainName(newDomain.value),
                      onChange: (event) => {
                        newDomain.value = event.target.value;
                      },
                    },
                    domainOptions.map((domain) => h("option", { value: normalizeDomainName(domain) }, normalizeDomainName(domain)))
                ),
              ]),
              h("label", { class: "diagnosis-method-field" }, [
                h("span", "Method A"),
                h(
                    "select",
                    {
                      value: selectedOriginalMethod.value,
                      onChange: (event) => {
                        selectedOriginalMethod.value = event.target.value;
                      },
                    },
                    comparisonMethodOptions.map((method) => h("option", { value: method }, method))
                ),
              ]),
              h("label", { class: "diagnosis-method-field success" }, [
                h("span", "Method B"),
                h(
                    "select",
                    {
                      value: selectedOursMethod.value,
                      onChange: (event) => {
                        selectedOursMethod.value = event.target.value;
                      },
                    },
                    comparisonMethodOptions.map((method) => h("option", { value: method }, method))
                ),
              ]),
            ]),
            h("div", { class: "class-accuracy-compare" }, [
              h("div", { class: "class-accuracy-card original-method" }, [
                h("h4", `${selectedOriginalMethod.value} - Accuracy`),
                h("div", { class: "class-accuracy-list" }, renderAccuracyRows("A")),
              ]),
              h("div", { class: "class-accuracy-card ours-method" }, [
                h("h4", `${selectedOursMethod.value} - Accuracy`),
                h("div", { class: "class-accuracy-list" }, renderAccuracyRows("B")),
              ]),
            ]),
          ]),
          renderTestingRecordModal(),
          renderRetrainingModal(),
          renderTestingModelInputModal(),
        ]);
  },
});
</script>















