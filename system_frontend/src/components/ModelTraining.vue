<template>
  <section class="system-panel reserved-panel">
    <PanelHeader
      code="MT"
      title="Model Selection and Training"
      subtitle="Feature extractor selection, activation inspection, and model training workflow."
    />
    <TrainingPanel />
  </section>
</template>

<script setup>
import { Transition, defineComponent, h, onBeforeUnmount, onMounted, reactive, ref } from "vue";
import { ElMessage } from "element-plus";
import PanelHeader from "./PanelHeader.vue";
import {
  domainOptions,
  originalDomain,
  newDomain,
  classifierModels,
} from "../shared/gprState";

const API_BASE = process.env.VUE_APP_API_BASE_URL || "http://127.0.0.1:8000/api";

const TrainingPanel = defineComponent({
  setup() {
    const selectedClassifier = ref(null);
    const classifierRows = ref([]);
    const classifierLoading = ref(true);
    const viewedClassifier = ref(null);
    const retrainTimer = ref(null);
    const retrainState = reactive({
      active: false,
      progress: 0,
      phase: "Preparing classifier retraining...",
      classifierId: null,
    });
    const currentPage = ref(1);
    const pageJump = ref("1");
     const sourceImageUrl = ref("");
     const activationReady = ref(false);
      const activationImageUrl = ref("");
      const uploadedFeatureImage = reactive({
        imageId: "",
        imageName: "",
        imagePath: "",
      });
     const imageFileInput = ref(null);
    const imageUploadPending = ref(false);
    const imageUploadTimer = ref(null);
    const activationGenerationTimer = ref(null);
    const activationGeneration = reactive({
      active: false,
      progress: 0,
      phase: "Preparing activation pipeline...",
    });
    const modelTrainingTimer = ref(null);
    const modelTraining = reactive({
      active: false,
      progress: 0,
      phase: "Preparing classifier and semantic inputs...",
    });
    const classifierFileInput = ref(null);
    const classifierUploadPending = ref(false);
    const classifierUploadTimer = ref(null);
    const classifierInputOpen = ref(false);
    const classifierSaving = ref(false);
    const classifierSaveTimer = ref(null);
    const classifierSaveProgress = reactive({
      active: false,
      progress: 0,
      phase: "Preparing original classifier training...",
    });
    const classifierDomainOptions = ref([]);
    const classifierForm = reactive({
      modelName: "",
      domainName: "",
      trainingType: "",
      modelDescription: "",
    });
    const trainingForm = reactive({
      semanticGenerator: "GPT-4o",
      embeddingModel: "CLIP",
      optimizer: "Adam",
    });
    const classifierNameOptions = classifierModels.map((model) => model.name);
    const domainNameMap = {
      a1: "a1 Sandy Loam",
      a2: "a2 Saturated Silty Clay",
      a3: "a3 Urban Backfill Soil",
      a4: "a4 Layered Road Structure",
      "GPR-SD": "GPR-SD",
      "GPR-Road": "GPR-Road",
    };
    const normalizeDomainName = (value) => {
      if (!value) return "";
      if (typeof value === "object") return value.name || value.code || "";
      return domainNameMap[value] || String(value);
    };
    const backendDomainNames = () => classifierDomainOptions.value.map((domain) => normalizeDomainName(domain)).filter(Boolean);
    const trainingDomainOptions = () => {
      const panelOrder = domainOptions
          .map(normalizeDomainName)
          .filter(Boolean);

      const backendNames = backendDomainNames();

      return Array.from(
          new Set([
            ...panelOrder,
            ...backendNames,
          ])
      );
    };
    const trainingParams = reactive({
      knowledgeEntries: { label: "Knowledge Items", value: 6, min: 1, max: 10, step: 1, unit: "per class", editableRatio: true },
      refinementIterations: { label: "Refinement Iterations", value: 3, min: 0, max: 6, step: 1, unit: "", editableRatio: true },
      learningRate: { label: "Learning Rate", value: 3, min: 1, max: 5, step: 1, options: ["1e-6", "5e-6", "1e-5", "5e-5", "1e-4"], editableOption: true, maxText: "1e-4" },
      batchSize: { label: "Batch Size", value: 3, min: 1, max: 7, step: 1, options: [4, 6, 8, 10, 12, 14, 16], editableOptionRatio: true, maxText: "16" },
      epochs: { label: "Epochs", value: 100, min: 50, max: 200, step: 10, unit: "", editableRatio: true },
    });
    const formatDateTime = (value) => {
      if (!value) return "-";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return String(value);
      return date.toLocaleString();
    };
     const normalizeClassifier = (row) => ({
       ...row,
       name: row.name || row.modelName,
       source: row.source || row.original,
       sourceCode: row.sourceCode || row.source || row.original,
       type: row.type || (row.trainingType === "fine_tuned" ? "Fine-tuned" : "Pre-trained"),
       acc: row.acc || `${Number(row.accuracy || 0).toFixed(1)}%`,
        modelFilePath: row.modelFilePath || row.model_file_path || "",
     });
    const loadClassifiers = async () => {
      classifierLoading.value = true;
      try {
        const response = await fetch(`${API_BASE}/original-classifiers/?page=1&pageSize=200`);
        if (!response.ok) throw new Error("Classifier records loading fails.");
        const payload = await response.json();
        const rows = (payload.classifiers || []).map(normalizeClassifier);
        classifierRows.value = rows;
        if (rows.length) {
          const current = selectedClassifier.value;
          selectedClassifier.value = rows.find((row) => current && row.id === current.id) || rows[0];
        }
        setPage(Math.min(currentPage.value, Math.max(1, Math.ceil(rows.length / pageSize))));
      } catch (error) {
        notifyMessage(error.message || "Classifier records loading fails.", "error");
      } finally {
        classifierLoading.value = false;
      }
    };
    const loadClassifierDomains = async () => {
      try {
        const response = await fetch(`${API_BASE}/domains/`);
        if (!response.ok) throw new Error("Dataset options loading fails.");
        const payload = await response.json();
        classifierDomainOptions.value = payload.domains || [];
      } catch (error) {
        notifyMessage(error.message || "Dataset options loading fails.", "error");
      }
    };
    const resetClassifierForm = () => {
      classifierForm.modelName = "";
      classifierForm.domainName = "";
      classifierForm.trainingType = "";
      classifierForm.modelDescription = "";
    };
    const openClassifierInput = () => {
      resetClassifierForm();
      classifierInputOpen.value = true;
    };
    const triggerClassifierModelImport = () => {
      classifierUploadPending.value = true;
      window.addEventListener("focus", handleClassifierUploadFocus);
      classifierFileInput.value?.click();
    };
    const classifierSavePhase = (progress) => {
      if (progress < 20) return "Preparing classifier configuration...";
      if (progress < 52) return "Running original-area classifier training script...";
      if (progress < 82) return "Reading generated classifier metrics...";
      return "Writing classifier record to database...";
    };
    const saveClassifierForm = async () => {
      if (!classifierForm.modelName || !classifierForm.domainName || !classifierForm.trainingType) {
        notifyMessage("Classifier save needs model, dataset, and type.", "warning");
        return;
      }
      classifierSaving.value = true;
      classifierSaveProgress.active = true;
      classifierSaveProgress.progress = 0;
      classifierSaveProgress.phase = "Preparing original classifier training...";
      if (classifierSaveTimer.value) window.clearInterval(classifierSaveTimer.value);
      classifierSaveTimer.value = window.setInterval(() => {
        const step = Math.floor(Math.random() * 7) + 3;
        classifierSaveProgress.progress = Math.min(92, classifierSaveProgress.progress + step);
        classifierSaveProgress.phase = classifierSavePhase(classifierSaveProgress.progress);
      }, 2000);

      try {
        const response = await fetch(`${API_BASE}/original-classifiers/`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            modelName: classifierForm.modelName,
            domainName: classifierForm.domainName,
            trainingType: classifierForm.trainingType,
            modelDescription: classifierForm.modelDescription,
          }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.message || "Classifier save fails.");
        const created = normalizeClassifier(payload.classifier);
        classifierRows.value = [created, ...classifierRows.value];
        selectedClassifier.value = created;
        setPage(1);
        if (classifierSaveTimer.value) window.clearInterval(classifierSaveTimer.value);
        classifierSaveTimer.value = null;
        classifierSaveProgress.progress = 100;
        classifierSaveProgress.phase = "Classifier training result is saved.";
        await new Promise((resolve) => window.setTimeout(resolve, 420));
        classifierInputOpen.value = false;
        classifierSaveProgress.active = false;
        notifyMessage(payload.message || "Classifier training succeeds.", "success");
      } catch (error) {
        if (classifierSaveTimer.value) window.clearInterval(classifierSaveTimer.value);
        classifierSaveTimer.value = null;
        classifierSaveProgress.active = false;
        notifyMessage(error.message || "Classifier training fails.", "error");
      } finally {
        classifierSaving.value = false;
      }
    };
    const retrainPhase = (progress) => {
      if (progress < 30) return "Preparing original classifier weights...";
      if (progress < 62) return "Re-optimizing classifier head on the original domain...";
      if (progress < 88) return "Validating updated classifier accuracy...";
      return "Writing retraining result to database...";
    };
    const applyClassifierUpdate = (nextRow) => {
      const normalized = normalizeClassifier(nextRow);
      classifierRows.value = classifierRows.value.map((row) => row.id === normalized.id ? normalized : row);
      if (selectedClassifier.value?.id === normalized.id) selectedClassifier.value = normalized;
      if (viewedClassifier.value?.id === normalized.id) viewedClassifier.value = normalized;
      return normalized;
    };
    const retrainClassifier = async () => {
      const row = viewedClassifier.value;
      if (!row?.id || retrainState.active) return;
      if (retrainTimer.value) window.clearInterval(retrainTimer.value);

      retrainState.active = true;
      retrainState.progress = 0;
      retrainState.phase = "Preparing classifier retraining...";
      retrainState.classifierId = row.id;

      retrainTimer.value = window.setInterval(() => {
        const step = Math.floor(Math.random() * 9) + 7;
        retrainState.progress = Math.min(92, retrainState.progress + step);
        retrainState.phase = retrainPhase(retrainState.progress);
      }, 2000);

      try {
        await new Promise((resolve) => window.setTimeout(resolve, 1800));
        const response = await fetch(`${API_BASE}/original-classifiers/${row.id}/retrain/`, { method: "POST" });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.message || "Classifier retraining fails.");
        if (retrainTimer.value) window.clearInterval(retrainTimer.value);
        retrainTimer.value = null;
        retrainState.progress = 100;
        retrainState.phase = "Classifier retraining result is updated.";
        applyClassifierUpdate(payload.classifier);
        window.setTimeout(() => {
          retrainState.active = false;
          retrainState.classifierId = null;
          viewedClassifier.value = null;
          notifyMessage("Classifier retraining succeeds.", "success");
        }, 420);
      } catch (error) {
        if (retrainTimer.value) window.clearInterval(retrainTimer.value);
        retrainTimer.value = null;
        retrainState.active = false;
        retrainState.classifierId = null;
        notifyMessage(error.message || "Classifier retraining fails.", "error");
      }
    };
    const pageSize = 4;
    const pageCount = () => Math.max(1, Math.ceil(classifierRows.value.length / pageSize));
    const pagedRows = () =>
      classifierRows.value.slice((currentPage.value - 1) * pageSize, currentPage.value * pageSize);
    const sameClassifier = (left, right) =>
      left && right && ((left.id && right.id && left.id === right.id) || (left.name === right.name && (left.sourceCode || left.source) === (right.sourceCode || right.source) && left.type === right.type));
    const setPage = (page) => {
      currentPage.value = Math.min(pageCount(), Math.max(1, page));
      pageJump.value = String(currentPage.value);
    };
    const selectClassifier = (row) => {
      selectedClassifier.value = row;
      const sourceName = normalizeDomainName(row.source || row.sourceCode || row.original);
      if (sourceName) originalDomain.value = sourceName;
    };
    const syncClassifierRecord = (modelName, sourceName) => {
      const matched =
        classifierRows.value.find(
          (row) =>
            row.name === modelName &&
            (row.sourceCode || row.source) === sourceName &&
            row.type === selectedClassifier.value?.type
        ) ||
        classifierRows.value.find((row) => row.name === modelName && (row.sourceCode || row.source) === sourceName) ||
        classifierRows.value.find((row) => row.name === modelName);

      if (matched) selectedClassifier.value = matched;
    };
    const deleteClassifier = async (row) => {
      if (!row?.id) {
        classifierRows.value = classifierRows.value.filter((item) => !sameClassifier(item, row));
        notifyMessage(`${row.name} classifier record is deleted.`, "success");
        return;
      }

      try {
        const response = await fetch(`${API_BASE}/original-classifiers/${row.id}/`, { method: "DELETE" });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.message || "Classifier delete fails.");
        classifierRows.value = classifierRows.value.filter((item) => item.id !== row.id);
        if (currentPage.value > pageCount()) setPage(pageCount());
        if (selectedClassifier.value?.id === row.id) selectedClassifier.value = classifierRows.value[0] || null;
        if (viewedClassifier.value?.id === row.id) viewedClassifier.value = null;
        notifyMessage("Classifier is deleted.", "success");
      } catch (error) {
        notifyMessage(error.message || "Classifier delete fails.", "error");
      }
    };
    const handleClassifierModelChange = (event) => {
      syncClassifierRecord(event.target.value, originalDomain.value);
    };
    const handleOriginalAreaChange = (event) => {
      const nextArea = normalizeDomainName(event.target.value);
      originalDomain.value = nextArea;
      if (selectedClassifier.value?.name) syncClassifierRecord(selectedClassifier.value.name, nextArea);
    };
    const handleNewAreaChange = (event) => {
      newDomain.value = normalizeDomainName(event.target.value);
    };
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
    const closeClassifierInput = () => {
      classifierInputOpen.value = false;
      notifyMessage("Classifier input is closed.", "info");
    };
    const closeClassifierDetail = () => {
      viewedClassifier.value = null;
      // notifyMessage("Classifier detail is closed.", "info");
    };
    const handleImageUploadFocus = () => {
      if (!imageUploadPending.value) return;
      if (imageUploadTimer.value) window.clearTimeout(imageUploadTimer.value);

      imageUploadTimer.value = window.setTimeout(() => {
        if (!imageUploadPending.value) return;
        imageUploadPending.value = false;
        window.removeEventListener("focus", handleImageUploadFocus);
        notifyMessage("B-scan image input is cancelled.", "warning");
      }, 220);
    };
    const triggerImageInput = () => {
      imageUploadPending.value = true;
      window.addEventListener("focus", handleImageUploadFocus);
      imageFileInput.value?.click();
    };
    const handleImageInput = async (event) => {
      const file = event.target.files?.[0];
      imageUploadPending.value = false;
      window.removeEventListener("focus", handleImageUploadFocus);
      if (imageUploadTimer.value) window.clearTimeout(imageUploadTimer.value);

      if (!file) {
        notifyMessage("B-scan image input is cancelled.", "warning");
        return;
      }

      if (!file.type.startsWith("image/")) {
        notifyMessage("B-scan image input fails: unsupported file type.", "error");
        event.target.value = "";
        return;
      }

      try {
        const formData = new FormData();
        formData.append("image", file);
        const response = await fetch(`${API_BASE}/feature-image-input/`, {
          method: "POST",
          body: formData,
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.message || "B-scan image input fails.");

        if (sourceImageUrl.value) URL.revokeObjectURL(sourceImageUrl.value);
        sourceImageUrl.value = URL.createObjectURL(file);
        activationImageUrl.value = "";
        activationReady.value = false;
        uploadedFeatureImage.imageId = payload.imageId || "";
        uploadedFeatureImage.imageName = payload.imageName || "";
        uploadedFeatureImage.imagePath = payload.imagePath || "";
        notifyMessage(payload.message || "B-scan image input succeeds.", "success");
      } catch (error) {
        notifyMessage(error.message || "B-scan image input fails.", "error");
      } finally {
        event.target.value = "";
      }
    };
    const handleClassifierUploadFocus = () => {
      if (!classifierUploadPending.value) return;
      if (classifierUploadTimer.value) window.clearTimeout(classifierUploadTimer.value);
      if (classifierSaveTimer.value) window.clearInterval(classifierSaveTimer.value);

      classifierUploadTimer.value = window.setTimeout(() => {
        if (!classifierUploadPending.value) return;
        classifierUploadPending.value = false;
        window.removeEventListener("focus", handleClassifierUploadFocus);
        notifyMessage("Classifier upload is cancelled.", "warning");
      }, 220);
    };
    const triggerClassifierUpload = () => {
      openClassifierInput();
    };
    const handleClassifierUpload = (event) => {
      const file = event.target.files?.[0];
      classifierUploadPending.value = false;
      window.removeEventListener("focus", handleClassifierUploadFocus);
      if (classifierUploadTimer.value) window.clearTimeout(classifierUploadTimer.value);
      if (classifierSaveTimer.value) window.clearInterval(classifierSaveTimer.value);

      if (!file) {
        notifyMessage("Classifier upload is cancelled.", "warning");
        return;
      }

      const validExtensions = [".pt", ".pth", ".pkl", ".ckpt", ".onnx", ".json"];
      const lowerName = file.name.toLowerCase();
      const isValid = validExtensions.some((extension) => lowerName.endsWith(extension));

      if (!isValid) {
        notifyMessage("Classifier upload fails: unsupported file type.", "error");
        event.target.value = "";
        return;
      }

      notifyMessage("Classifier upload succeeds.", "success");
      event.target.value = "";
    };
    onMounted(() => {
      loadClassifiers();
      loadClassifierDomains();
    });
    onBeforeUnmount(() => {
      if (classifierUploadTimer.value) window.clearTimeout(classifierUploadTimer.value);
      if (classifierSaveTimer.value) window.clearInterval(classifierSaveTimer.value);
      window.removeEventListener("focus", handleClassifierUploadFocus);
      if (imageUploadTimer.value) window.clearTimeout(imageUploadTimer.value);
      window.removeEventListener("focus", handleImageUploadFocus);
      if (activationGenerationTimer.value) window.clearInterval(activationGenerationTimer.value);
      if (modelTrainingTimer.value) window.clearInterval(modelTrainingTimer.value);
      if (retrainTimer.value) window.clearInterval(retrainTimer.value);
      if (sourceImageUrl.value) URL.revokeObjectURL(sourceImageUrl.value);
    });
    const activationPhase = (progress) => {
      if (progress < 28) return "Loading original B-scan image...";
      if (progress < 56) return "Extracting feature responses from selected classifier...";
      if (progress < 84) return "Projecting activation intensity map...";
      return "Rendering activation output...";
    };
    const generateActivationMap = async () => {
      if (!sourceImageUrl.value || !uploadedFeatureImage.imageId) {
        notifyMessage("Activation generation needs an input image.", "warning");
        return;
      }
      if (!selectedClassifier.value?.id) {
        notifyMessage("Activation generation needs a selected classifier.", "warning");
        return;
      }

      if (activationGeneration.active) return;
      if (activationGenerationTimer.value) window.clearInterval(activationGenerationTimer.value);

      activationReady.value = false;
      activationImageUrl.value = "";
      activationGeneration.active = true;
      activationGeneration.progress = 0;
      activationGeneration.phase = "Preparing activation pipeline...";

      activationGenerationTimer.value = window.setInterval(() => {
        const step = Math.floor(Math.random() * 7) + 3;
        activationGeneration.progress = Math.min(95, activationGeneration.progress + step);
        activationGeneration.phase = activationPhase(activationGeneration.progress);
      }, 520);

      try {
        const response = await fetch(`${API_BASE}/feature-activation-map/`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            classifierId: selectedClassifier.value.id,
            imageId: uploadedFeatureImage.imageId,
            imageName: uploadedFeatureImage.imageName,
          }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.message || "Activation map generation fails.");

        if (activationGenerationTimer.value) window.clearInterval(activationGenerationTimer.value);
        activationGenerationTimer.value = null;
        activationGeneration.progress = 100;
        activationGeneration.phase = "Activation map generation completes.";
        activationImageUrl.value = payload.activationImage || "";
        activationReady.value = Boolean(payload.activationImage);
        window.setTimeout(() => {
          activationGeneration.active = false;
          notifyMessage(payload.message || "Activation map generation succeeds.", "success");
        }, 420);
      } catch (error) {
        if (activationGenerationTimer.value) window.clearInterval(activationGenerationTimer.value);
        activationGenerationTimer.value = null;
        activationGeneration.active = false;
        notifyMessage(error.message || "Activation map generation fails.", "error");
      }
    };
    const trainingPhase = (progress) => {
      if (progress < 24) return "Preparing classifier and semantic descriptions...";
      if (progress < 48) return "Building semantic embedding supervision...";
      if (progress < 76) return "Optimizing classifier with cross-area constraints...";
      return "Validating target-area training response...";
    };
    const startModelTraining = async () => {
      if (!selectedClassifier.value) {
        notifyMessage("Model training needs a selected classifier.", "warning");
        return;
      }

      if (modelTraining.active) return;
      if (modelTrainingTimer.value) window.clearInterval(modelTrainingTimer.value);
      if (retrainTimer.value) window.clearInterval(retrainTimer.value);

      modelTraining.active = true;
      modelTraining.progress = 0;
      modelTraining.phase = "Preparing classifier and semantic inputs...";

      modelTrainingTimer.value = window.setInterval(() => {
        const step = Math.floor(Math.random() * 5) + 2;
        modelTraining.progress = Math.min(95, modelTraining.progress + step);
        modelTraining.phase = trainingPhase(modelTraining.progress);
      }, 4000);

      try {
        const response = await fetch(`${API_BASE}/model-training-runs/`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            classifierId: selectedClassifier.value.id,
            modelName: "Ours",
            sourceDataset: originalDomain.value,
            targetDataset: newDomain.value,
            backbone: selectedClassifier.value.name,
            semanticGenerator: trainingForm.semanticGenerator,
            embeddingModel: trainingForm.embeddingModel,
            optimizer: trainingForm.optimizer,
            knowledgeItems: trainingParams.knowledgeEntries.value,
            refinementIterations: trainingParams.refinementIterations.value,
            learningRate: trainingParams.learningRate.options[trainingParams.learningRate.value - 1],
            batchSize: trainingParams.batchSize.options[trainingParams.batchSize.value - 1],
            epochs: trainingParams.epochs.value,
            modelDescription: "Our method enhances cross-area classification by combining original-area visual features with semantic descriptions of both areas. Semantic guidance helps align class representations across different area conditions, improving robustness when the new domain has limited or no labels.",
          }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.message || "Model training save fails.");
        if (modelTrainingTimer.value) window.clearInterval(modelTrainingTimer.value);
        modelTrainingTimer.value = null;
        modelTraining.progress = 100;
        modelTraining.phase = "Model training completes.";
        window.dispatchEvent(new CustomEvent("model-training-run-created", { detail: payload.run }));
        window.setTimeout(() => {
          modelTraining.active = false;
          notifyMessage(payload.message || "Model training succeeds.", "success");
        }, 420);
      } catch (error) {
        if (modelTrainingTimer.value) window.clearInterval(modelTrainingTimer.value);
        modelTrainingTimer.value = null;
        modelTraining.active = false;
        notifyMessage(error.message || "Model training save fails.", "error");
      }
    };
    const paramValueText = (param) => {
      if (param.editableRatio) return param.unit ? `${param.value}/ ${param.max} ${param.unit}` : `${param.value}/ ${param.max}`;
      if (param.editableOptionRatio) return `${param.options[param.value - 1]}/ ${param.maxText}`;
      if (param.options) return param.options[param.value - 1];
      return param.unit ? `${param.value} ${param.unit}` : `${param.value}`;
    };
    const setParamValue = (param, value) => {
      const numericValue = Number(value);
      if (!Number.isFinite(numericValue)) return;
      param.value = Math.min(param.max, Math.max(param.min, numericValue));
    };
    const setOptionParamValue = (param, value) => {
      const optionIndex = param.options.map(String).indexOf(String(value));
      if (optionIndex !== -1) {
        param.value = optionIndex + 1;
      }
    };
    const renderTrainingSlider = (param) =>
      h("div", { class: "train-slider-row" }, [
        h("div", [
          h("span", param.label),
          param.editableOption
            ? h("div", { class: "knowledge-value rate-value" }, [
                h("input", {
                  value: paramValueText(param),
                  onInput: (event) => {
                    setOptionParamValue(param, event.target.value);
                  },
                }),
                h("b", `/ ${param.maxText}`),
              ])
            : param.editableOptionRatio
            ? h("div", { class: "knowledge-value" }, [
                h("input", {
                  value: param.options[param.value - 1],
                  onInput: (event) => {
                    setOptionParamValue(param, event.target.value);
                  },
                }),
                h("b", `/ ${param.maxText}`),
              ])
            : param.editableRatio
            ? h("div", { class: "knowledge-value" }, [
                h("input", {
                  type: "number",
                  min: param.min,
                  max: param.max,
                  value: param.value,
                  onInput: (event) => {
                    setParamValue(param, event.target.value);
                  },
                }),
                h("b", param.unit ? `/ ${param.max} ${param.unit}` : `/ ${param.max}`),
              ])
            : h("b", paramValueText(param)),
        ]),
        h("input", {
          type: "range",
          min: param.min,
          max: param.max,
          step: param.step,
          value: param.value,
          style: {
            "--slider-percent": `${((param.value - param.min) / (param.max - param.min)) * 100}%`,
          },
          onInput: (event) => {
            param.value = Number(event.target.value);
          },
        }),
      ]);
    const renderCompactTrainingSlider = (param) =>
      h("div", { class: "train-slider-row compact" }, [
        h("div", [
          h("span", param.label),
          param.editableOptionRatio
            ? h("div", { class: "knowledge-value" }, [
                h("input", {
                  value: param.options[param.value - 1],
                  onInput: (event) => {
                    setOptionParamValue(param, event.target.value);
                  },
                }),
                h("b", `/ ${param.maxText}`),
              ])
            : param.editableRatio
            ? h("div", { class: "knowledge-value" }, [
                h("input", {
                  type: "number",
                  min: param.min,
                  max: param.max,
                  value: param.value,
                  onInput: (event) => {
                    setParamValue(param, event.target.value);
                  },
                }),
                h("b", param.unit ? `/ ${param.max} ${param.unit}` : `/ ${param.max}`),
              ])
            : h("b", paramValueText(param)),
        ]),
        h("input", {
          type: "range",
          min: param.min,
          max: param.max,
          step: param.step,
          value: param.value,
          style: {
            "--slider-percent": `${((param.value - param.min) / (param.max - param.min)) * 100}%`,
          },
          onInput: (event) => {
            param.value = Number(event.target.value);
          },
        }),
      ]);
    const visiblePages = () => {
      const totalPages = pageCount();
      const firstPages = Array.from({ length: Math.min(3, totalPages) }, (_, index) => index + 1);
      return totalPages > 4 ? [...firstPages, "ellipsis", totalPages] : firstPages;
    };
    const renderClassifierInputModal = () =>
      h(Transition, { name: "modal-fade" }, () =>
        classifierInputOpen.value
          ? h("div", { class: "area-input-overlay", onClick: closeClassifierInput }, [
            h("section", { class: "classifier-input-modal", onClick: (event) => event.stopPropagation() }, [
              h("div", { class: "area-input-head" }, [
                h("div", [
                  h("span", "Original Classifier Input"),
                  h("h3", "Load Classifier"),
                ]),
                h("div", { class: "modal-head-actions" }, [
                  h("button", {
                    type: "button",
                    class: "import-button",
                    onClick: triggerClassifierModelImport,
                  }, "Model Import"),
                  h("input", {
                    ref: classifierFileInput,
                    class: "classifier-upload-input",
                    type: "file",
                    accept: ".pt,.pth,.pkl,.ckpt,.onnx,.json",
                    onChange: handleClassifierUpload,
                  }),
                  h("button", { type: "button", onClick: closeClassifierInput }, "Close"),
                ]),
              ]),
              h("div", { class: "classifier-input-form" }, [
                h("label", { class: "required-field" }, [
                  h("span", [h("i", { class: "required-star" }, "*"), h("b", "Model")]),
                  h("select", {
                    value: classifierForm.modelName,
                    class: classifierForm.modelName ? "" : "is-placeholder",
                    onChange: (event) => (classifierForm.modelName = event.target.value),
                  }, [
                    h("option", { value: "" , disabled: true, hidden: true}, "Select model"),
                    ...classifierNameOptions.map((name) => h("option", { value: name }, name)),
                  ]),
                ]),
                h("label", { class: "required-field" }, [
                  h("span", [h("i", { class: "required-star" }, "*"), h("b", "Dataset")]),
                  h("select", {
                    value: classifierForm.domainName,
                    class: classifierForm.domainName ? "" : "is-placeholder",
                    onChange: (event) => (classifierForm.domainName = event.target.value),
                  }, [
                    h("option", { value: "", disabled: true, hidden: true}, "Select dataset"),
                    ...classifierDomainOptions.value.map((domain) => h("option", { value: domain.name || domain.code }, domain.name || domain.code)),
                  ]),
                ]),
                h("label", { class: "required-field" }, [
                  h("span", [h("i", { class: "required-star" }, "*"), h("b", "Type")]),
                  h("select", {
                    value: classifierForm.trainingType,
                    class: classifierForm.trainingType ? "" : "is-placeholder",
                    onChange: (event) => (classifierForm.trainingType = event.target.value),
                  }, [
                    h("option", { value: "", disabled: true, hidden: true }, "Select type"),
                    h("option", { value: "fine_tuned" }, "Fine-tuned"),
                    h("option", { value: "pre_trained" }, "Pre-trained"),
                  ]),
                ]),
                h("label", { class: "description-field" }, [
                  h("span", [h("b", "Model Description")]),
                  h("textarea", {
                    value: classifierForm.modelDescription,
                    placeholder: "Describe backbone initialization, training area, validation behavior, or deployment notes.",
                    onInput: (event) => (classifierForm.modelDescription = event.target.value),
                  }),
                ]),
              ]),
              classifierSaveProgress.active
                ? h("div", { class: "classifier-save-progress" }, [
                  h("div", { class: "generation-modal-head" }, [
                    h("div", [
                      h("span", "Original Classifier Training"),
                      h("h3", "Running Classifier Script"),
                    ]),
                    h("b", `${classifierSaveProgress.progress}%`),
                  ]),
                  h("div", { class: "generation-progress" }, [
                    h("i", { style: { width: `${classifierSaveProgress.progress}%` } }),
                  ]),
                  h("p", classifierSaveProgress.phase),
                ])
                : null,
              h("div", { class: "area-input-actions" }, [
                h("button", { class: "cancel", type: "button", onClick: closeClassifierInput }, "Cancel"),
                h("button", { class: "save-input", type: "button", disabled: classifierSaving.value, onClick: saveClassifierForm }, classifierSaving.value ? "Saving" : "Save"),
              ]),
            ]),
          ])
          : null
      );
    const renderClassifierDetailModal = () =>
      h(Transition, { name: "modal-fade" }, () =>
        viewedClassifier.value
          ? h("div", { class: "area-detail-overlay", onClick: closeClassifierDetail }, [
            h("section", { class: "classifier-detail-modal wide", onClick: (event) => event.stopPropagation() }, [
              h("div", { class: "area-detail-head" }, [
                h("div", [
                  h("span", "Classifier Detail"),
                  h("h3", viewedClassifier.value.name),
                ]),
                h("div", { class: "modal-head-actions" }, [
                  h("button", {
                    type: "button",
                    class: "download-button retrain-detail-button",
                    disabled: retrainState.active,
                    onClick: retrainClassifier,
                  }, "Retrain"),
                  h("button", {
                    type: "button",
                    onClick: closeClassifierDetail,
                  }, "Close"),
                ]),
              ]),
              h("div", { class: "classifier-detail-grid detail-rows" }, [
                h("div", [h("span", "Model"), h("strong", viewedClassifier.value.name)]),
                h("div", [h("span", "Original Dataset"), h("strong", viewedClassifier.value.source)]),
                h("div", [h("span", "Type"), h("strong", viewedClassifier.value.type)]),
                h("div", [h("span", "Accuracy"), h("strong", viewedClassifier.value.acc)]),
                h("div", [h("span", "Created Time"), h("strong", formatDateTime(viewedClassifier.value.createdAt))]),
                h("div", [h("span", "Updated Time"), h("strong", formatDateTime(viewedClassifier.value.updatedAt))]),
                ]),
              h("div", { class: "classifier-detail-notes model-description" }, [
                h("strong", "Model Description"),
                h("p", viewedClassifier.value.description || viewedClassifier.value.modelDescription || "No model description is available."),
              ]),
              retrainState.active && retrainState.classifierId === viewedClassifier.value.id
                ? h("div", { class: "classifier-retrain-progress" }, [
                  h("div", { class: "generation-modal-head" }, [
                    h("div", [
                      h("span", "Classifier Retraining"),
                      h("h3", "Updating Original Classifier"),
                    ]),
                    h("b", `${retrainState.progress}%`),
                  ]),
                  h("div", { class: "generation-progress" }, [
                    h("i", { style: { width: `${retrainState.progress}%` } }),
                  ]),
                  h("p", retrainState.phase),
                ])
                : null,
            ]),
          ])
          : null
      );
    return () =>
      h("div", { class: "training-workspace" }, [
        h("section", { class: "training-card model-card classifier-card" }, [
          h("div", { class: "training-section-head" }, [
            h("div", [
              h("h3", "Supported Original Classifiers"),
            ]),
            h("button", {
              class: "mini-action",
              type: "button",
              onClick: triggerClassifierUpload,
            }, "Load Classifier"),
          ]),
          h("div", { class: "classifier-layout" }, [
            h("div", { class: "model-table" }, [
              h("div", { class: "model-table-head" }, [
                h("span", ""),
                h("span", "Model"),
                h("span", "Original"),
                h("span", "Type"),
                h("span", "Acc"),
                h("span", "Operations"),
              ]),
              ...(classifierLoading.value
                ? [h("div", { class: "model-row table-state" }, [h("span"), h("strong", "Loading classifiers..."), h("span"), h("span"), h("span"), h("span")])]
                : pagedRows().length
                  ? pagedRows().map((row) =>
                      h(
                        "div",
                        {
                          class: ["model-row", sameClassifier(selectedClassifier.value, row) ? "active" : ""],
                          onClick: () => {
                            selectClassifier(row);
                          },
                        },
                        [
                          h(
                            "button",
                            {
                              class: [
                                "classifier-check",
                                sameClassifier(selectedClassifier.value, row)
                                  ? "checked"
                                  : "",
                              ],
                              type: "button",
                              onClick: (event) => {
                                event.stopPropagation();
                                selectClassifier(row);
                              },
                            },
                            [h("span")]
                          ),
                          h("strong", row.name),
                          h("span", row.source),
                          h("em", { class: row.type === "Fine-tuned" ? "fine-tuned" : "pre-trained" }, row.type),
                          h("b", row.acc),
                          h("div", { class: "row-actions" }, [
                            h(
                              "button",
                              {
                                type: "button",
                                onClick: (event) => {
                                  event.stopPropagation();
                                  viewedClassifier.value = row;
                                },
                              },
                              "View"
                            ),
                            h(
                              "button",
                              {
                                class: "delete",
                                type: "button",
                                onClick: (event) => {
                                  event.stopPropagation();
                                  deleteClassifier(row);
                                },
                              },
                              "Delete"
                            ),
                          ]),
                        ]
                      )
                    )
                  : [h("div", { class: "model-row table-state" }, [h("span"), h("strong", "No classifier records"), h("span"), h("span"), h("span"), h("span")])]),
              h("div", { class: "table-pagination" }, [
                h(
                  "button",
                  {
                    type: "button",
                    disabled: currentPage.value === 1,
                    onClick: () => {
                      setPage(currentPage.value - 1);
                    },
                  },
                  "Prev"
                ),
                ...visiblePages().map((page) =>
                  page === "ellipsis"
                    ? h("span", { class: "page-ellipsis" }, "...")
                    : h(
                        "button",
                        {
                          class: ["page-number", currentPage.value === page ? "active" : ""],
                          type: "button",
                          onClick: () => {
                            setPage(page);
                          },
                        },
                        `${page}`
                      )
                ),
                h(
                  "button",
                    {
                      type: "button",
                      disabled: currentPage.value === pageCount(),
                      onClick: () => {
                        setPage(currentPage.value + 1);
                      },
                  },
                  "Next"
                ),
                h("input", {
                  class: "page-jump-input",
                  value: pageJump.value,
                  onInput: (event) => {
                    pageJump.value = event.target.value;
                  },
                  onKeydown: (event) => {
                    if (event.key === "Enter") {
                      setPage(Number(pageJump.value) || 1);
                    }
                  },
                }),
                h(
                  "button",
                  {
                    class: "page-go",
                    type: "button",
                    onClick: () => {
                      setPage(Number(pageJump.value) || 1);
                    },
                  },
                  "Go"
                ),
              ]),
            ]),
          ]),
        ]),

        h("section", { class: "training-card image-card" }, [
          h("div", { class: "training-section-head compact" }, [
            h("div", [
              h("h3", "Feature Visualization"),
              ]),
              h("div", { class: "visual-actions" }, [
                h("button", {
                  class: "image-upload",
                  type: "button",
                  onClick: triggerImageInput,
                }, [
                  h("span", "Image Input"),
                ]),
                h("input", {
                  ref: imageFileInput,
                  class: "feature-image-input",
                  type: "file",
                  accept: "image/*",
                  onChange: handleImageInput,
                }),
                h("button", {
                  class: "activation-button",
                  type: "button",
                  onClick: generateActivationMap,
                }, "Generate"),
              ]),
            ]),
          h("div", { class: "image-compare" }, [
            h("div", { class: "scan-panel raw" }, [
              h("div", { class: "scan-label" }, "Original B-scan Image"),
              h(
                "div",
                { class: ["square-frame", "source-frame", sourceImageUrl.value ? "has-image" : ""] },
                sourceImageUrl.value
                  ? [h("img", { src: sourceImageUrl.value, alt: "Original B-scan image" })]
                  : []
              ),
              ]),
              h("div", { class: "scan-panel activation" }, [
                h("div", { class: "scan-label" }, "Training Activation Map"),
                  h("div", {
                    class: [
                      "square-frame",
                      "activation-frame",
                      activationImageUrl.value ? "has-image" : "",
                    ],
                  }, activationImageUrl.value
                    ? [h("img", { src: activationImageUrl.value, alt: "Training activation map" })]
                    : []),
                ]),
            ]),
            h(Transition, { name: "modal-fade" }, () =>
              activationGeneration.active
                ? h("div", { class: "activation-generation-overlay" }, [
                  h("section", { class: "activation-generation-modal" }, [
                    h("div", { class: "generation-modal-head" }, [
                      h("div", [
                        h("span", "Feature Visualization"),
                        h("h3", "Generating Activation Map"),
                      ]),
                      h("b", `${activationGeneration.progress}%`),
                    ]),
                    h("div", { class: "generation-progress" }, [
                      h("i", { style: { width: `${activationGeneration.progress}%` } }),
                    ]),
                    h("p", activationGeneration.phase),
                  ]),
                ])
                : null
            ),
          ]),

        h("section", { class: "training-card train-card" }, [
          h("div", { class: "training-section-head" }, [
            h("div", [
              h("h3", "Model Training (Semantics + Classifier)"),
            ]),
            h("button", {
              class: "train-button",
              type: "button",
              onClick: startModelTraining,
            }, "Start"),
          ]),
          h("div", { class: "train-form" }, [
            h("div", { class: "train-form-row three" }, [
              h("label", [
                h("span", "Selected Classifier"),
                h(
                  "select",
                  {
                    value: selectedClassifier.value?.name || "",
                    onChange: handleClassifierModelChange,
                  },
                  classifierNameOptions.map((name) => h("option", { value: name }, name))
                ),
              ]),
              h("label", [
                h("span", "Original Area"),
                h(
                  "select",
                  {
                    value: originalDomain.value,
                    onChange: handleOriginalAreaChange,
                  },
                  trainingDomainOptions().map((domain) => h("option", { value: domain }, domain))
                ),
              ]),
              h("label", [
                h("span", "New Area"),
                h(
                  "select",
                  {
                    value: newDomain.value,
                    onChange: handleNewAreaChange,
                  },
                  trainingDomainOptions().map((domain) => h("option", { value: domain }, domain))
                ),
              ]),
            ]),
            h("div", { class: "train-form-row three" }, [
              h("label", [
                h("span", "Semantic Generator"),
                h("select", { value: trainingForm.semanticGenerator, onChange: (event) => (trainingForm.semanticGenerator = event.target.value) }, [
                  h("option", "GPT-4o"),
                  h("option", "GPT-3.5-turbo"),
                  h("option", "GPT-4o-mini"),
                  h("option", "Gemini-2.5"),
                  h("option", "LLaMA-3.1"),
                  h("option", "Qwen-2.5"),
                ]),
              ]),
              h("label", [
                h("span", "Embedding Model"),
                h("select", { value: trainingForm.embeddingModel, onChange: (event) => (trainingForm.embeddingModel = event.target.value) }, [
                  h("option", "CLIP"),
                  h("option", "SBERT"),
                  h("option", "LLaMA"),
                  h("option", "Qwen"),
                ]),
              ]),
              h("label", [
                h("span", "Optimizer"),
                h("select", { value: trainingForm.optimizer, onChange: (event) => (trainingForm.optimizer = event.target.value) }, [
                  h("option", "Adam"),
                  h("option", "AdamW"),
                  h("option", "SGD"),
                ]),
              ]),
            ]),
            h("div", { class: "train-slider-stack" }, [
              renderTrainingSlider(trainingParams.knowledgeEntries),
              renderTrainingSlider(trainingParams.refinementIterations),
              renderTrainingSlider(trainingParams.learningRate),
              h("div", { class: "dual-slider-row" }, [
                renderCompactTrainingSlider(trainingParams.batchSize),
                renderCompactTrainingSlider(trainingParams.epochs),
              ]),
            ]),
          ]),
          h(Transition, { name: "modal-fade" }, () =>
            modelTraining.active
              ? h("div", { class: "training-progress-overlay" }, [
                h("section", { class: "activation-generation-modal training-run-modal" }, [
                  h("div", { class: "generation-modal-head" }, [
                    h("div", [
                      h("span", "Model Training"),
                      h("h3", "Classifier Construction"),
                    ]),
                    h("b", `${modelTraining.progress}%`),
                  ]),
                  h("div", { class: "generation-progress" }, [
                    h("i", { style: { width: `${modelTraining.progress}%` } }),
                  ]),
                  h("p", modelTraining.phase),
                ]),
              ])
              : null
          ),
        ]),
        renderClassifierInputModal(),
        renderClassifierDetailModal(),
      ]);
  },
});
</script>













