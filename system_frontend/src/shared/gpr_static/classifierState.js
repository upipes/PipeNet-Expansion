export const originalClassifierSources = [
  "GPR-SD",
  "GPR-Road",
  "a1 Sandy Loam",
  "a2 Saturated Silty Clay",
  "a3 Urban Backfill Soil",
  "a4 Layered Road Structure",
];

export const classifierModels = [
  { name: "ResNet-50", feature: "2048-d residual embedding", baseAcc: 90.2 },
  { name: "ResNet-101", feature: "Deep residual embedding", baseAcc: 91.4 },
  { name: "ViT-S/16", feature: "Patch-token transformer embedding", baseAcc: 92.0 },
];

export const classifierTypes = [
  { name: "Fine-tuned", offset: 1.1 },
  { name: "Pre-trained", offset: 0 },
];

export const modelRows = originalClassifierSources.flatMap((source, sourceIndex) =>
  classifierModels.flatMap((model, modelIndex) =>
    classifierTypes.map((type, typeIndex) => ({
      name: model.name,
      source,
      type: type.name,
      feature: model.feature,
      status: sourceIndex === 0 && modelIndex === 0 && typeIndex === 0 ? "Selected" : "Ready",
      acc: `${(model.baseAcc + type.offset + sourceIndex * 0.35).toFixed(1)}%`,
      details: {
        tuner:
          type.name === "Fine-tuned"
            ? "Fine-tune the classifier head and final residual stage with domain-aware augmentation."
            : "Load the original-domain pre-trained backbone and adapt the classifier head.",
        schedule:
          model.name === "ResNet-50"
            ? "60 epochs, cosine LR decay, warm-up 5 epochs."
            : "80 epochs, AdamW optimizer, semantic consistency loss enabled.",
        adaptation: `${source} classifier features are aligned with generated semantic descriptions.`,
      },
    }))
  )
);

