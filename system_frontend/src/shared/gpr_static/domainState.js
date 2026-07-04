import { reactive, ref } from "vue";

export const domainOptions = reactive([
  "GPR-SD",
  "GPR-Road",
  "a1 Sandy Loam",
  "a2 Saturated Silty Clay",
  "a3 Urban Backfill Soil",
  "a4 Layered Road Structure",
]);

export const originalDomain = ref("a1 Sandy Loam");
export const newDomain = ref("a3 Urban Backfill Soil");

export const roadClasses = ["Cavity", "Crack", "Loose", "Normal", "Pipeline"];
export const soilClasses = ["Cavity", "Crack", "Metal Pipeline"];

export const sourceMeta = reactive({
  condition: "Validated source area",
  roadSurface: "Concrete / asphalt roads",
  metrics: [
    {
      name: "Frequency Range",
      value: "200-400 MHz",
      type: "range",
      start: 33.3,
      width: 33.4,
      min: 0,
      max: 600,
      step: 10,
      unit: "MHz",
    },
    {
      name: "Time Window",
      value: "97 ns",
      type: "point",
      point: 97,
      min: 0,
      max: 100,
      step: 1,
      unit: "ns",
    },
  ],
});

export const targetMeta = reactive({
  condition: "Changed sensing condition",
  roadSurface: "Concrete / asphalt / unpaved",
  metrics: [
    {
      name: "Frequency Range",
      value: "50-600 MHz",
      type: "range",
      start: 7.7,
      width: 84.6,
      min: 0,
      max: 650,
      step: 10,
      unit: "MHz",
    },
    {
      name: "Time Window",
      value: "70 ns",
      type: "point",
      point: 70,
      min: 0,
      max: 100,
      step: 1,
      unit: "ns",
    },
  ],
});

export const soilProfiles = reactive({
  "a1 Sandy Loam": {
    composition: [
      { name: "Sand", value: "64%", type: "point", point: 64, min: 0, max: 100, step: 1, unit: "%" },
      { name: "Silt", value: "24%", type: "point", point: 24, min: 0, max: 100, step: 1, unit: "%" },
      { name: "Clay", value: "12%", type: "point", point: 12, min: 0, max: 100, step: 1, unit: "%" },
    ],
    water: {
      name: "Water Content",
      value: "5-18%",
      type: "range",
      start: 12.5,
      width: 32.5,
      min: 0,
      max: 40,
      step: 1,
      unit: "%",
    },
    electrical: [
      { name: "Relative Permittivity", value: "6.8-12.4", type: "range", start: 23, width: 22, min: 0, max: 30, step: 0.1, unit: "" },
      { name: "Conductivity", value: "0.006-0.021 S/m", type: "range", start: 12, width: 30, min: 0, max: 0.05, step: 0.001, unit: "S/m" },
    ],
    peplinski: { name: "Peplinski Model Fractal Dimension", value: "1.32", type: "point", point: 42, min: 1.0, max: 1.8, step: 0.01, unit: "" },
  },
  "a2 Saturated Silty Clay": {
    composition: [
      { name: "Sand", value: "18%", type: "point", point: 18, min: 0, max: 100, step: 1, unit: "%" },
      { name: "Silt", value: "42%", type: "point", point: 42, min: 0, max: 100, step: 1, unit: "%" },
      { name: "Clay", value: "40%", type: "point", point: 40, min: 0, max: 100, step: 1, unit: "%" },
    ],
    water: {
      name: "Water Content",
      value: "22-38%",
      type: "range",
      start: 55,
      width: 40,
      min: 0,
      max: 40,
      step: 1,
      unit: "%",
    },
    electrical: [
      { name: "Relative Permittivity", value: "18.5-31.2", type: "range", start: 46, width: 32, min: 0, max: 40, step: 0.1, unit: "" },
      { name: "Conductivity", value: "0.038-0.092 S/m", type: "range", start: 35, width: 49, min: 0, max: 0.11, step: 0.001, unit: "S/m" },
    ],
    peplinski: { name: "Peplinski Model Fractal Dimension", value: "1.58", type: "point", point: 73, min: 1.0, max: 1.8, step: 0.01, unit: "" },
  },
  "a3 Urban Backfill Soil": {
    composition: [
      { name: "Sand", value: "46%", type: "point", point: 46, min: 0, max: 100, step: 1, unit: "%" },
      { name: "Silt", value: "34%", type: "point", point: 34, min: 0, max: 100, step: 1, unit: "%" },
      { name: "Clay", value: "20%", type: "point", point: 20, min: 0, max: 100, step: 1, unit: "%" },
    ],
    water: {
      name: "Water Content",
      value: "10-26%",
      type: "range",
      start: 25,
      width: 40,
      min: 0,
      max: 40,
      step: 1,
      unit: "%",
    },
    electrical: [
      { name: "Relative Permittivity", value: "9.2-21.6", type: "range", start: 23, width: 31, min: 0, max: 40, step: 0.1, unit: "" },
      { name: "Conductivity", value: "0.014-0.058 S/m", type: "range", start: 13, width: 40, min: 0, max: 0.11, step: 0.001, unit: "S/m" },
    ],
    peplinski: { name: "Peplinski Model Fractal Dimension", value: "1.46", type: "point", point: 58, min: 1.0, max: 1.8, step: 0.01, unit: "" },
  },
  "a4 Layered Road Structure": {
    composition: [
      { name: "Sand", value: "38%", type: "point", point: 38, min: 0, max: 100, step: 1, unit: "%" },
      { name: "Silt", value: "37%", type: "point", point: 37, min: 0, max: 100, step: 1, unit: "%" },
      { name: "Clay", value: "25%", type: "point", point: 25, min: 0, max: 100, step: 1, unit: "%" },
    ],
    water: {
      name: "Water Content",
      value: "8-24%",
      type: "range",
      start: 20,
      width: 40,
      min: 0,
      max: 40,
      step: 1,
      unit: "%",
    },
    electrical: [
      { name: "Relative Permittivity", value: "8.4-19.8", type: "range", start: 21, width: 29, min: 0, max: 40, step: 0.1, unit: "" },
      { name: "Conductivity", value: "0.010-0.046 S/m", type: "range", start: 9, width: 33, min: 0, max: 0.11, step: 0.001, unit: "S/m" },
    ],
    peplinski: { name: "Peplinski Model Fractal Dimension", value: "1.41", type: "point", point: 51, min: 1.0, max: 1.8, step: 0.01, unit: "" },
  },
});

