<template>
  <div class="gpr-page">
    <header class="app-topbar">
      <div class="brand-block">
        <div class="brand-symbol">GPR</div>
        <div>
          <h1>Cross-Area GPR Subsurface Diagnosis System</h1>
          <p>Cross-Area Adaptation for Subsurface Diagnosis Guided by Environmental Knowledge</p>
        </div>
      </div>
      <div class="topbar-actions">
        <el-button round>Settings</el-button>
      </div>
    </header>

    <main class="system-grid">
      <SemanticGenerationPanel />
      <ModelTrainingPanel />
      <ModelTestingPanel />
    </main>
  </div>
</template>

<script setup>
import { SemanticGenerationPanel, ModelTrainingPanel, ModelTestingPanel } from "../components/gpr";
</script>
