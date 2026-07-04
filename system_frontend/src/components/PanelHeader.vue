<template>
  <div class="panel-header">
    <div class="panel-code">{{ code }}</div>
    <div>
      <h2>{{ title }}</h2>
      <p>{{ subtitle }}</p>
    </div>
  </div>
</template>

<script setup>
defineProps({
  code: String,
  title: String,
  subtitle: String,
});
</script>
