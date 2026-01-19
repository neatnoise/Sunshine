<script setup>
import { ref } from 'vue'

const props = defineProps([
  'platform',
  'config',
])

const config = ref(props.config)
</script>

<template>
  <div id="vulkan-encoder" class="config-page">
    <p>FFmpeg Vulkan Video encoder uses GPU hardware encoding via Vulkan API.</p>
  </div>
</template>

<style scoped>

</style>
