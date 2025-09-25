<template>
  <div class="flex">
    <UNavigationMenu orientation="vertical" :items="navigation" class="data-[orientation=vertical]:w-48" />
    <NuxtPage />
  </div>
</template>

<script setup lang="ts">
import type { NavigationMenuItem } from '@nuxt/ui'

const { data } = await useFetch('http://localhost:8000/folders');

const navigation = computed<NavigationMenuItem[][]>(() => {
  let result = [{
    label: 'Folders',
    type: 'folder',
  }]
  let folders = data.value?.map(folder => ({
    label: folder.name,
    children: folder.annotations.map(ann => ({
      label: ann.filename,
      to: "/annotate/" + folder.name + "/" + ann.filename,
    })),
  })) || [];
  return [result.concat(folders)];
});
</script>
