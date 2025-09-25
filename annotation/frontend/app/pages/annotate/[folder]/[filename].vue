<template>
    <div class="relative">
    <img :src="imageUrl" alt="Detected image" class="w-full rounded-lg" />
    <div v-for="(xyxy, index) in annotations.xyxy" :key="index">
        <div
            :style="{
                position: 'absolute',
                border: '2px solid red',
                left: `${xyxy[0]}px`,
                top: `${xyxy[1]}px`,
                width: `${xyxy[2] - xyxy[0]}px`,
                height: `${xyxy[3] - xyxy[1]}px`,
            }"
        ></div>
        <div
            :style="{
                position: 'absolute',
                border: '2px solid red',
                left: `${xyxy[4]}px`,
                top: `${xyxy[5]}px`,
                width: `${xyxy[6] - xyxy[4]}px`,
                height: `${xyxy[7] - xyxy[5]}px`,
            }"
        ></div>
    </div>
    </div>
</template>

<script setup lang="ts">
const route = useRoute()

const imageUrl = computed(() => {
    return `http://localhost:8000/images/${route.params.folder}/${route.params.filename}`;
});

const {data: annotations} = await useFetch(`http://localhost:8000/predict/${route.params.folder}/${route.params.filename}`);

</script>