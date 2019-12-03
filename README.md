# VK_RAYTRACE
![vk_raytrace](doc/vk_raytrace.png)

Similar to vk_scene, it reads [glTF](https://www.khronos.org/gltf/) scenes but renders the scene using NVIDIA raytracing. Each object goes to a BLAS and a TLAS is created from instances referencing the BLAS. Besides the shading, the materials are all uploaded and accessible at shading time. For this to work, unsized arrays of textures, materials, matrices, vertices, indices and all other vertex attributes are in buffers, in conjunction with a primitive buffer having the offets to arrays. The example shows as well how to implement a picking ray, which is using the same acceleration struction for drawing, but is using the data to return the information under the mouse cursor. This information can be use for setting the camera interest position, or to debug any shading data. 


Tags: 
- raytracing, GLTF, HDR, tonemapper, picking, BLAS, TLAS, PBR material

Extensions: 
- VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, VK_NV_RAY_TRACING_EXTENSION_NAME, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME, VK_KHR_MAINTENANCE3_EXTENSION_NAME, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME