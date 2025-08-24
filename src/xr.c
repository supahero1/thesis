/*
 *   Copyright 2025 Franciszek Balcerak
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <thesis/xr.h>
#include <thesis/file.h>
#include <thesis/debug.h>
#include <thesis/atomic.h>
#include <thesis/shared.h>
#include <thesis/options.h>
#include <thesis/threads.h>
#include <thesis/alloc_ext.h>

#include <volk.h>

#define XR_USE_PLATFORM_WAYLAND
#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr_platform.h>

#include <string.h>

#define VK_MAX_IMAGES 8
#define VK_MAX_FRAMES 2
#define VK_MAX_INSTANCES 128
#define VK_POOL_SIZE 16
#define VK_COMMANDS 8

#define VK_WINDOW_WIDTH 1280
#define VK_WINDOW_HEIGHT 720
#define VK_WINDOW_SENSITIVITY 0.003f
#define VK_WINDOW_SPEED 500.0f


typedef enum vk_image_type
{
	VK_IMAGE_TYPE_DEPTH,
	VK_IMAGE_TYPE_ATTACHMENT,
	VK_IMAGE_TYPE_SAMPLED,
	VK_IMAGE_TYPE_MULTISAMPLED,
	VK_IMAGE_TYPE_TRANSIENT,
	VK_IMAGE_TYPE_TEXTURE,
	VK_IMAGE_TYPE_CUBE,
	VK_IMAGE_TYPE_CUSTOM_FORMAT,
	VK_IMAGE_TYPE_CUSTOM_SIZE,
	MACRO_ENUM_BITS_EXP(VK_IMAGE_TYPE),

	VK_IMAGE_TYPE_DEPTH_BIT			= MACRO_POWER_OF_2(VK_IMAGE_TYPE_DEPTH),
	VK_IMAGE_TYPE_ATTACHMENT_BIT	= MACRO_POWER_OF_2(VK_IMAGE_TYPE_ATTACHMENT),
	VK_IMAGE_TYPE_SAMPLED_BIT		= MACRO_POWER_OF_2(VK_IMAGE_TYPE_SAMPLED),
	VK_IMAGE_TYPE_MULTISAMPLED_BIT	= MACRO_POWER_OF_2(VK_IMAGE_TYPE_MULTISAMPLED),
	VK_IMAGE_TYPE_TRANSIENT_BIT		= MACRO_POWER_OF_2(VK_IMAGE_TYPE_TRANSIENT),
	VK_IMAGE_TYPE_TEXTURE_BIT		= MACRO_POWER_OF_2(VK_IMAGE_TYPE_TEXTURE),
	VK_IMAGE_TYPE_CUBE_BIT			= MACRO_POWER_OF_2(VK_IMAGE_TYPE_CUBE),
	VK_IMAGE_TYPE_CUSTOM_FORMAT_BIT	= MACRO_POWER_OF_2(VK_IMAGE_TYPE_CUSTOM_FORMAT),
	VK_IMAGE_TYPE_CUSTOM_SIZE_BIT	= MACRO_POWER_OF_2(VK_IMAGE_TYPE_CUSTOM_SIZE),

	VK_IMAGE_TYPE_TEXTURE_2D_BITS = VK_IMAGE_TYPE_SAMPLED_BIT | VK_IMAGE_TYPE_TEXTURE_BIT,
	VK_IMAGE_TYPE_TEXTURE_CUBE_BITS = VK_IMAGE_TYPE_TEXTURE_2D_BITS | VK_IMAGE_TYPE_CUBE_BIT
}
vk_image_type_t;

typedef struct vk_image
{
	str_t path;

	void* data;
	uint32_t size;
	uint32_t width;
	uint32_t height;
	uint32_t levels;
	uint32_t layers;

	VkFormat format;
	vk_image_type_t type;

	VkImage image;
	VkImageView view;
	VkDeviceMemory memory;

	VkImageAspectFlags aspect;
	VkImageUsageFlags usage;
	VkSampleCountFlagBits samples;
}
vk_image_t;

typedef struct vk_buffer
{
	VkBuffer buffer;
	VkDeviceMemory memory;
}
vk_buffer_t;

typedef struct vk_shadow_vert_constant_data
{
	mat4 transform;
}
vk_shadow_vert_constant_data_t;

typedef struct vk_shadow_vertex_data
{
	vec3 position;
}
vk_shadow_vertex_data_t;

typedef struct vk_scene_vert_ubo_data
{
	mat4 projection;
	mat4 view;
	mat4 light_transform;
	vec4 light_direction;
	vec3 camera_position;
}
vk_scene_vert_ubo_data_t;

typedef struct vk_mesh_vertex_data
{
	vec3 position;
	vec3 normal;
	vec2 coords;
}
vk_mesh_vertex_data_t;

typedef struct vk_material_constant_data
{
	vec4 diffuse;
	vec4 ambient;
	float shininess;
	float shininess_strength;
}
vk_material_constant_data_t;

typedef struct vk_scene_frag_constant_data
{
	vec4 diffuse;
	vec4 ambient;
	float shininess;
	float shininess_strength;

	float near;
}
vk_scene_frag_constant_data_t;

typedef struct vk_ssao_frag_ubo_data
{
	mat4 projection;
}
vk_ssao_frag_ubo_data_t;

typedef struct vk_ssao_frag_kernel_ubo_data
{
	vec4 samples[];
}
vk_ssao_frag_kernel_ubo_data_t;

typedef struct vk_skybox_constant_data
{
	mat4 transform;
}
vk_skybox_constant_data_t;

typedef struct vk_skybox_vertex_data
{
	vec3 position;
}
vk_skybox_vertex_data_t;

typedef struct vk_material
{
	vk_image_t texture;
	vk_material_constant_data_t constant_data;
	VkDescriptorSet set;
}
vk_material_t;

typedef struct vk_mesh
{
	uint32_t material_idx;
	uint32_t vertex_count;
	uint32_t index_count;

	vk_buffer_t shadow_vertex_buffer;
	vk_buffer_t scene_vertex_buffer;
	vk_buffer_t index_buffer;
}
vk_mesh_t;

typedef struct vk_model
{
	vk_mesh_t* meshes;
	uint32_t mesh_count;

	vk_buffer_t instance_buffer;
}
vk_model_t;

typedef struct vk_model_instance_data
{
	mat4 transform;
}
vk_model_instance_data_t;

typedef struct vk_entities_per_model
{
	simulation_entity_data_t** entities;
	uint32_t entities_used;
	uint32_t entities_size;
}
vk_entities_per_model_t;

typedef struct vk_frame_image
{
	vk_image_t image;
	VkDescriptorSet set;
}
vk_frame_image_t;

typedef struct vk_frame_buffer
{
	vk_buffer_t buffer;
	VkDescriptorSet set;
}
vk_frame_buffer_t;

typedef struct barrier vk_barrier_t;

typedef struct frame
{
	struct
	{
		vk_frame_image_t map;

		VkFramebuffer framebuffer;
	}
	shadow;

	struct
	{
		vk_frame_buffer_t vert_ubo;

		vk_frame_image_t position_ms;
		vk_frame_image_t normal_ms;

		vk_frame_image_t position;
		vk_frame_image_t normal;
		VkDescriptorSet set;

		vk_frame_image_t map_ms;
		vk_frame_image_t map;

		vk_image_t depth;
		VkFramebuffer framebuffer;
	}
	scene;

	struct
	{
		vk_frame_buffer_t frag_ubo;

		vk_frame_image_t map;

		VkFramebuffer framebuffer;
	}
	ssao;

	struct
	{
		vk_frame_image_t map;

		VkFramebuffer framebuffer;
	}
	ssao_blur;

	struct
	{
		VkImage image;
		VkImageView image_view;
		VkFramebuffer framebuffer;
	}
	output;

	VkSemaphore semaphore;
	vk_barrier_t* barrier;
}
vk_frame_t;

struct barrier
{
	VkSemaphore semaphore;
	VkFence fence;
	VkCommandBuffer command_buffer;
};

typedef enum vk_preview
{
	VK_PREVIEW_NONE,
	VK_PREVIEW_SHADOW_MAP,
	VK_PREVIEW_SCENE_POSITION_MAP,
	VK_PREVIEW_SCENE_NORMAL_MAP,
	VK_PREVIEW_SCENE_MAP,
	VK_PREVIEW_SSAO_MAP,
	VK_PREVIEW_SSAO_BLUR_MAP,
	MACRO_ENUM_BITS(VK_PREVIEW)
}
vk_preview_t;

typedef struct vk_extent
{
	uint32_t width;
	uint32_t height;
	VkExtent2D extent;
	VkViewport viewport;
	VkRect2D scissor;
}
vk_extent_t;

typedef struct vk_command
{
	VkCommandBuffer buffer;
	VkFence fence;

	vk_buffer_t staging_buffer;
}
vk_command_t;

typedef struct vk_descriptor_pool vk_descriptor_pool_t;

struct vk_descriptor_pool
{
	vk_descriptor_pool_t* next;
	vk_descriptor_pool_t* prev;
	vk_descriptor_pool_t* free_next;
	vk_descriptor_pool_t* free_prev;

	VkDescriptorPool pool;
	uint32_t allocations;
};

typedef struct vk_descriptor_set_pool
{
	vk_descriptor_pool_t* head;
	vk_descriptor_pool_t* free_head;
	VkDescriptorPoolSize* sizes;
	uint32_t size_count;
	uint32_t refs;
}
vk_descriptor_set_pool_t;

typedef struct vk_descriptor_set_layout
{
	VkDescriptorSetLayout layout;
	vk_descriptor_set_pool_t* set_pool;
	uint32_t multiplier;
}
vk_descriptor_set_layout_t;

struct xr
{
	simulation_t simulation;

	struct
	{
		bool xr_enable;

		uint32_t max_msaa_samples;
		bool sample_shading;
		float min_sample_shading;
		uint32_t mipmap_levels;
		float max_anisotropy;
		vk_preview_t preview;

		uint32_t shadow_map_size;
		bool enable_depth_shadows;
		bool enable_backface_shadows;
		bool enable_specular;
		float shadow_value;
		float lambert_start_angle;

		bool enable_ssao;
		uint32_t ssao_kernel_size;
		uint32_t ssao_noise_size;
		float ssao_radius;
		float ssao_bias;
		float ssao_power;
		float ssao_depth_k;
		float ssao_depth_gamma;
		bool ssao_debug;
		float ssao_scale;

		float ssao_blur_radius;
		float ssao_blur_falloff;
		float ssao_blur_depth_tolerance;
	}
	options;

#ifndef NDEBUG
	XrDebugUtilsMessengerEXT debug_messenger;
#endif

	XrInstance instance;
	XrSystemId system;
	XrSession session;

	struct
	{
		PFN_vkGetInstanceProcAddr proc_addr_fn;

#ifndef NDEBUG
		VkDebugUtilsMessengerEXT debug_messenger;
#endif

		VkInstance instance;

		VkSurfaceKHR surface;
		VkSurfaceCapabilitiesKHR surface_capabilities;

		VkPhysicalDevice physical_device;
		uint32_t queue_id;

		VkDevice device;
		struct VolkDeviceTable table;
		VkQueue queue;

		VkSampleCountFlagBits samples;
		VkPhysicalDeviceLimits device_limits;
		VkPhysicalDeviceMemoryProperties memory_properties;
	}
	vk;
};


private const char* xr_xr_instance_extensions[] =
{
#ifndef NDEBUG
	XR_EXT_DEBUG_UTILS_EXTENSION_NAME,
#endif
	XR_KHR_VULKAN_ENABLE_EXTENSION_NAME,
	XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME,
	XR_EXT_HAND_TRACKING_EXTENSION_NAME,
};

private const char* xr_xr_instance_layers[] =
{
#ifndef NDEBUG
	"XR_APILAYER_LUNARG_core_validation",
#endif
};

private const char* xr_vk_instance_extensions[] =
{
#ifndef NDEBUG
	VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
#endif
};

private const char* xr_vk_instance_layers[] =
{
#ifndef NDEBUG
	"VK_LAYER_KHRONOS_validation"
#endif
};

private const char* xr_vk_device_extensions[] =
{
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	VK_KHR_MULTIVIEW_EXTENSION_NAME,
};

private const char* xr_vk_device_layers[] =
{
#ifndef NDEBUG
	"VK_LAYER_KHRONOS_validation"
#endif
};


private vk_skybox_vertex_data_t xr_vk_skybox_vertex_data[] =
{
	{ { -1.0f, -1.0f, -1.0f } },
	{ { -1.0f, -1.0f,  1.0f } },
	{ { -1.0f,  1.0f, -1.0f } },
	{ { -1.0f,  1.0f,  1.0f } },
	{ {  1.0f, -1.0f, -1.0f } },
	{ {  1.0f, -1.0f,  1.0f } },
	{ {  1.0f,  1.0f, -1.0f } },
	{ {  1.0f,  1.0f,  1.0f } },
};

private uint16_t xr_vk_skybox_index_data[] =
{
	0, 1, 2, 2, 1, 3,
	4, 6, 5, 5, 6, 7,
	0, 4, 1, 1, 4, 5,
	2, 3, 6, 6, 3, 7,
	0, 2, 4, 4, 2, 6,
	1, 5, 3, 3, 5, 7
};


private VkPipelineColorBlendAttachmentState xr_vk_no_blending_attachment =
{
	.blendEnable = VK_FALSE,
	.srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
	.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
	.colorBlendOp = VK_BLEND_OP_ADD,
	.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
	.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
	.alphaBlendOp = VK_BLEND_OP_ADD,
	.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
};


private VkPipelineColorBlendAttachmentState xr_vk_blending_attachment =
{
	.blendEnable = VK_TRUE,
	.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
	.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
	.colorBlendOp = VK_BLEND_OP_ADD,
	.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
	.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
	.alphaBlendOp = VK_BLEND_OP_ADD,
	.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
};


private void
xr_init_options(
	xr_t xr
	)
{
	assert_not_null(xr);

	puts("\nXR options:");

	xr->options.xr_enable =
		options_get_boolean(global_options, "xr_enable", true);
	printf("- xr_enable: %d\n", xr->options.xr_enable);

	xr->options.max_msaa_samples =
		options_get_i64(global_options, "vk_max_msaa_samples", 1, 64, 8);
	printf("- max_msaa_samples: %u\n", xr->options.max_msaa_samples);

	xr->options.sample_shading =
		options_get_boolean(global_options, "vk_sample_shading", false);
	printf("- sample_shading: %d\n", xr->options.sample_shading);

	xr->options.min_sample_shading =
		options_get_f32(global_options, "vk_min_sample_shading", 0.0f, 1.0f, 0.2f);
	printf("- min_sample_shading: %.2f\n", xr->options.min_sample_shading);

	xr->options.mipmap_levels =
		options_get_i64(global_options, "vk_mipmap_levels", 0, 16, 3);
	printf("- mipmap_levels: %u\n", xr->options.mipmap_levels);

	xr->options.max_anisotropy =
		options_get_f32(global_options, "vk_max_anisotropy", 0.0f, 100.0f, 100.0f);
	printf("- max_anisotropy: %.1f\n", xr->options.max_anisotropy);

	xr->options.preview =
		options_get_i64(global_options, "vk_preview", 0, VK_PREVIEW__COUNT - 1, VK_PREVIEW_NONE);
	printf("- preview: %d\n", xr->options.preview);

	xr->options.shadow_map_size =
		options_get_i64(global_options, "vk_shadow_map_size", 1, 16384, 4096);
	printf("- shadow_map_size: %u\n", xr->options.shadow_map_size);

	xr->options.enable_depth_shadows =
		options_get_boolean(global_options, "vk_enable_depth_shadows", true);
	printf("- enable_depth_shadows: %d\n", xr->options.enable_depth_shadows);

	xr->options.enable_backface_shadows =
		options_get_boolean(global_options, "vk_enable_backface_shadows", true);
	printf("- enable_backface_shadows: %d\n", xr->options.enable_backface_shadows);

	xr->options.enable_specular =
		options_get_boolean(global_options, "vk_enable_specular", true);
	printf("- enable_specular: %d\n", xr->options.enable_specular);

	xr->options.shadow_value =
		options_get_f32(global_options, "vk_shadow_value", 0.0f, 1.0f, 0.2f);
	printf("- shadow_value: %.2f\n", xr->options.shadow_value);

	xr->options.lambert_start_angle =
		options_get_f32(global_options, "vk_lambert_start_angle", 0.0f, 90.0f, 80.0f);
	printf("- lambert_start_angle: %.1f\n", xr->options.lambert_start_angle);

	xr->options.enable_ssao =
		options_get_boolean(global_options, "vk_enable_ssao", true);
	printf("- enable_ssao: %d\n", xr->options.enable_ssao);

	xr->options.ssao_kernel_size =
		options_get_i64(global_options, "vk_ssao_kernel_size", 1, 256, 40);
	printf("- ssao_kernel_size: %u\n", xr->options.ssao_kernel_size);

	xr->options.ssao_noise_size =
		options_get_i64(global_options, "vk_ssao_noise_size", 1, 64, 4);
	printf("- ssao_noise_size: %u\n", xr->options.ssao_noise_size);

	xr->options.ssao_radius =
		options_get_f32(global_options, "vk_ssao_radius", 0.0f, 64.0f, 6.0f);
	printf("- ssao_radius: %.2f\n", xr->options.ssao_radius);

	xr->options.ssao_bias =
		options_get_f32(global_options, "vk_ssao_bias", 0.0f, 1.0f, 0.05f);
	printf("- ssao_bias: %.3f\n", xr->options.ssao_bias);

	xr->options.ssao_power =
		options_get_f32(global_options, "vk_ssao_power", 0.0f, 5.0f, 2.0f);
	printf("- ssao_power: %.2f\n", xr->options.ssao_power);

	xr->options.ssao_depth_k =
		options_get_f32(global_options, "vk_ssao_depth_k", 0.0f, 1.0f, 0.007f);
	printf("- ssao_depth_k: %.4f\n", xr->options.ssao_depth_k);

	xr->options.ssao_depth_gamma =
		options_get_f32(global_options, "vk_ssao_depth_gamma", 0.0f, 8.0f, 1.5f);
	printf("- ssao_depth_gamma: %.2f\n", xr->options.ssao_depth_gamma);

	xr->options.ssao_debug =
		options_get_boolean(global_options, "vk_ssao_debug", false);
	printf("- ssao_debug: %d\n", xr->options.ssao_debug);

	xr->options.ssao_scale =
		options_get_f32(global_options, "vk_ssao_scale", 0.0f, 1.0f, 1.0f);
	printf("- ssao_scale: %.2f\n", xr->options.ssao_scale);

	xr->options.ssao_blur_radius =
		options_get_f32(global_options, "vk_ssao_blur_radius", 0.0f, 16.0f, 4.0f);
	printf("- ssao_blur_radius: %.2f\n", xr->options.ssao_blur_radius);

	xr->options.ssao_blur_falloff =
		options_get_f32(global_options, "vk_ssao_blur_falloff", 0.0f, 4.0f, 1.9f);
	printf("- ssao_blur_falloff: %.2f\n", xr->options.ssao_blur_falloff);

	xr->options.ssao_blur_depth_tolerance =
		options_get_f32(global_options, "vk_ssao_blur_depth_tolerance", 0.0f, 16.0f, 2.0f);
	printf("- ssao_blur_depth_tolerance: %.2f\n", xr->options.ssao_blur_depth_tolerance);
}


#ifndef NDEBUG

private XRAPI_ATTR XrBool32 XRAPI_CALL
xr_xr_debug_callback(
	XrDebugUtilsMessageSeverityFlagsEXT severity,
	XrDebugUtilsMessageTypeFlagsEXT type,
	const XrDebugUtilsMessengerCallbackDataEXT* data,
	void* user_data
	)
{
	fputs(data->message, stderr);
	return false;
}

private VKAPI_ATTR VkBool32 VKAPI_CALL
xr_vk_debug_callback(
	VkDebugUtilsMessageSeverityFlagBitsEXT severity,
	VkDebugUtilsMessageTypeFlagsEXT type,
	const VkDebugUtilsMessengerCallbackDataEXT* data,
	void* user_data
	)
{
	fputs(data->pMessage, stderr);
	return false;
}

#endif


private void*
xr_xr_load_func(
	xr_t xr,
	const char* name
	)
{
	assert_not_null(xr);
	assert_not_null(name);

	PFN_xrVoidFunction func;
	XrResult result = xrGetInstanceProcAddr(xr->instance, name, &func);
	hard_assert_eq(result, XR_SUCCESS, fprintf(stderr, "XR function %s not found\n", name));
	assert_not_null(func);

	return func;
}


private void*
xr_vk_load_func(
	xr_t xr,
	const char* name
	)
{
	assert_not_null(xr);
	assert_not_null(name);

	void* func = xr->vk.proc_addr_fn(xr->vk.instance, name);
	hard_assert_not_null(func, fprintf(stderr, "VK function %s not found\n", name));

	return func;
}


private const char**
xr_xr_get_instance_extensions(
	xr_t xr,
	const char** extension
	)
{
	assert_not_null(xr);
	assert_not_null(extension);

	uint32_t available_instance_extension_count = 0;
	XrResult result = xrEnumerateInstanceExtensionProperties(NULL, 0, &available_instance_extension_count, NULL);
	hard_assert_eq(result, XR_SUCCESS);

	XrExtensionProperties available_instance_extensions[available_instance_extension_count];

	XrExtensionProperties* available_instance_extension = available_instance_extensions;
	XrExtensionProperties* available_instance_extension_end =
		available_instance_extension + available_instance_extension_count;

	while(available_instance_extension < available_instance_extension_end)
	{
		*(available_instance_extension++) = (XrExtensionProperties){XR_TYPE_EXTENSION_PROPERTIES};
	}

	result = xrEnumerateInstanceExtensionProperties(NULL,
		available_instance_extension_count, &available_instance_extension_count, available_instance_extensions);
	hard_assert_eq(result, XR_SUCCESS);

	puts("\nXR instance extensions:");

	for(
		available_instance_extension = available_instance_extensions;
		available_instance_extension < available_instance_extension_end;
		available_instance_extension++
		)
	{
		printf("- %s\n", available_instance_extension->extensionName);
	}

	puts("");

	const char* const* instance_extension = xr_xr_instance_extensions;
	const char* const* instance_extension_end = instance_extension + MACRO_ARRAY_LEN(xr_xr_instance_extensions);

	while(instance_extension < instance_extension_end)
	{
		bool found = false;
		const char* extension_name = *(instance_extension++);

		available_instance_extension = available_instance_extensions;
		while(available_instance_extension < available_instance_extension_end)
		{
			if(strcmp(extension_name, available_instance_extension->extensionName) == 0)
			{
				found = true;
				break;
			}

			available_instance_extension++;
		}

		hard_assert_true(found, fprintf(stderr, "XR instance extension %s not found\n", extension_name));
		printf("+ %s\n", extension_name);
		*(extension++) = cstr_init(extension_name);
	}

	return extension;
}


private const char**
xr_xr_get_instance_layers(
	xr_t xr,
	const char** layer
	)
{
	assert_not_null(xr);
	assert_not_null(layer);

	uint32_t available_instance_layer_count = 0;
	XrResult result = xrEnumerateApiLayerProperties(0, &available_instance_layer_count, NULL);
	hard_assert_eq(result, XR_SUCCESS);

	XrApiLayerProperties available_instance_layers[available_instance_layer_count];

	XrApiLayerProperties* available_instance_layer = available_instance_layers;
	XrApiLayerProperties* available_instance_layer_end = available_instance_layer + available_instance_layer_count;

	while(available_instance_layer < available_instance_layer_end)
	{
		*(available_instance_layer++) = (XrApiLayerProperties){XR_TYPE_API_LAYER_PROPERTIES};
	}

	result = xrEnumerateApiLayerProperties(available_instance_layer_count,
		&available_instance_layer_count, available_instance_layers);
	hard_assert_eq(result, XR_SUCCESS);

	puts("\nXR instance layers:");

	for(
		available_instance_layer = available_instance_layers;
		available_instance_layer < available_instance_layer_end;
		available_instance_layer++
		)
	{
		printf("- %s\n", available_instance_layer->layerName);
	}

	puts("");

	const char* const* instance_layer = xr_xr_instance_layers;
	const char* const* instance_layer_end = instance_layer + MACRO_ARRAY_LEN(xr_xr_instance_layers);

	while(instance_layer < instance_layer_end)
	{
		bool found = false;
		const char* layer_name = *(instance_layer++);

		available_instance_layer = available_instance_layers;
		while(available_instance_layer < available_instance_layer_end)
		{
			if(strcmp(layer_name, available_instance_layer->layerName) == 0)
			{
				found = true;
				break;
			}

			available_instance_layer++;
		}

		hard_assert_true(found, fprintf(stderr, "XR instance layer %s not found\n", layer_name));
		printf("+ %s\n", layer_name);
		*(layer++) = cstr_init(layer_name);
	}

	return layer;
}


private void
xr_init_xr_instance(
	xr_t xr
	)
{
	assert_not_null(xr);

	const char* instance_extensions[64];
	const char** instance_extension = xr_xr_get_instance_extensions(xr, instance_extensions);
	assert_lt(instance_extension, instance_extensions + MACRO_ARRAY_LEN(instance_extensions));

	const char* instance_layers[64];
	const char** instance_layer = xr_xr_get_instance_layers(xr, instance_layers);
	assert_lt(instance_layer, instance_layers + MACRO_ARRAY_LEN(instance_layers));

	XrInstanceCreateInfo instance_info =
	{
		.type = XR_TYPE_INSTANCE_CREATE_INFO,
		.next = NULL,
		.createFlags = 0,
		.applicationInfo =
		{
			.apiVersion = XR_CURRENT_API_VERSION
		},
		.enabledExtensionCount = instance_extension - instance_extensions,
		.enabledExtensionNames = instance_extensions,
		.enabledApiLayerCount = instance_layer - instance_layers,
		.enabledApiLayerNames = instance_layers,
	};

	strcpy(instance_info.applicationInfo.applicationName, "Thesis");

#ifndef NDEBUG
	XrDebugUtilsMessengerCreateInfoEXT debug_info =
	{
		.type = XR_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
		.next = NULL,
		.messageSeverities =
			XR_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
			XR_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
			XR_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			XR_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
		.messageTypes =
			XR_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
			XR_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
			XR_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
			XR_DEBUG_UTILS_MESSAGE_TYPE_CONFORMANCE_BIT_EXT,
		.userCallback = xr_xr_debug_callback,
		.userData = NULL
	};

	instance_info.next = &debug_info;
#endif

	XrResult result = xrCreateInstance(&instance_info, &xr->instance);
	hard_assert_eq(result, XR_SUCCESS);

	shared_free_str_array(instance_extensions, instance_extension);
	shared_free_str_array(instance_layers, instance_layer);

#ifndef NDEBUG
	PFN_xrCreateDebugUtilsMessengerEXT xrCreateDebugUtilsMessengerEXT =
		xr_xr_load_func(xr, "xrCreateDebugUtilsMessengerEXT");

	result = xrCreateDebugUtilsMessengerEXT(xr->instance, &debug_info, &xr->debug_messenger);
	hard_assert_eq(result, XR_SUCCESS);
#endif

	XrInstanceProperties instance_properties = {XR_TYPE_INSTANCE_PROPERTIES};
	result = xrGetInstanceProperties(xr->instance, &instance_properties);
	hard_assert_eq(result, XR_SUCCESS);

	printf(
		"\nXR runtime: '%s' ver. %u.%u.%u\n",
		instance_properties.runtimeName,
		XR_VERSION_MAJOR(instance_properties.runtimeVersion),
		XR_VERSION_MINOR(instance_properties.runtimeVersion),
		XR_VERSION_PATCH(instance_properties.runtimeVersion)
		);

	XrSystemGetInfo system_info =
	{
		.type = XR_TYPE_SYSTEM_GET_INFO,
		.next = NULL,
		.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY
	};
	result = xrGetSystem(xr->instance, &system_info, &xr->system);
	hard_assert_eq(result, XR_SUCCESS);

	XrSystemHandTrackingPropertiesEXT hand_tracking_properties =
	{
		.type = XR_TYPE_SYSTEM_HAND_TRACKING_PROPERTIES_EXT,
		.next = NULL,
		.supportsHandTracking = false
	};

	XrSystemProperties system_properties =
	{
		.type = XR_TYPE_SYSTEM_PROPERTIES,
		.next = &hand_tracking_properties
	};

	result = xrGetSystemProperties(xr->instance, xr->system, &system_properties);
	hard_assert_eq(result, XR_SUCCESS);

	hard_assert_true(hand_tracking_properties.supportsHandTracking);

	printf("\nXR system: %s\n", system_properties.systemName);
	printf("XR system vendor: %u\n", system_properties.vendorId);
	printf(
		"XR system properties:\n"
		"\tmaxSwapchainImageHeight: %u\n"
		"\tmaxSwapchainImageWidth: %u\n"
		"\tmaxLayerCount: %u\n"
		"\torientationTracking: %d\n"
		"\tpositionTracking: %d\n",
		system_properties.graphicsProperties.maxSwapchainImageHeight,
		system_properties.graphicsProperties.maxSwapchainImageWidth,
		system_properties.graphicsProperties.maxLayerCount,
		system_properties.trackingProperties.orientationTracking,
		system_properties.trackingProperties.positionTracking
		);
}


private void
xr_free_xr_instance(
	xr_t xr
	)
{
	assert_not_null(xr);

#ifndef NDEBUG
	PFN_xrDestroyDebugUtilsMessengerEXT xrDestroyDebugUtilsMessengerEXT =
		xr_xr_load_func(xr, "xrDestroyDebugUtilsMessengerEXT");

	XrResult result = xrDestroyDebugUtilsMessengerEXT(xr->debug_messenger);
	hard_assert_eq(result, XR_SUCCESS);
#endif

	xrDestroyInstance(xr->instance);
}


private const char**
xr_vk_get_instance_extensions(
	xr_t xr,
	const char** extension
	)
{
	assert_not_null(xr);
	assert_not_null(extension);

	uint32_t available_instance_extension_count = 0;
	VkResult result = vkEnumerateInstanceExtensionProperties(NULL, &available_instance_extension_count, NULL);
	hard_assert_eq(result, VK_SUCCESS);

	VkExtensionProperties available_instance_extensions[available_instance_extension_count];

	VkExtensionProperties* available_instance_extension = available_instance_extensions;
	VkExtensionProperties* available_instance_extension_end =
		available_instance_extension + available_instance_extension_count;

	while(available_instance_extension < available_instance_extension_end)
	{
		*(available_instance_extension++) = (VkExtensionProperties){0};
	}

	result = vkEnumerateInstanceExtensionProperties(NULL,
		&available_instance_extension_count, available_instance_extensions);
	hard_assert_eq(result, VK_SUCCESS);

	puts("\nVK instance extensions:");

	for(
		available_instance_extension = available_instance_extensions;
		available_instance_extension < available_instance_extension_end;
		available_instance_extension++
		)
	{
		printf("- %s\n", available_instance_extension->extensionName);
	}

	puts("");

	const char* const* instance_extension = xr_vk_instance_extensions;
	const char* const* instance_extension_end = instance_extension + MACRO_ARRAY_LEN(xr_vk_instance_extensions);

	while(instance_extension < instance_extension_end)
	{
		bool found = false;
		const char* extension_name = *(instance_extension++);

		available_instance_extension = available_instance_extensions;
		while(available_instance_extension < available_instance_extension_end)
		{
			if(strcmp(extension_name, available_instance_extension->extensionName) == 0)
			{
				found = true;
				break;
			}

			available_instance_extension++;
		}

		hard_assert_true(found, fprintf(stderr, "VK instance extension %s not found\n", extension_name));
		printf("+ %s\n", extension_name);
		*(extension++) = cstr_init(extension_name);
	}

	return extension;
}


private const char**
xr_vk_get_instance_layers(
	xr_t xr,
	const char** layer
	)
{
	assert_not_null(xr);
	assert_not_null(layer);

	uint32_t available_instance_layer_count = 0;
	VkResult result = vkEnumerateInstanceLayerProperties(&available_instance_layer_count, NULL);
	hard_assert_eq(result, VK_SUCCESS);

	VkLayerProperties available_instance_layers[available_instance_layer_count];

	VkLayerProperties* available_instance_layer = available_instance_layers;
	VkLayerProperties* available_instance_layer_end = available_instance_layer + available_instance_layer_count;

	while(available_instance_layer < available_instance_layer_end)
	{
		*(available_instance_layer++) = (VkLayerProperties){0};
	}

	result = vkEnumerateInstanceLayerProperties(&available_instance_layer_count, available_instance_layers);
	hard_assert_eq(result, VK_SUCCESS);

	puts("\nVK instance layers:");

	for(
		available_instance_layer = available_instance_layers;
		available_instance_layer < available_instance_layer_end;
		available_instance_layer++
		)
	{
		printf("- %s\n", available_instance_layer->layerName);
	}

	puts("");

	const char* const* instance_layer = xr_vk_instance_layers;
	const char* const* instance_layer_end = instance_layer + MACRO_ARRAY_LEN(xr_vk_instance_layers);

	while(instance_layer < instance_layer_end)
	{
		bool found = false;
		const char* layer_name = *(instance_layer++);

		available_instance_layer = available_instance_layers;
		while(available_instance_layer < available_instance_layer_end)
		{
			if(strcmp(layer_name, available_instance_layer->layerName) == 0)
			{
				found = true;
				break;
			}

			available_instance_layer++;
		}

		hard_assert_true(found, fprintf(stderr, "VK instance layer %s not found\n", layer_name));
		printf("+ %s\n", layer_name);
		*(layer++) = cstr_init(layer_name);
	}

	return layer;
}


private void
xr_init_vk_instance(
	xr_t xr
	)
{
	assert_not_null(xr);

	VkResult vk_result = volkInitialize();
	hard_assert_eq(vk_result, VK_SUCCESS);

	hard_assert_not_null(vkGetInstanceProcAddr);
	xr->vk.proc_addr_fn = vkGetInstanceProcAddr;

	XrGraphicsRequirementsVulkanKHR requirements =
	{
		.type = XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR,
		.next = NULL,
		.minApiVersionSupported = XR_MAKE_VERSION(1, 4, 309),
		.maxApiVersionSupported = XR_MAKE_VERSION(1, 4, 309)
	};

	PFN_xrGetVulkanGraphicsRequirementsKHR xrGetVulkanGraphicsRequirementsKHR =
		xr_xr_load_func(xr, "xrGetVulkanGraphicsRequirementsKHR");

	XrResult result = xrGetVulkanGraphicsRequirementsKHR(xr->instance, xr->system, &requirements);
	hard_assert_eq(result, XR_SUCCESS);

	const char* instance_extensions[64];
	const char** instance_extension = xr_vk_get_instance_extensions(xr, instance_extensions);
	assert_lt(instance_extension, instance_extensions + MACRO_ARRAY_LEN(instance_extensions));

	const char* instance_layers[64];
	const char** instance_layer = xr_vk_get_instance_layers(xr, instance_layers);
	assert_lt(instance_layer, instance_layers + MACRO_ARRAY_LEN(instance_layers));

	VkApplicationInfo application_info =
	{
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pNext = NULL,
		.pApplicationName = "Thesis",
		.apiVersion = VK_API_VERSION_1_0
	};

	VkInstanceCreateInfo instance_info =
	{
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.pApplicationInfo = &application_info,
		.enabledLayerCount = instance_layer - instance_layers,
		.ppEnabledLayerNames = instance_layers,
		.enabledExtensionCount = instance_extension - instance_extensions,
		.ppEnabledExtensionNames = instance_extensions
	};

#ifndef NDEBUG
	VkDebugUtilsMessengerCreateInfoEXT debug_info =
	{
		.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
		.pNext = NULL,
		.messageSeverity =
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
		.messageType =
			VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
		.pfnUserCallback = xr_vk_debug_callback,
		.pUserData = NULL
	};

	instance_info.pNext = &debug_info;
#endif

	XrVulkanInstanceCreateInfoKHR xr_instance_info =
	{
		.type = XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR,
		.next = NULL,
		.systemId = xr->system,
		.createFlags = 0,
		.pfnGetInstanceProcAddr = xr->vk.proc_addr_fn,
		.vulkanCreateInfo = &instance_info,
		.vulkanAllocator = NULL
	};

	PFN_xrCreateVulkanInstanceKHR xrCreateVulkanInstanceKHR =
		xr_xr_load_func(xr, "xrCreateVulkanInstanceKHR");

	result = xrCreateVulkanInstanceKHR(xr->instance, &xr_instance_info, &xr->vk.instance, &vk_result);
	hard_assert_eq(result, XR_SUCCESS);
	hard_assert_eq(vk_result, VK_SUCCESS);

	shared_free_str_array(instance_extensions, instance_extension);
	shared_free_str_array(instance_layers, instance_layer);

#ifndef NDEBUG
	PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT =
		xr_vk_load_func(xr, "vkCreateDebugUtilsMessengerEXT");

	vk_result = vkCreateDebugUtilsMessengerEXT(xr->vk.instance, &debug_info, NULL, &xr->vk.debug_messenger);
	hard_assert_eq(vk_result, VK_SUCCESS);
#endif

	volkLoadInstanceOnly(xr->vk.instance);
}


private void
xr_free_vk_instance(
	xr_t xr
	)
{
	assert_not_null(xr);

#ifndef NDEBUG
	/* Volk loaded the function already */
	vkDestroyDebugUtilsMessengerEXT(xr->vk.instance, xr->vk.debug_messenger, NULL);
#endif

	vkDestroyInstance(xr->vk.instance, NULL);

	volkFinalize();
}


private void
xr_init_vk_physical_device(
	xr_t xr
	)
{
	assert_not_null(xr);

	XrVulkanGraphicsDeviceGetInfoKHR xr_vk_device_info =
	{
		.type = XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR,
		.next = NULL,
		.systemId = xr->system,
		.vulkanInstance = xr->vk.instance
	};

	PFN_xrGetVulkanGraphicsDevice2KHR xrGetVulkanGraphicsDevice2KHR =
		xr_xr_load_func(xr, "xrGetVulkanGraphicsDevice2KHR");

	XrResult xr_result = xrGetVulkanGraphicsDevice2KHR(
		xr->instance, &xr_vk_device_info, &xr->vk.physical_device);
	assert_eq(xr_result, XR_SUCCESS);

	uint32_t vk_queue_family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(
		xr->vk.physical_device, &vk_queue_family_count, NULL);
	assert_gt(vk_queue_family_count, 0);

	VkQueueFamilyProperties vk_queue_family_properties[vk_queue_family_count];

	vkGetPhysicalDeviceQueueFamilyProperties(
		xr->vk.physical_device, &vk_queue_family_count, vk_queue_family_properties);
	assert_gt(vk_queue_family_count, 0);

	VkQueueFamilyProperties* vk_queue_family_property = vk_queue_family_properties;
	VkQueueFamilyProperties* vk_queue_family_property_end =
		vk_queue_family_property + vk_queue_family_count;

	while(vk_queue_family_property < vk_queue_family_property_end)
	{
		if(vk_queue_family_property->queueFlags & VK_QUEUE_GRAPHICS_BIT)
		{
			break;
		}

		vk_queue_family_property++;
	}

	assert_lt(vk_queue_family_property, vk_queue_family_property_end);
	xr->vk.queue_id = vk_queue_family_property - vk_queue_family_properties;
}


private void
xr_free_vk_physical_device(
	xr_t xr
	)
{
	assert_not_null(xr);

	/* Physical devices are not freed */
}


private const char**
xr_vk_get_device_extensions(
	xr_t xr,
	const char** extension
	)
{
	assert_not_null(xr);
	assert_not_null(extension);

	uint32_t vk_device_extension_count = 0;
	vkEnumerateDeviceExtensionProperties(
		xr->vk.physical_device, NULL, &vk_device_extension_count, NULL);

	VkExtensionProperties vk_device_extensions[vk_device_extension_count];

	VkExtensionProperties* vk_device_extension = vk_device_extensions;
	VkExtensionProperties* vk_device_extension_end =
		vk_device_extension + vk_device_extension_count;

	vkEnumerateDeviceExtensionProperties(xr->vk.physical_device,
		NULL, &vk_device_extension_count, vk_device_extensions);

	puts("VK device extensions:");

	for(
		vk_device_extension = vk_device_extensions;
		vk_device_extension < vk_device_extension_end;
		vk_device_extension++
		)
	{
		printf("- %s\n", vk_device_extension->extensionName);
	}

	puts("");

	const char* const* xr_vk_device_extension = xr_vk_device_extensions;
	const char* const* xr_vk_device_extension_end =
		xr_vk_device_extension + MACRO_ARRAY_LEN(xr_vk_device_extensions);

	while(xr_vk_device_extension < xr_vk_device_extension_end)
	{
		bool found = false;
		const char* extension_name = *(xr_vk_device_extension++);

		vk_device_extension = vk_device_extensions;
		while(vk_device_extension < vk_device_extension_end)
		{
			if(strcmp(extension_name, vk_device_extension->extensionName) == 0)
			{
				found = true;
				break;
			}

			vk_device_extension++;
		}

		assert_true(found, fprintf(stderr, "VK device extension %s not found\n", extension_name));
		*(extension++) = strdup(extension_name);
	}

	uint32_t xr_vk_device_extension_count = 0;

	PFN_xrGetVulkanDeviceExtensionsKHR xrGetVulkanDeviceExtensionsKHR =
		xr_xr_load_func(xr, "xrGetVulkanDeviceExtensionsKHR");

	XrResult xr_result = xrGetVulkanDeviceExtensionsKHR(xr->instance,
		xr->system, 0, &xr_vk_device_extension_count, NULL);
	assert_eq(xr_result, XR_SUCCESS);

	char xr_vk_device_extensions[xr_vk_device_extension_count + 1];
	xr_vk_device_extensions[xr_vk_device_extension_count] = '\0';

	xr_result = xrGetVulkanDeviceExtensionsKHR(xr->instance, xr->system,
		xr_vk_device_extension_count, &xr_vk_device_extension_count, xr_vk_device_extensions);
	assert_eq(xr_result, XR_SUCCESS);

	char* strtok_r_state = NULL;
	const char* extension_name = strtok_r(xr_vk_device_extensions, " ", &strtok_r_state);

	while(extension_name)
	{
		bool found = false;

		vk_device_extension = vk_device_extensions;
		while(vk_device_extension < vk_device_extension_end)
		{
			if(strcmp(extension_name, vk_device_extension->extensionName) == 0)
			{
				found = true;
				break;
			}

			vk_device_extension++;
		}

		assert_true(found, fprintf(stderr, "XR VK device extension %s not found\n", extension_name));
		*(extension++) = strdup(extension_name);

		extension_name = strtok_r(NULL, " ", &strtok_r_state);
	}

	return extension;
}


private const char**
xr_vk_get_device_layers(
	xr_t xr,
	const char** layer
	)
{
	assert_not_null(xr);
	assert_not_null(layer);

	uint32_t vk_device_layer_count = 0;
	vkEnumerateDeviceLayerProperties(
		xr->vk.physical_device, &vk_device_layer_count, NULL);

	VkLayerProperties vk_device_layers[vk_device_layer_count];

	VkLayerProperties* vk_device_layer = vk_device_layers;
	VkLayerProperties* vk_device_layer_end = vk_device_layer + vk_device_layer_count;

	vkEnumerateDeviceLayerProperties(xr->vk.physical_device,
		&vk_device_layer_count, vk_device_layers);

	puts("VK device layers:");

	for(
		vk_device_layer = vk_device_layers;
		vk_device_layer < vk_device_layer_end;
		vk_device_layer++
		)
	{
		printf("- %s\n", vk_device_layer->layerName);
	}

	puts("");

	const char* const* xr_vk_device_layer = xr_vk_device_layers;
	const char* const* xr_vk_device_layer_end =
		xr_vk_device_layer + MACRO_ARRAY_LEN(xr_vk_device_layers);

	while(xr_vk_device_layer < xr_vk_device_layer_end)
	{
		bool found = false;
		const char* layer_name = *(xr_vk_device_layer++);

		vk_device_layer = vk_device_layers;
		while(vk_device_layer < vk_device_layer_end)
		{
			if(strcmp(layer_name, vk_device_layer->layerName) == 0)
			{
				found = true;
				break;
			}

			vk_device_layer++;
		}

		assert_true(found, fprintf(stderr, "VK device layer %s not found\n", layer_name));
		*(layer++) = strdup(layer_name);
	}

	return layer;
}


private void
xr_init_vk_logical_device(
	xr_t xr
	)
{
	assert_not_null(xr);

	const char* vk_device_extensions[64];
	const char** vk_device_extension =
		xr_vk_get_device_extensions(xr, vk_device_extensions);
	assert_lt(vk_device_extension,
		vk_device_extensions + MACRO_ARRAY_LEN(vk_device_extensions));

	const char* vk_device_layers[64];
	const char** vk_device_layer =
		xr_vk_get_device_layers(xr, vk_device_layers);
	assert_lt(vk_device_layer,
		vk_device_layers + MACRO_ARRAY_LEN(vk_device_layers));

	float vk_queue_priority = 1.0f;

	VkDeviceQueueCreateInfo vk_queue_info =
	{
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.queueFamilyIndex = xr->vk.queue_id,
		.queueCount = 1,
		.pQueuePriorities = &vk_queue_priority
	};

	VkPhysicalDeviceMultiviewFeatures vk_multiview_features =
	{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES,
		.pNext = NULL,
		.multiview = true,
		.multiviewGeometryShader = false,
		.multiviewTessellationShader = false
	};

	VkPhysicalDeviceFeatures2 vk_device_features2 =
	{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = &vk_multiview_features,
		.features = {0}
	};

	VkDeviceCreateInfo vk_device_info =
	{
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext = &vk_device_features2,
		.flags = 0,
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &vk_queue_info,
		.enabledLayerCount = vk_device_layer - vk_device_layers,
		.ppEnabledLayerNames = vk_device_layers,
		.enabledExtensionCount = vk_device_extension - vk_device_extensions,
		.ppEnabledExtensionNames = vk_device_extensions,
		.pEnabledFeatures = NULL
	};

	XrVulkanDeviceCreateInfoKHR xr_vk_device_info =
	{
		.type = XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR,
		.next = NULL,
		.systemId = xr->system,
		.createFlags = 0,
		.pfnGetInstanceProcAddr = xr->vk.proc_addr_fn,
		.vulkanPhysicalDevice = xr->vk.physical_device,
		.vulkanCreateInfo = &vk_device_info,
		.vulkanAllocator = NULL
	};

	PFN_xrCreateVulkanDeviceKHR xrCreateVulkanDeviceKHR =
		xr_xr_load_func(xr, "xrCreateVulkanDeviceKHR");

	VkResult vk_result;
	XrResult xr_result = xrCreateVulkanDeviceKHR(
		xr->instance, &xr_vk_device_info, &xr->vk.device, &vk_result);
	assert_eq(xr_result, XR_SUCCESS);
	assert_eq(vk_result, VK_SUCCESS);

	shared_free_str_array(vk_device_extensions, vk_device_extension);
	shared_free_str_array(vk_device_layers, vk_device_layer);

	volkLoadDeviceTable(&xr->vk.table, xr->vk.device);

	xr->vk.table.vkGetDeviceQueue(xr->vk.device,
		xr->vk.queue_id, 0, &xr->vk.queue);

	// vkGetPhysicalDeviceMemoryProperties(
	// 	xr->vk.physical_device, &xr->vk.memory_properties);
}


private void
xr_free_vk_logical_device(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr->vk.table.vkDestroyDevice(xr->vk.device, NULL);
}


private void
xr_init_xr_session(
	xr_t xr
	)
{
	assert_not_null(xr);

	XrGraphicsBindingVulkanKHR xr_vk_binding =
	{
		.type = XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR,
		.next = NULL,
		.instance = xr->vk.instance,
		.physicalDevice = xr->vk.physical_device,
		.device = xr->vk.device,
		.queueFamilyIndex = xr->vk.queue_id,
		.queueIndex = 0
	};

	XrSessionCreateInfo xr_session_info =
	{
		.type = XR_TYPE_SESSION_CREATE_INFO,
		.next = &xr_vk_binding,
		.createFlags = 0,
		.systemId = xr->system
	};

	XrResult xr_result = xrCreateSession(
		xr->instance, &xr_session_info, &xr->session);
	assert_eq(xr_result, XR_SUCCESS);
}


private void
xr_free_xr_session(
	xr_t xr
	)
{
	assert_not_null(xr);

	XrResult xr_result = xrDestroySession(xr->session);
	assert_eq(xr_result, XR_SUCCESS);
}


private void
xr_init_xr(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_init_xr_instance(xr);
	xr_init_vk_instance(xr);
	xr_init_vk_physical_device(xr);
	xr_init_vk_logical_device(xr);
	xr_init_xr_session(xr);
}


private void
xr_free_xr(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_xr_session(xr);
	xr_free_vk_logical_device(xr);
	xr_free_vk_physical_device(xr);
	xr_free_vk_instance(xr);
	xr_free_xr_instance(xr);
}


private void
xr_free(
	xr_t xr,
	simulation_free_event_data_t* event_data
	)
{
	assert_not_null(xr);

	xr_free_xr(xr);

	alloc_free(xr, sizeof(*xr));
}


xr_t
xr_init(
	simulation_t simulation
	)
{
	assert_not_null(simulation);

	xr_t xr = alloc_calloc(sizeof(*xr));
	assert_ptr(xr, sizeof(*xr));

	xr_init_options(xr);

	xr->simulation = simulation;

	simulation_event_table_t* table = simulation_get_event_table(xr->simulation);

	event_listener_data_t free_data =
	{
		.fn = (void*) xr_free,
		.data = xr
	};
	event_target_once(&table->free_target, free_data);

	xr_init_xr(xr);

	return xr;
}


