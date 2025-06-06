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

#include <thesis/vk.h>
#include <thesis/file.h>
#include <thesis/debug.h>
#include <thesis/shared.h>
#include <thesis/window.h>
#include <thesis/options.h>
#include <thesis/threads.h>
#include <thesis/alloc_ext.h>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <volk.h>

#include <signal.h>
#include <string.h>
#include <stdatomic.h>

#define VK_MAX_FRAMES 8
#define VK_MAX_INSTANCES 128

#define VK_WINDOW_WIDTH 1280
#define VK_WINDOW_HEIGHT 720
#define VK_WINDOW_SENSITIVITY 0.005f
#define VK_WINDOW_SPEED 500.0f


typedef enum vk_image_type
{
	VK_IMAGE_TYPE_DEPTH_STENCIL,
	VK_IMAGE_TYPE_MULTISAMPLED,
	VK_IMAGE_TYPE_DEPTH_MAP,
	VK_IMAGE_TYPE_TEXTURE_2D,
	VK_IMAGE_TYPE_TEXTURE_CUBE,
	MACRO_ENUM_BITS(VK_IMAGE_TYPE)
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

typedef struct vk_depth_vert_constant_data
{
	mat4 transform;
}
vk_depth_vert_constant_data_t;

typedef struct vk_depth_vertex_data
{
	vec3 position;
}
vk_depth_vertex_data_t;

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

typedef struct vk_mesh_vert_ubo_data
{
	mat4 projection;
	mat4 view;
	mat4 light_transform;
	vec4 light_position;
}
vk_mesh_vert_ubo_data_t;

typedef struct vk_mesh_frag_constant_data
{
	vec4 diffuse;
	vec4 ambient;
}
vk_mesh_frag_constant_data_t;

typedef struct vk_mesh_vertex_data
{
	vec3 position;
	vec3 normal;
	vec2 coords;
}
vk_mesh_vertex_data_t;

typedef struct vk_material
{
	vk_image_t texture;
	vec4 diffuse;
	vec4 ambient;

	VkDescriptorSet set;
}
vk_material_t;

typedef struct vk_mesh
{
	uint32_t material_idx;
	uint32_t vertex_count;
	uint32_t index_count;

	vk_buffer_t depth_vertex_buffer;
	vk_buffer_t mesh_vertex_buffer;
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

typedef struct vk_frame
{
	struct
	{
		VkFramebuffer framebuffer;
		vk_image_t image;
		VkDescriptorSet set;
	}
	shadow;

	struct
	{
		VkFramebuffer framebuffer;
		VkImageView image_view;
		VkImage image;
	}
	scene;
}
vk_frame_t;

typedef enum vk_barrier_semaphore
{
	VK_BARRIER_SEMAPHORE_IMAGE_AVAILABLE,
	VK_BARRIER_SEMAPHORE_RENDER_FINISHED,
	MACRO_ENUM_BITS(VK_BARRIER_SEMAPHORE)
}
vk_barrier_semaphore_t;

typedef enum vk_barrier_fence
{
	VK_BARRIER_FENCE_IN_FLIGHT,
	MACRO_ENUM_BITS(VK_BARRIER_FENCE)
}
vk_barrier_fence_t;

typedef struct vk_barrier
{
	VkSemaphore semaphores[VK_BARRIER_SEMAPHORE__COUNT];
	VkFence fences[VK_BARRIER_FENCE__COUNT];
	VkCommandBuffer command_buffer;
}
vk_barrier_t;

struct vk
{
	simulation_t simulation;

	event_listener_t* window_close_once_listener;
	event_listener_t* window_resize_listener;
	event_listener_t* window_mouse_down_listener;
	event_listener_t* window_mouse_up_listener;
	event_listener_t* window_mouse_move_listener;
	event_listener_t* window_key_down_listener;
	event_listener_t* window_key_up_listener;

	float window_default_width;
	float window_default_height;

	window_manager_t window_manager;
	window_t window;
	thread_t window_thread;

	_Atomic bool window_resize_bool;
	sync_mtx_t window_resize_mtx;
	sync_cond_t window_resize_cond;

	bool window_mouse_holding;

	bool vk_sample_shading;
	uint32_t vk_mipmap_levels;
	uint32_t vk_depth_map_size;
	bool vk_preview_depth_map;

	thread_t vk_thread;
	_Atomic bool vk_running;

	PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;

#ifndef NDEBUG
	VkDebugUtilsMessengerEXT vk_debug_messenger;
#endif

	VkInstance vk_instance;

	VkSurfaceKHR vk_surface;
	VkSurfaceCapabilitiesKHR vk_surface_capabilities;

	uint32_t vk_queue_id;
	VkFormat vk_format;
	VkSampleCountFlagBits vk_samples;
	float vk_anisotropy;
	VkPhysicalDeviceLimits vk_limits;

	VkPhysicalDevice vk_physical_device;
	VkDevice vk_device;

	struct VolkDeviceTable vk_table;

	VkQueue vk_queue;
	VkPhysicalDeviceMemoryProperties vk_memory_properties;

	VkExtent2D vk_extent;
	VkViewport vk_viewport;
	VkRect2D vk_scissor;
	uint32_t vk_image_count;
	VkSurfaceTransformFlagBitsKHR vk_transform;
	VkPresentModeKHR vk_present_mode;

	VkCommandPool vk_command_pool;
	VkCommandBuffer vk_command_buffer;
	VkFence vk_fence;

	vk_buffer_t vk_staging_buffer;

	VkDescriptorPool vk_descriptor_pool;

	VkSampler vk_depth_sampler;
	VkSampler vk_image_sampler;

	struct
	{
		VkRenderPass render_pass;

		struct
		{
			VkPipelineLayout pipeline_layout;
			VkPipeline pipeline;
		}
		depth;
	}
	vk_shadow;

	struct
	{
		vk_image_t depth_image;
		vk_image_t multisampled_image;
		VkRenderPass render_pass;

		struct
		{
			VkDescriptorSetLayout set_layout;
			VkPipelineLayout pipeline_layout;
			VkPipeline pipeline;

			vk_image_t image;
			VkDescriptorSet set;

			vk_buffer_t vertex_buffer;
			vk_buffer_t index_buffer;
		}
		skybox;

		struct
		{
			VkDescriptorSetLayout ubo_set_layout;
			VkDescriptorSetLayout texture_set_layout;
			VkDescriptorSetLayout depth_map_set_layout;
			VkPipelineLayout pipeline_layout;
			VkPipeline pipeline;

			vk_buffer_t ubo_buffer;
			VkDescriptorSet ubo_set;
		}
		mesh;

		struct
		{
			VkDescriptorSetLayout set_layout;
			VkPipelineLayout pipeline_layout;
			VkPipeline pipeline;
		}
		depth_map;
	}
	vk_scene;

	vk_material_t* vk_materials;
	vk_model_t* vk_models;
	uint32_t vk_material_count;
	uint32_t vk_model_count;

	vk_frame_t vk_frames[VK_MAX_FRAMES];
	vk_barrier_t vk_barriers[VK_MAX_FRAMES];
	vk_barrier_t* vk_barrier;

	VkSwapchainKHR vk_swapchain;
};


private const char* vk_instance_extensions[] =
{
#ifndef NDEBUG
	VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
#endif
};

private const char* vk_instance_layers[] =
{
#ifndef NDEBUG
	"VK_LAYER_KHRONOS_validation"
#endif
};

private const char* vk_device_extensions[] =
{
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

private const char* vk_device_layers[] =
{
#ifndef NDEBUG
	"VK_LAYER_KHRONOS_validation"
#endif
};


private vk_skybox_vertex_data_t vk_skybox_vertex_data[] =
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

private uint16_t vk_skybox_index_data[] =
{
	0, 1, 2, 2, 1, 3,
	4, 6, 5, 5, 6, 7,
	0, 4, 1, 1, 4, 5,
	2, 3, 6, 6, 3, 7,
	0, 2, 4, 4, 2, 6,
	1, 5, 3, 3, 5, 7
};


private void
vk_init_options(
	vk_t vk
	)
{
	assert_not_null(vk);

	puts("\nVK options:");

	str_t window_width = options_get(global_options, "window_width");
	if(window_width && !str_is_empty(window_width))
	{
		float window_width_value = strtof(window_width->str, NULL);
		if(window_width_value <= 0.0f)
		{
			window_width_value = VK_WINDOW_WIDTH;
		}

		vk->window_default_width = window_width_value;
	}
	else
	{
		vk->window_default_width = VK_WINDOW_WIDTH;
	}
	printf("- window_width: %.0f\n", vk->window_default_width);

	str_t window_height = options_get(global_options, "window_height");
	if(window_height && !str_is_empty(window_height))
	{
		float window_height_value = strtof(window_height->str, NULL);
		if(window_height_value <= 0.0f)
		{
			window_height_value = VK_WINDOW_HEIGHT;
		}

		vk->window_default_height = window_height_value;
	}
	else
	{
		vk->window_default_height = VK_WINDOW_HEIGHT;
	}
	printf("- window_height: %.0f\n", vk->window_default_height);

	str_t sample_shading = options_get(global_options, "vk_sample_shading");
	vk->vk_sample_shading = sample_shading && str_case_cmp_len(sample_shading, "true", 4);
	printf("- vk_sample_shading: %d\n", vk->vk_sample_shading);

	str_t mipmap_levels = options_get(global_options, "vk_mipmap_levels");
	vk->vk_mipmap_levels = mipmap_levels && !str_is_empty(mipmap_levels) ?
		strtoul(mipmap_levels->str, NULL, 10) : 3;
	printf("- vk_mipmap_levels: %u\n", vk->vk_mipmap_levels);

	str_t anisotropy = options_get(global_options, "vk_anisotropy");
	if(anisotropy && !str_is_empty(anisotropy))
	{
		float anisotropy_value = strtof(anisotropy->str, NULL);
		if(anisotropy_value <= 0.0f)
		{
			anisotropy_value = 0.0f;
		}

		vk->vk_anisotropy = anisotropy_value;
	}
	else
	{
		vk->vk_anisotropy = 100.0f;
	}
	printf("- vk_anisotropy: %.1f\n", vk->vk_anisotropy);

	str_t depth_map_size = options_get(global_options, "vk_depth_map_size");
	if(depth_map_size && !str_is_empty(depth_map_size))
	{
		uint32_t depth_map_size_value = strtoul(depth_map_size->str, NULL, 10);
		if(depth_map_size_value <= 0)
		{
			depth_map_size_value = 4096;
		}

		vk->vk_depth_map_size = depth_map_size_value;
	}
	else
	{
		vk->vk_depth_map_size = 4096;
	}
	printf("- vk_depth_map_size: %u\n", vk->vk_depth_map_size);

	str_t preview_depth_map = options_get(global_options, "vk_preview_depth_map");
	if(preview_depth_map && !str_is_empty(preview_depth_map))
	{
		vk->vk_preview_depth_map = str_case_cmp_len(preview_depth_map, "true", 4);
	}
	else
	{
		vk->vk_preview_depth_map = false;
	}
	printf("- vk_preview_depth_map: %d\n", vk->vk_preview_depth_map);
}


#ifndef NDEBUG

private VKAPI_ATTR VkBool32 VKAPI_CALL
vk_debug_callback(
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


private void* assert_used
vk_load_func(
	vk_t vk,
	const char* name
	)
{
	assert_not_null(vk);
	assert_not_null(name);

	void* func = vk->vkGetInstanceProcAddr(vk->vk_instance, name);
	hard_assert_not_null(func, fprintf(stderr, "VK function %s not found\n", name));

	return func;
}


private const char**
vk_get_instance_extensions(
	vk_t vk,
	const char** extension
	)
{
	assert_not_null(vk);
	assert_not_null(extension);

	uint32_t vk_available_instance_extension_count = 0;
	VkResult vk_result = vkEnumerateInstanceExtensionProperties(
		NULL, &vk_available_instance_extension_count, NULL);
	hard_assert_eq(vk_result, VK_SUCCESS);

	VkExtensionProperties vk_available_instance_extensions[vk_available_instance_extension_count];

	VkExtensionProperties* vk_available_instance_extension = vk_available_instance_extensions;
	VkExtensionProperties* vk_available_instance_extension_end =
		vk_available_instance_extension + vk_available_instance_extension_count;

	while(vk_available_instance_extension < vk_available_instance_extension_end)
	{
		*(vk_available_instance_extension++) = (VkExtensionProperties){0};
	}

	vk_result = vkEnumerateInstanceExtensionProperties(
		NULL, &vk_available_instance_extension_count, vk_available_instance_extensions);
	hard_assert_eq(vk_result, VK_SUCCESS);

	puts("\nVK available instance extensions:");

	for(
		vk_available_instance_extension = vk_available_instance_extensions;
		vk_available_instance_extension < vk_available_instance_extension_end;
		vk_available_instance_extension++
		)
	{
		printf("- %s\n", vk_available_instance_extension->extensionName);
	}

	puts("");

	const char* const* vk_instance_extension = vk_instance_extensions;
	const char* const* vk_instance_extension_end =
		vk_instance_extension + MACRO_ARRAY_LEN(vk_instance_extensions);

	while(vk_instance_extension < vk_instance_extension_end)
	{
		bool found = false;
		const char* extension_name = *(vk_instance_extension++);

		vk_available_instance_extension = vk_available_instance_extensions;
		while(vk_available_instance_extension < vk_available_instance_extension_end)
		{
			if(strcmp(extension_name, vk_available_instance_extension->extensionName) == 0)
			{
				found = true;
				break;
			}

			vk_available_instance_extension++;
		}

		hard_assert_true(found, fprintf(stderr, "VK instance extension %s not found\n", extension_name));
		printf("+ %s\n", extension_name);
		*(extension++) = cstr_init(extension_name);
	}

	uint32_t sdl_instance_extension_count = 0;
	const char* const* sdl_instance_extensions =
		window_get_vulkan_extensions(&sdl_instance_extension_count);
	hard_assert_ptr(sdl_instance_extensions, sdl_instance_extension_count);

	const char* const* sdl_instance_extension = sdl_instance_extensions;
	const char* const* sdl_instance_extension_end =
		sdl_instance_extension + sdl_instance_extension_count;

	while(sdl_instance_extension < sdl_instance_extension_end)
	{
		bool found = false;
		const char* extension_name = *(sdl_instance_extension++);

		vk_available_instance_extension = vk_available_instance_extensions;
		while(vk_available_instance_extension < vk_available_instance_extension_end)
		{
			if(strcmp(extension_name, vk_available_instance_extension->extensionName) == 0)
			{
				found = true;
				break;
			}

			vk_available_instance_extension++;
		}

		hard_assert_true(found, fprintf(stderr, "SDL VK instance extension %s not found\n", extension_name));
		printf("+ %s\n", extension_name);
		*(extension++) = cstr_init(extension_name);
	}

	return extension;
}


private const char**
vk_get_instance_layers(
	vk_t vk,
	const char** layer
	)
{
	assert_not_null(vk);
	assert_not_null(layer);

	uint32_t vk_available_instance_layer_count = 0;
	VkResult vk_result = vkEnumerateInstanceLayerProperties(&vk_available_instance_layer_count, NULL);
	hard_assert_eq(vk_result, VK_SUCCESS);

	VkLayerProperties vk_available_instance_layers[vk_available_instance_layer_count];

	VkLayerProperties* vk_available_instance_layer = vk_available_instance_layers;
	VkLayerProperties* vk_available_instance_layer_end =
		vk_available_instance_layer + vk_available_instance_layer_count;

	while(vk_available_instance_layer < vk_available_instance_layer_end)
	{
		*(vk_available_instance_layer++) = (VkLayerProperties){0};
	}

	vk_result = vkEnumerateInstanceLayerProperties(
		&vk_available_instance_layer_count, vk_available_instance_layers);
	hard_assert_eq(vk_result, VK_SUCCESS);

	puts("\nVK available instance layers:");

	for(
		vk_available_instance_layer = vk_available_instance_layers;
		vk_available_instance_layer < vk_available_instance_layer_end;
		vk_available_instance_layer++
		)
	{
		printf("- %s\n", vk_available_instance_layer->layerName);
	}

	puts("");

	const char* const* vk_instance_layer = vk_instance_layers;
	const char* const* vk_instance_layer_end =
		vk_instance_layer + MACRO_ARRAY_LEN(vk_instance_layers);

	while(vk_instance_layer < vk_instance_layer_end)
	{
		bool found = false;
		const char* layer_name = *(vk_instance_layer++);

		vk_available_instance_layer = vk_available_instance_layers;
		while(vk_available_instance_layer < vk_available_instance_layer_end)
		{
			if(strcmp(layer_name, vk_available_instance_layer->layerName) == 0)
			{
				found = true;
				break;
			}

			vk_available_instance_layer++;
		}

		hard_assert_true(found, fprintf(stderr, "VK instance layer %s not found\n", layer_name));
		printf("+ %s\n", layer_name);
		*(layer++) = cstr_init(layer_name);
	}

	return layer;
}


private void
vk_init_instance(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->vkGetInstanceProcAddr = window_get_vulkan_proc_addr_fn();
	volkInitializeCustom(vk->vkGetInstanceProcAddr);

	const char* vk_instance_extensions[MAX_EXTENSIONS];
	const char** vk_instance_extension =
		vk_get_instance_extensions(vk, vk_instance_extensions);
	assert_lt(vk_instance_extension,
		vk_instance_extensions + MACRO_ARRAY_LEN(vk_instance_extensions));

	const char* vk_instance_layers[MAX_EXTENSIONS];
	const char** vk_instance_layer =
		vk_get_instance_layers(vk, vk_instance_layers);
	assert_lt(vk_instance_layer,
		vk_instance_layers + MACRO_ARRAY_LEN(vk_instance_layers));

	VkApplicationInfo vk_application_info =
	{
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pNext = NULL,
		.pApplicationName = "Thesis",
		.apiVersion = VK_API_VERSION_1_0
	};

	VkInstanceCreateInfo vk_instance_info =
	{
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.pApplicationInfo = &vk_application_info,
		.enabledLayerCount = vk_instance_layer - vk_instance_layers,
		.ppEnabledLayerNames = vk_instance_layers,
		.enabledExtensionCount = vk_instance_extension - vk_instance_extensions,
		.ppEnabledExtensionNames = vk_instance_extensions
	};

#ifndef NDEBUG
	VkDebugUtilsMessengerCreateInfoEXT vk_debug_info =
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
		.pfnUserCallback = vk_debug_callback,
		.pUserData = NULL
	};

	vk_instance_info.pNext = &vk_debug_info;
#endif

	VkResult vk_result = vkCreateInstance(&vk_instance_info, NULL, &vk->vk_instance);
	hard_assert_eq(vk_result, VK_SUCCESS);


	shared_free_str_array(vk_instance_extensions, vk_instance_extension);
	shared_free_str_array(vk_instance_layers, vk_instance_layer);

#ifndef NDEBUG
	PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT =
		vk_load_func(vk, "vkCreateDebugUtilsMessengerEXT");

	vk_result = vkCreateDebugUtilsMessengerEXT(vk->vk_instance,
		&vk_debug_info, NULL, &vk->vk_debug_messenger);
	hard_assert_eq(vk_result, VK_SUCCESS);
#endif

	volkLoadInstanceOnly(vk->vk_instance);
}


private void
vk_free_instance(
	vk_t vk
	)
{
	assert_not_null(vk);

#ifndef NDEBUG
	vkDestroyDebugUtilsMessengerEXT(
		vk->vk_instance, vk->vk_debug_messenger, NULL);
#endif

	vkDestroyInstance(vk->vk_instance, NULL);

	volkFinalize();
}


private void
vk_init_surface(
	vk_t vk
	)
{
	assert_not_null(vk);

	window_init_vulkan_surface(vk->window, vk->vk_instance, &vk->vk_surface);
}


private void
vk_free_surface(
	vk_t vk
	)
{
	assert_not_null(vk);

	window_free_vulkan_surface(vk->vk_instance, vk->vk_surface);
}


typedef struct vk_device_score
{
	uint32_t score;
	uint32_t queue_id;
	VkFormat format;
	VkSampleCountFlagBits samples;
	float anisotropy;
	VkPhysicalDeviceLimits limits;
}
vk_device_score_t;


private bool
vk_get_device_features(
	vk_t vk,
	VkPhysicalDevice device,
	vk_device_score_t* device_score
	)
{
	VkPhysicalDeviceFeatures vk_features;
	vkGetPhysicalDeviceFeatures(device, &vk_features);

	if(vk->vk_anisotropy && !vk_features.samplerAnisotropy)
	{
		hard_assert_log();
		return false;
	}

	if(vk->vk_sample_shading && !vk_features.sampleRateShading)
	{
		hard_assert_log();
		return false;
	}

	VkFormatProperties vk_format_properties;
	vkGetPhysicalDeviceFormatProperties(device,
		VK_FORMAT_R8G8B8A8_SRGB, &vk_format_properties);

	if(
		vk->vk_mipmap_levels &&
		!(
			vk_format_properties.optimalTilingFeatures &
				VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT
			)
		)
	{
		hard_assert_log();
		return false;
	}

	return true;
}


private bool
vk_get_device_queues(
	vk_t vk,
	VkPhysicalDevice device,
	vk_device_score_t* device_score
	)
{
	uint32_t queue_count;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, NULL);
	if(queue_count == 0)
	{
		hard_assert_log();
		return false;
	}

	VkQueueFamilyProperties queues[queue_count];
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, queues);

	VkQueueFamilyProperties* queue = queues;
	for(uint32_t i = 0; i < queue_count; ++i, ++queue)
	{
		VkBool32 present;
		VkResult vk_result =
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, vk->vk_surface, &present);
		if(vk_result != VK_SUCCESS)
		{
			hard_assert_log();
			continue;
		}

		if(present && (queue->queueFlags & VK_QUEUE_GRAPHICS_BIT))
		{
			device_score->queue_id = i;
			return true;
		}
	}

	hard_assert_log();
	return false;
}


private bool
vk_get_device_extensions(
	vk_t vk,
	VkPhysicalDevice device,
	vk_device_score_t* device_score
	)
{
	if(MACRO_ARRAY_LEN(vk_device_extensions) == 0)
	{
		return true;
	}

	uint32_t vk_available_device_extension_count = 0;
	vkEnumerateDeviceExtensionProperties(device,
		NULL, &vk_available_device_extension_count, NULL);
	if(vk_available_device_extension_count == 0)
	{
		hard_assert_log();
		return false;
	}

	VkExtensionProperties vk_available_device_extensions[vk_available_device_extension_count];
	VkExtensionProperties* vk_available_device_extension =
		vk_available_device_extensions;
	VkExtensionProperties* vk_available_device_extension_end =
		vk_available_device_extension + vk_available_device_extension_count;

	while(vk_available_device_extension < vk_available_device_extension_end)
	{
		*(vk_available_device_extension++) = (VkExtensionProperties){0};
	}

	vkEnumerateDeviceExtensionProperties(device,
		NULL, &vk_available_device_extension_count, vk_available_device_extensions);

	puts("\nVK available device extensions:");

	for(
		vk_available_device_extension = vk_available_device_extensions;
		vk_available_device_extension < vk_available_device_extension_end;
		vk_available_device_extension++
		)
	{
		printf("- %s\n", vk_available_device_extension->extensionName);
	}

	puts("");

	const char* const* vk_device_extension = vk_device_extensions;
	const char* const* vk_device_extension_end =
		vk_device_extension + MACRO_ARRAY_LEN(vk_device_extensions);

	while(vk_device_extension < vk_device_extension_end)
	{
		bool found = false;
		const char* extension_name = *(vk_device_extension++);

		vk_available_device_extension = vk_available_device_extensions;
		while(vk_available_device_extension < vk_available_device_extension_end)
		{
			if(strcmp(extension_name, vk_available_device_extension->extensionName) == 0)
			{
				found = true;
				break;
			}

			vk_available_device_extension++;
		}

		if(!found)
		{
			printf("VK device extension %s not found\n", extension_name);
			return false;
		}

		printf("+ %s\n", extension_name);
	}

	return true;
}


private bool
vk_get_device_layers(
	vk_t vk,
	VkPhysicalDevice device,
	vk_device_score_t* device_score
	)
{
	if(MACRO_ARRAY_LEN(vk_device_layers) == 0)
	{
		return true;
	}

	uint32_t vk_available_device_layer_count = 0;
	VkResult vk_result = vkEnumerateDeviceLayerProperties(
		device, &vk_available_device_layer_count, NULL);
	if(vk_result != VK_SUCCESS || vk_available_device_layer_count == 0)
	{
		hard_assert_log();
		return false;
	}

	VkLayerProperties vk_available_device_layers[vk_available_device_layer_count];
	vk_result = vkEnumerateDeviceLayerProperties(
		device, &vk_available_device_layer_count, vk_available_device_layers);
	if(vk_result != VK_SUCCESS)
	{
		hard_assert_log();
		return false;
	}

	puts("\nVK available device layers:");

	for(uint32_t i = 0; i < vk_available_device_layer_count; ++i)
	{
		printf("- %s\n", vk_available_device_layers[i].layerName);
	}

	puts("");

	const char* const* vk_device_layer = vk_device_layers;
	const char* const* vk_device_layer_end =
		vk_device_layer + MACRO_ARRAY_LEN(vk_device_layers);

	while(vk_device_layer < vk_device_layer_end)
	{
		bool found = false;
		const char* layer_name = *(vk_device_layer++);

		VkLayerProperties* layer = vk_available_device_layers;
		VkLayerProperties* layer_end = layer + vk_available_device_layer_count;

		while(layer < layer_end)
		{
			if(strcmp(layer_name, layer->layerName) == 0)
			{
				found = true;
				break;
			}

			layer++;
		}

		if(!found)
		{
			printf("VK device layer %s not found\n", layer_name);
			return false;
		}

		printf("+ %s\n", layer_name);
	}

	return true;
}


private bool
vk_get_device_swapchain(
	vk_t vk,
	VkPhysicalDevice device,
	vk_device_score_t* device_score
	)
{
	uint32_t format_count;
	VkResult vk_result = vkGetPhysicalDeviceSurfaceFormatsKHR(
		device, vk->vk_surface, &format_count, NULL);
	if(vk_result != VK_SUCCESS || format_count == 0)
	{
		hard_assert_log();
		return false;
	}

	VkSurfaceFormatKHR formats[format_count];
	vk_result = vkGetPhysicalDeviceSurfaceFormatsKHR(
		device, vk->vk_surface, &format_count, formats);
	if(vk_result != VK_SUCCESS)
	{
		hard_assert_log();
		return false;
	}

	VkSurfaceFormatKHR* format = formats;
	VkSurfaceFormatKHR* format_end = format + format_count;

	while(1)
	{
		if(
			(
				format->format == VK_FORMAT_R8G8B8A8_SRGB ||
				format->format == VK_FORMAT_B8G8R8A8_SRGB
				) &&
			format->colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
			)
		{
			device_score->format = format->format;
			return true;
		}

		if(++format == format_end)
		{
			hard_assert_log();
			return false;
		}
	}
}


private bool
vk_get_device_properties(
	vk_t vk,
	VkPhysicalDevice device,
	vk_device_score_t* device_score
	)
{
	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(device, &properties);

	if(properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
	{
		device_score->score += 1000;
	}

	VkSampleCountFlags count =
		properties.limits.framebufferColorSampleCounts &
		properties.limits.framebufferDepthSampleCounts;

	if(count >= VK_SAMPLE_COUNT_8_BIT)
	{
		device_score->samples = VK_SAMPLE_COUNT_8_BIT;
	}
	else if(count >= VK_SAMPLE_COUNT_4_BIT)
	{
		device_score->samples = VK_SAMPLE_COUNT_4_BIT;
	}
	else if(count >= VK_SAMPLE_COUNT_2_BIT)
	{
		device_score->samples = VK_SAMPLE_COUNT_2_BIT;
	}
	else
	{
		hard_assert_log();
		return false;
	}

	device_score->score += device_score->samples * 16;

	device_score->anisotropy = properties.limits.maxSamplerAnisotropy;
	device_score->score += device_score->anisotropy * 10;

	if(properties.limits.maxImageDimension2D < 1024)
	{
		hard_assert_log("%u\n", properties.limits.maxImageDimension2D);
		return false;
	}

	if(
		properties.limits.maxPushConstantsSize <
			sizeof(vk_skybox_constant_data_t)
		)
	{
		hard_assert_log("%u\n", properties.limits.maxPushConstantsSize);
		return false;
	}

	if(properties.limits.maxBoundDescriptorSets < 1)
	{
		hard_assert_log("%u\n", properties.limits.maxBoundDescriptorSets);
		return false;
	}

	device_score->score += properties.limits.maxImageDimension2D;
	device_score->limits = properties.limits;

	return true;
}


private vk_device_score_t
vk_get_device_score(
	vk_t vk,
	VkPhysicalDevice device
	)
{
	assert_not_null(vk);
	assert_not_null(device);

	vk_device_score_t device_score = {0};

	if(!vk_get_device_extensions(vk, device, &device_score))
	{
		goto goto_err;
	}

	if(!vk_get_device_layers(vk, device, &device_score))
	{
		goto goto_err;
	}

	if(!vk_get_device_features(vk, device, &device_score))
	{
		goto goto_err;
	}

	if(!vk_get_device_queues(vk, device, &device_score))
	{
		goto goto_err;
	}

	if(!vk_get_device_swapchain(vk, device, &device_score))
	{
		goto goto_err;
	}

	if(!vk_get_device_properties(vk, device, &device_score))
	{
		goto goto_err;
	}

	return device_score;


	goto_err:
	device_score.score = 0;
	return device_score;
}


private bool
vk_load_bool(
	_Atomic bool* value
	)
{
	assert_not_null(value);

	return atomic_load_explicit(value, memory_order_acquire);
}


private void
vk_store_bool(
	_Atomic bool* value,
	bool new_value
	)
{
	assert_not_null(value);

	atomic_store_explicit(value, new_value, memory_order_release);
}


private bool
vk_exchange_bool(
	_Atomic bool* value,
	bool old_value,
	bool new_value
	)
{
	assert_not_null(value);

	atomic_compare_exchange_strong_explicit(value,
		&old_value, new_value, memory_order_acq_rel, memory_order_acquire);
	return old_value;
}


private void
vk_get_extent(
	vk_t vk
	)
{
	assert_not_null(vk);

	uint32_t width = 0;
	uint32_t height = 0;

	while(1)
	{
		VkResult vk_result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
			vk->vk_physical_device, vk->vk_surface, &vk->vk_surface_capabilities);
		hard_assert_eq(vk_result, VK_SUCCESS);

		width = vk->vk_surface_capabilities.currentExtent.width;
		height = vk->vk_surface_capabilities.currentExtent.height;

		if(width != 0 && height != 0)
		{
			break;
		}

		sync_mtx_lock(&vk->window_resize_mtx);
			while(!vk_load_bool(&vk->window_resize_bool))
			{
				sync_cond_wait(&vk->window_resize_cond, &vk->window_resize_mtx);
			}

			vk_store_bool(&vk->window_resize_bool, false);
		sync_mtx_unlock(&vk->window_resize_mtx);
	}

	if(width == UINT32_MAX || height == UINT32_MAX)
	{
		window_info_t window_info;
		window_get_info(vk->window, &window_info);

		width = window_info.extent.w;
		height = window_info.extent.h;
	}

	width = MACRO_CLAMP(
		width,
		vk->vk_surface_capabilities.minImageExtent.width,
		vk->vk_surface_capabilities.maxImageExtent.width
		);

	height = MACRO_CLAMP(
		height,
		vk->vk_surface_capabilities.minImageExtent.height,
		vk->vk_surface_capabilities.maxImageExtent.height
		);

	vk->vk_extent =
	(VkExtent2D)
	{
		.width = width,
		.height = height
	};

	vk->vk_viewport =
	(VkViewport)
	{
		.x = 0.0f,
		.y = 0.0f,
		.width = vk->vk_extent.width,
		.height = vk->vk_extent.height,
		.minDepth = 0.0f,
		.maxDepth = 1.0f
	};

	vk->vk_scissor =
	(VkRect2D)
	{
		.offset = { 0, 0 },
		.extent = vk->vk_extent
	};
}


private void
vk_init_device(
	vk_t vk
	)
{
	assert_not_null(vk);

	uint32_t vk_physical_device_count = 0;
	VkResult vk_result = vkEnumeratePhysicalDevices(
		vk->vk_instance, &vk_physical_device_count, NULL);
	hard_assert_eq(vk_result, VK_SUCCESS);
	hard_assert_neq(vk_physical_device_count, 0);

	VkPhysicalDevice vk_physical_devices[vk_physical_device_count];
	VkPhysicalDevice* vk_physical_device = vk_physical_devices;
	VkPhysicalDevice* vk_physical_device_end =
		vk_physical_device + vk_physical_device_count;

	vk_result = vkEnumeratePhysicalDevices(
		vk->vk_instance, &vk_physical_device_count, vk_physical_devices);
	hard_assert_eq(vk_result, VK_SUCCESS);

	VkPhysicalDevice best_device = NULL;
	vk_device_score_t best_device_score = {0};

	do
	{
		vk_device_score_t this_device_score = vk_get_device_score(vk, *vk_physical_device);

		if(this_device_score.score > best_device_score.score)
		{
			best_device = *vk_physical_device;
			best_device_score = this_device_score;
		}
	}
	while(++vk_physical_device != vk_physical_device_end);

	hard_assert_not_null(best_device);

	vk->vk_queue_id = best_device_score.queue_id;
	vk->vk_format = best_device_score.format;
	vk->vk_samples = best_device_score.samples;
	if(vk->vk_anisotropy)
	{
		vk->vk_anisotropy = MACRO_MIN(vk->vk_anisotropy, best_device_score.anisotropy);
	}
	vk->vk_limits = best_device_score.limits;

	vk->vk_physical_device = best_device;


	float priority = 1.0f;

	VkDeviceQueueCreateInfo vk_device_queue_info =
	{
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.queueFamilyIndex = vk->vk_queue_id,
		.queueCount = 1,
		.pQueuePriorities = &priority
	};

	VkPhysicalDeviceFeatures vk_device_features =
	{
		.samplerAnisotropy = !!vk->vk_anisotropy,
		.sampleRateShading = vk->vk_sample_shading
	};

	VkDeviceCreateInfo vk_device_info =
	{
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &vk_device_queue_info,
		.enabledLayerCount = MACRO_ARRAY_LEN(vk_device_layers),
		.ppEnabledLayerNames = vk_device_layers,
		.enabledExtensionCount = MACRO_ARRAY_LEN(vk_device_extensions),
		.ppEnabledExtensionNames = vk_device_extensions,
		.pEnabledFeatures = &vk_device_features
	};

	vk_result = vkCreateDevice(
		best_device, &vk_device_info, NULL, &vk->vk_device);
	hard_assert_eq(vk_result, VK_SUCCESS);


	volkLoadDeviceTable(&vk->vk_table, vk->vk_device);

	vk->vk_table.vkGetDeviceQueue(vk->vk_device, vk->vk_queue_id, 0, &vk->vk_queue);

	vkGetPhysicalDeviceMemoryProperties(
		vk->vk_physical_device, &vk->vk_memory_properties);


	VkCommandPoolCreateInfo vk_command_pool_info =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.pNext = NULL,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = vk->vk_queue_id
	};

	vk_result = vk->vk_table.vkCreateCommandPool(
		vk->vk_device, &vk_command_pool_info, NULL, &vk->vk_command_pool);
	hard_assert_eq(vk_result, VK_SUCCESS);

	VkCommandBufferAllocateInfo vk_command_buffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = NULL,
		.commandPool = vk->vk_command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1
	};

	vk_result = vk->vk_table.vkAllocateCommandBuffers(
		vk->vk_device, &vk_command_buffer_info, &vk->vk_command_buffer);
	hard_assert_eq(vk_result, VK_SUCCESS);

	VkFenceCreateInfo vk_fence_info =
	{
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.pNext = NULL,
		.flags = VK_FENCE_CREATE_SIGNALED_BIT
	};

	vk_result = vk->vk_table.vkCreateFence(
		vk->vk_device, &vk_fence_info, NULL, &vk->vk_fence);
	hard_assert_eq(vk_result, VK_SUCCESS);


	vk_get_extent(vk);
	if(!vk->vk_surface_capabilities.maxImageCount)
	{
		vk->vk_surface_capabilities.maxImageCount = UINT32_MAX;
	}
	vk->vk_image_count = MACRO_CLAMP(
		3,
		vk->vk_surface_capabilities.minImageCount,
		vk->vk_surface_capabilities.maxImageCount
		);
	vk->vk_transform = vk->vk_surface_capabilities.currentTransform;


	uint32_t present_mode_count;
	vk_result = vkGetPhysicalDeviceSurfacePresentModesKHR(
		vk->vk_physical_device, vk->vk_surface, &present_mode_count, NULL);
	hard_assert_eq(vk_result, VK_SUCCESS);
	hard_assert_neq(present_mode_count, 0);

	VkPresentModeKHR vk_present_modes[present_mode_count];
	VkPresentModeKHR* vk_present_mode = vk_present_modes;
	VkPresentModeKHR* vk_present_mode_end =
		vk_present_mode + present_mode_count;

	vk_result = vkGetPhysicalDeviceSurfacePresentModesKHR(
		vk->vk_physical_device, vk->vk_surface, &present_mode_count, vk_present_modes);
	hard_assert_eq(vk_result, VK_SUCCESS);

	while(1)
	{
		if(*vk_present_mode == VK_PRESENT_MODE_FIFO_RELAXED_KHR)
		{
			vk->vk_present_mode = *vk_present_mode;
			break;
		}

		if(++vk_present_mode == vk_present_mode_end)
		{
			vk->vk_present_mode = VK_PRESENT_MODE_FIFO_KHR;
			break;
		}
	}
}


private void
vk_free_device(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->vk_table.vkDestroyFence(vk->vk_device, vk->vk_fence, NULL);
	vk->vk_table.vkFreeCommandBuffers(vk->vk_device, vk->vk_command_pool, 1, &vk->vk_command_buffer);
	vk->vk_table.vkDestroyCommandPool(vk->vk_device, vk->vk_command_pool, NULL);

	vk->vk_table.vkDestroyDevice(vk->vk_device, NULL);
}


private uint32_t
vk_get_memory(
	vk_t vk,
	uint32_t bits,
	VkMemoryPropertyFlags flags
	)
{
	for(uint32_t i = 0; i < vk->vk_memory_properties.memoryTypeCount; ++i)
	{
		if(
			(bits & (1 << i)) &&
			(vk->vk_memory_properties.memoryTypes[i].propertyFlags & flags) == flags
			)
		{
			return i;
		}
	}

	hard_assert_unreachable();
}


private void
vk_begin_command_buffer(
	vk_t vk
	)
{
	VkResult vk_result = vk->vk_table.vkWaitForFences(
		vk->vk_device, 1, &vk->vk_fence, VK_TRUE, UINT64_MAX);
	hard_assert_eq(vk_result, VK_SUCCESS);

	vk_result = vk->vk_table.vkResetFences(vk->vk_device, 1, &vk->vk_fence);
	hard_assert_eq(vk_result, VK_SUCCESS);

	vk_result = vk->vk_table.vkResetCommandBuffer(vk->vk_command_buffer, 0);
	hard_assert_eq(vk_result, VK_SUCCESS);

	VkCommandBufferBeginInfo vk_command_buffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = NULL,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		.pInheritanceInfo = NULL
	};

	vk_result = vk->vk_table.vkBeginCommandBuffer(
		vk->vk_command_buffer, &vk_command_buffer_info);
	hard_assert_eq(vk_result, VK_SUCCESS);
}


private void
vk_end_command_buffer(
	vk_t vk
	)
{
	VkResult vk_result = vk->vk_table.vkEndCommandBuffer(vk->vk_command_buffer);
	hard_assert_eq(vk_result, VK_SUCCESS);

	VkSubmitInfo vk_submit_info =
	{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = NULL,
		.waitSemaphoreCount = 0,
		.pWaitSemaphores = NULL,
		.pWaitDstStageMask = NULL,
		.commandBufferCount = 1,
		.pCommandBuffers = &vk->vk_command_buffer,
		.signalSemaphoreCount = 0,
		.pSignalSemaphores = NULL
	};

	vk_result = vk->vk_table.vkQueueSubmit(
		vk->vk_queue, 1, &vk_submit_info, vk->vk_fence);
	hard_assert_eq(vk_result, VK_SUCCESS);
}


private void
vk_init_buffer(
	vk_t vk,
	VkDeviceSize size,
	VkBufferUsageFlags usage,
	VkMemoryPropertyFlags flags,
	vk_buffer_t* buffer
	)
{
	assert_not_null(vk);
	assert_not_null(buffer);

	VkBufferCreateInfo vk_buffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 0,
		.pQueueFamilyIndices = NULL
	};

	VkResult vk_result = vk->vk_table.vkCreateBuffer(
		vk->vk_device, &vk_buffer_info, NULL, &buffer->buffer);
	hard_assert_eq(vk_result, VK_SUCCESS);

	VkMemoryRequirements vk_memory_requirements;
	vk->vk_table.vkGetBufferMemoryRequirements(
		vk->vk_device, buffer->buffer, &vk_memory_requirements);

	uint32_t memory_type_index = vk_get_memory(
		vk, vk_memory_requirements.memoryTypeBits, flags);

	VkMemoryAllocateInfo vk_memory_info =
	{
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext = NULL,
		.allocationSize = vk_memory_requirements.size,
		.memoryTypeIndex = memory_type_index
	};

	vk_result = vk->vk_table.vkAllocateMemory(
		vk->vk_device, &vk_memory_info, NULL, &buffer->memory);
	hard_assert_eq(vk_result, VK_SUCCESS);

	vk_result = vk->vk_table.vkBindBufferMemory(
		vk->vk_device, buffer->buffer, buffer->memory, 0);
	hard_assert_eq(vk_result, VK_SUCCESS);
}


private void
vk_free_buffer(
	vk_t vk,
	vk_buffer_t* buffer
	)
{
	assert_not_null(vk);

	vk->vk_table.vkFreeMemory(vk->vk_device, buffer->memory, NULL);
	vk->vk_table.vkDestroyBuffer(vk->vk_device, buffer->buffer, NULL);
}


private void
vk_init_staging_buffer(
	vk_t vk,
	VkDeviceSize size
	)
{
	assert_not_null(vk);

	vk_init_buffer(vk, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		&vk->vk_staging_buffer);
}


private void
vk_free_staging_buffer(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_buffer(vk, &vk->vk_staging_buffer);
}


private void
vk_init_vertex_buffer(
	vk_t vk,
	VkDeviceSize size,
	vk_buffer_t* buffer
	)
{
	assert_not_null(vk);
	assert_not_null(buffer);

	vk_init_buffer(vk, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT |
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		buffer);
}


private void
vk_init_index_buffer(
	vk_t vk,
	VkDeviceSize size,
	vk_buffer_t* buffer
	)
{
	assert_not_null(vk);
	assert_not_null(buffer);

	vk_init_buffer(vk, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT |
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		buffer);
}


private void
vk_init_ubo_buffer(
	vk_t vk,
	VkDeviceSize size,
	vk_buffer_t* buffer,
	VkDescriptorSet* set,
	VkDescriptorSetLayout set_layout
	)
{
	assert_not_null(vk);
	assert_not_null(buffer);

	vk_init_buffer(vk, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT |
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		buffer);

	VkDescriptorSetAllocateInfo vk_set_info =
	{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.pNext = NULL,
		.descriptorPool = vk->vk_descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &set_layout
	};

	VkResult vk_result = vk->vk_table.vkAllocateDescriptorSets(
		vk->vk_device, &vk_set_info, set);
	hard_assert_eq(vk_result, VK_SUCCESS);

	VkDescriptorBufferInfo vk_descriptor_buffer_info =
	{
		.buffer = buffer->buffer,
		.offset = 0,
		.range = VK_WHOLE_SIZE
	};

	VkWriteDescriptorSet vk_write_set =
	{
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.pNext = NULL,
		.dstSet = *set,
		.dstBinding = 0,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pImageInfo = NULL,
		.pBufferInfo = &vk_descriptor_buffer_info,
		.pTexelBufferView = NULL
	};

	vk->vk_table.vkUpdateDescriptorSets(
		vk->vk_device, 1, &vk_write_set, 0, NULL);
}


private void
vk_copy_to_buffer(
	vk_t vk,
	vk_buffer_t* buffer,
	const void* data,
	VkDeviceSize size
	)
{
	assert_not_null(vk);
	assert_not_null(buffer);
	assert_ptr(data, size);

	if(!size)
	{
		return;
	}

	vk_begin_command_buffer(vk);

	vk_free_staging_buffer(vk);
	vk_init_staging_buffer(vk, size);

	void* mapped_data;
	VkResult vk_result = vk->vk_table.vkMapMemory(
		vk->vk_device, vk->vk_staging_buffer.memory, 0, size, 0, &mapped_data);
	hard_assert_eq(vk_result, VK_SUCCESS);

	memcpy(mapped_data, data, size);

	vk->vk_table.vkUnmapMemory(vk->vk_device, vk->vk_staging_buffer.memory);

	VkBufferCopy vk_buffer_copy =
	{
		.srcOffset = 0,
		.dstOffset = 0,
		.size = size
	};

	vk->vk_table.vkCmdCopyBuffer(vk->vk_command_buffer,
		vk->vk_staging_buffer.buffer, buffer->buffer, 1, &vk_buffer_copy);

	vk_end_command_buffer(vk);
}


private void
vk_copy_texture_to_image(
	vk_t vk,
	vk_image_t* image
	)
{
	assert_not_null(vk);
	assert_not_null(image);

	vk_begin_command_buffer(vk);

	vk_free_staging_buffer(vk);
	vk_init_staging_buffer(vk, image->size);

	void* mapped_data;
	VkResult vk_result = vk->vk_table.vkMapMemory(vk->vk_device,
		vk->vk_staging_buffer.memory, 0, image->size, 0, &mapped_data);
	hard_assert_eq(vk_result, VK_SUCCESS);

	memcpy(mapped_data, image->data, image->size);

	vk->vk_table.vkUnmapMemory(vk->vk_device, vk->vk_staging_buffer.memory);

	uint32_t count = image->levels * image->layers;
	VkBufferImageCopy vk_buffer_image_copies[count];
	VkBufferImageCopy* vk_buffer_image_copy = vk_buffer_image_copies;

	uint32_t width = image->width;
	uint32_t height = image->height;
	uint32_t offset = 0;

	for(uint32_t level = 0; level < image->levels; ++level)
	{
		uint32_t stride = width * height * 4;

		for(uint32_t layer = 0; layer < image->layers; ++layer)
		{
			*(vk_buffer_image_copy++) =
			(VkBufferImageCopy)
			{
				.bufferOffset = offset,
				.bufferRowLength = 0,
				.bufferImageHeight = 0,
				.imageSubresource =
				{
					.aspectMask = image->aspect,
					.mipLevel = level,
					.baseArrayLayer = layer,
					.layerCount = 1
				},
				.imageOffset =
				{
					.x = 0,
					.y = 0,
					.z = 0
				},
				.imageExtent = { width, height, 1 }
			};

			offset += stride;
		}

		width = MACRO_MAX(width >> 1, 1);
		height = MACRO_MAX(height >> 1, 1);
	}

	vk->vk_table.vkCmdCopyBufferToImage(vk->vk_command_buffer, vk->vk_staging_buffer.buffer,
		image->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, count, vk_buffer_image_copies);

	vk_end_command_buffer(vk);
}


private void
vk_transition_image_layout(
	vk_t vk,
	vk_image_t* image,
	VkImageLayout from,
	VkImageLayout to
	)
{
	assert_not_null(vk);
	assert_not_null(image);

	vk_begin_command_buffer(vk);

	VkPipelineStageFlags src_stage;
	VkPipelineStageFlags dst_stage;

	VkImageMemoryBarrier vk_barrier =
	{
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.pNext = NULL,
		.oldLayout = from,
		.newLayout = to,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image->image,
		.subresourceRange =
		{
			.aspectMask = image->aspect,
			.baseMipLevel = 0,
			.levelCount = image->levels,
			.baseArrayLayer = 0,
			.layerCount = image->layers
		}
	};

	if(
		from == VK_IMAGE_LAYOUT_UNDEFINED &&
		(
			to == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ||
			to == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
			)
		)
	{
		vk_barrier.srcAccessMask = 0;
		vk_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else if(
		(
			from == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ||
			from == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
			) &&
		to == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		)
	{
		vk_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		vk_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}
	else
	{
		hard_assert_unreachable();
	}

	vk->vk_table.vkCmdPipelineBarrier(vk->vk_command_buffer,
		src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &vk_barrier);

	vk_end_command_buffer(vk);
}


private void
vk_init_image(
	vk_t vk,
	vk_image_t* image
	)
{
	assert_not_null(vk);
	assert_not_null(image);

	VkImageCreateFlags vk_create_flags = 0;
	VkImageViewType vk_view_type = VK_IMAGE_VIEW_TYPE_2D;


	switch(image->type)
	{

	case VK_IMAGE_TYPE_DEPTH_STENCIL:
	{
		image->format = VK_FORMAT_D32_SFLOAT;

		image->aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
		image->usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		image->samples = vk->vk_samples;

		image->data = NULL;
		image->size = 0;
		image->width = vk->vk_extent.width;
		image->height = vk->vk_extent.height;
		image->levels = 1;
		image->layers = 1;

		break;
	}

	case VK_IMAGE_TYPE_MULTISAMPLED:
	{
		image->format = vk->vk_format;

		image->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
		image->usage = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		image->samples = vk->vk_samples;

		image->data = NULL;
		image->size = 0;
		image->width = vk->vk_extent.width;
		image->height = vk->vk_extent.height;
		image->levels = 1;
		image->layers = 1;

		break;
	}

	case VK_IMAGE_TYPE_DEPTH_MAP:
	{
		image->format = VK_FORMAT_D32_SFLOAT;

		image->aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
		image->usage = VK_IMAGE_USAGE_SAMPLED_BIT |
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		image->samples = VK_SAMPLE_COUNT_1_BIT;

		image->data = NULL;
		image->size = 0;
		image->width = vk->vk_depth_map_size;
		image->height = vk->vk_depth_map_size;
		image->levels = 1;
		image->layers = 1;

		break;
	}

	case VK_IMAGE_TYPE_TEXTURE_2D:
	{
		image->format = VK_FORMAT_R8G8B8A8_SRGB;

		image->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
		image->usage = VK_IMAGE_USAGE_SAMPLED_BIT |
		VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		image->samples = VK_SAMPLE_COUNT_1_BIT;

		simulation_texture_t* texture = simulation_get_texture(vk->simulation, image->path, false);
		image->data = texture->data;
		image->size = texture->size;
		image->width = texture->width;
		image->height = texture->height;
		image->levels = 1 + MACRO_MIN(
			MACRO_LOG2(MACRO_MAX(texture->width, texture->height)),
			vk->vk_mipmap_levels
			);
		image->layers = texture->layers;

		break;
	}

	case VK_IMAGE_TYPE_TEXTURE_CUBE:
	{
		image->format = VK_FORMAT_R8G8B8A8_SRGB;

		image->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
		image->usage = VK_IMAGE_USAGE_SAMPLED_BIT |
		VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		image->samples = VK_SAMPLE_COUNT_1_BIT;

		simulation_texture_t* texture = simulation_get_texture(vk->simulation, image->path, true);
		image->data = texture->data;
		image->size = texture->size;
		image->width = texture->width;
		image->height = texture->height;
		image->levels = 1 + MACRO_MIN(
			MACRO_LOG2(MACRO_MAX(image->width, image->height)),
			vk->vk_mipmap_levels
			);
		image->layers = texture->layers;

		vk_create_flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
		vk_view_type = VK_IMAGE_VIEW_TYPE_CUBE;

		break;
	}

	default: assert_unreachable();

	}


	VkImageCreateInfo vk_image_info =
	{
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.pNext = NULL,
		.flags = vk_create_flags,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = image->format,
		.extent = { image->width, image->height, 1 },
		.mipLevels = image->levels,
		.arrayLayers = image->layers,
		.samples = image->samples,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = image->usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 0,
		.pQueueFamilyIndices = NULL,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
	};

	VkResult vk_result = vk->vk_table.vkCreateImage(
		vk->vk_device, &vk_image_info, NULL, &image->image);
	hard_assert_eq(vk_result, VK_SUCCESS);

	VkMemoryRequirements vk_memory_requirements;
	vk->vk_table.vkGetImageMemoryRequirements(
		vk->vk_device, image->image, &vk_memory_requirements);

	uint32_t memory_type_index = vk_get_memory(vk,
		vk_memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	VkMemoryAllocateInfo vk_memory_info =
	{
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext = NULL,
		.allocationSize = vk_memory_requirements.size,
		.memoryTypeIndex = memory_type_index
	};

	vk_result = vk->vk_table.vkAllocateMemory(
		vk->vk_device, &vk_memory_info, NULL, &image->memory);
	hard_assert_eq(vk_result, VK_SUCCESS);

	vk_result = vk->vk_table.vkBindImageMemory(
		vk->vk_device, image->image, image->memory, 0);
	hard_assert_eq(vk_result, VK_SUCCESS);

	VkImageViewCreateInfo vk_image_view_info =
	{
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.image = image->image,
		.viewType = vk_view_type,
		.format = image->format,
		.components =
		{
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY
		},
		.subresourceRange =
		{
			.aspectMask = image->aspect,
			.baseMipLevel = 0,
			.levelCount = image->levels,
			.baseArrayLayer = 0,
			.layerCount = image->layers
		}
	};

	vk_result = vk->vk_table.vkCreateImageView(
		vk->vk_device, &vk_image_view_info, NULL, &image->view);
	hard_assert_eq(vk_result, VK_SUCCESS);

	if(image->type >= VK_IMAGE_TYPE_TEXTURE_2D)
	{
		vk_transition_image_layout(vk, image,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

		vk_copy_texture_to_image(vk, image);

		vk_transition_image_layout(vk, image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	}
}


private void
vk_free_image(
	vk_t vk,
	vk_image_t* image
	)
{
	assert_not_null(vk);
	assert_not_null(image);

	vk->vk_table.vkDestroyImageView(vk->vk_device, image->view, NULL);
	vk->vk_table.vkDestroyImage(vk->vk_device, image->image, NULL);
	vk->vk_table.vkFreeMemory(vk->vk_device, image->memory, NULL);
}


private VkShaderModule
vk_create_shader(
	vk_t vk,
	const char* path
	)
{
	file_t file;
	bool status = file_read(path, &file);
	hard_assert_true(status);

	VkShaderModuleCreateInfo vk_shader_info =
	{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.codeSize = file.len,
		.pCode = (void*) file.data
	};

	VkShaderModule vk_shader_module;
	VkResult vk_result = vk->vk_table.vkCreateShaderModule(
		vk->vk_device, &vk_shader_info, NULL, &vk_shader_module);
	hard_assert_eq(vk_result, VK_SUCCESS);

	file_free(file);

	return vk_shader_module;
}


private void
vk_destroy_shader(
	vk_t vk,
	VkShaderModule shader
	)
{
	assert_not_null(vk);
	assert_not_null(shader);

	vk->vk_table.vkDestroyShaderModule(vk->vk_device, shader, NULL);
}


private VkPipelineCache
vk_init_pipeline_cache(
	vk_t vk,
	const char* path
	)
{
	assert_not_null(vk);
	assert_not_null(path);

	VkPipelineCacheCreateInfo vk_pipeline_cache_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.initialDataSize = 0,
		.pInitialData = NULL
	};

	file_t file = {0};
	VkPipelineCache vk_pipeline_cache;

	if(file_exists(path))
	{
		bool status = file_read(path, &file);
		if(status)
		{
			vk_pipeline_cache_info.initialDataSize = file.len;
			vk_pipeline_cache_info.pInitialData = file.data;
		}
		else
		{
			hard_assert_log("file_read(\"%s\")", path);
		}
	}

	VkResult vk_result = vk->vk_table.vkCreatePipelineCache(
		vk->vk_device, &vk_pipeline_cache_info, NULL, &vk_pipeline_cache);
	hard_assert_eq(vk_result, VK_SUCCESS);

	file_free(file);

	return vk_pipeline_cache;
}


private void
vk_free_pipeline_cache(
	vk_t vk,
	const char* path,
	VkPipelineCache vk_pipeline_cache
	)
{
	assert_not_null(vk);

	file_t file;
	VkResult vk_result = vk->vk_table.vkGetPipelineCacheData(vk->vk_device,
		vk_pipeline_cache, &file.len, NULL);
	hard_assert_eq(vk_result, VK_SUCCESS);

	file.data = alloc_malloc(file.len);
	assert_ptr(file.data, file.len);

	vk_result = vk->vk_table.vkGetPipelineCacheData(vk->vk_device,
		vk_pipeline_cache, &file.len, file.data);
	hard_assert_eq(vk_result, VK_SUCCESS);

	bool status = file_write(path, file);
	if(!status)
	{
		hard_assert_log("file_write(\"%s\")", path);
	}

	file_free(file);

	vk->vk_table.vkDestroyPipelineCache(vk->vk_device, vk_pipeline_cache, NULL);

}


private void
vk_free_sampler(
	vk_t vk,
	VkSampler sampler
	)
{
	assert_not_null(vk);

	vk->vk_table.vkDestroySampler(vk->vk_device, sampler, NULL);
}


private void
vk_init_depth_sampler(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkSamplerCreateInfo vk_sampler_info =
	{
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.mipLodBias = 0.0f,
		.anisotropyEnable = VK_FALSE,
		.maxAnisotropy = 1.0f,
		.compareEnable = VK_TRUE,
		.compareOp = VK_COMPARE_OP_LESS,
		.minLod = 0.0f,
		.maxLod = 1.0f,
		.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
		.unnormalizedCoordinates = VK_FALSE
	};

	VkResult vk_result = vk->vk_table.vkCreateSampler(
		vk->vk_device, &vk_sampler_info, NULL, &vk->vk_depth_sampler);
	hard_assert_eq(vk_result, VK_SUCCESS);
}


private void
vk_init_image_sampler(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkSamplerCreateInfo vk_sampler_info =
	{
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.mipLodBias = 0.0f,
		.anisotropyEnable = !!vk->vk_anisotropy,
		.maxAnisotropy = vk->vk_anisotropy,
		.compareEnable = VK_FALSE,
		.compareOp = VK_COMPARE_OP_ALWAYS,
		.minLod = 0.0f,
		.maxLod = vk->vk_mipmap_levels,
		.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
		.unnormalizedCoordinates = VK_FALSE
	};

	VkResult vk_result = vk->vk_table.vkCreateSampler(
		vk->vk_device, &vk_sampler_info, NULL, &vk->vk_image_sampler);
	hard_assert_eq(vk_result, VK_SUCCESS);
}


private void
vk_init_samplers(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_depth_sampler(vk);
	vk_init_image_sampler(vk);
}


private void
vk_free_samplers(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_sampler(vk, vk->vk_image_sampler);
	vk_free_sampler(vk, vk->vk_depth_sampler);
}


private void
vk_init_shadow_render_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkAttachmentDescription vk_attachments[] =
	{
		{
			.flags = 0,
			.format = VK_FORMAT_D32_SFLOAT,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
		}
	};

	VkAttachmentReference vk_depth_attachment =
	{
		.attachment = 0,
		.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
	};

	VkSubpassDescription vk_subpasses[] =
	{
		{
			.flags = 0,
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount = 0,
			.pInputAttachments = NULL,
			.colorAttachmentCount = 0,
			.pColorAttachments = NULL,
			.pResolveAttachments = NULL,
			.pDepthStencilAttachment = &vk_depth_attachment,
			.preserveAttachmentCount = 0,
			.pPreserveAttachments = NULL
		}
	};

	VkSubpassDependency vk_subpass_dependencies[] =
	{
		{
			.srcSubpass = VK_SUBPASS_EXTERNAL,
			.dstSubpass = 0,
			.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			.srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
			.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT
		},
		{
			.srcSubpass = 0,
			.dstSubpass = VK_SUBPASS_EXTERNAL,
			.srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
			.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT
		}
	};

	VkRenderPassCreateInfo vk_render_pass_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.attachmentCount = MACRO_ARRAY_LEN(vk_attachments),
		.pAttachments = vk_attachments,
		.subpassCount = MACRO_ARRAY_LEN(vk_subpasses),
		.pSubpasses = vk_subpasses,
		.dependencyCount = MACRO_ARRAY_LEN(vk_subpass_dependencies),
		.pDependencies = vk_subpass_dependencies
	};

	VkResult vk_result = vk->vk_table.vkCreateRenderPass(vk->vk_device,
		&vk_render_pass_info, NULL, &vk->vk_shadow.render_pass);
	hard_assert_eq(vk_result, VK_SUCCESS);
}


private void
vk_free_shadow_render_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->vk_table.vkDestroyRenderPass(vk->vk_device, vk->vk_shadow.render_pass, NULL);
}


private void
vk_init_shadow_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_shadow_render_pass(vk);
}


private void
vk_free_shadow_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_shadow_render_pass(vk);
}


private void
vk_init_shadow_pipeline(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkPipelineShaderStageCreateInfo vk_shader_stages[] =
	{
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vk_create_shader(vk, "shaders/depth.vert.spv"),
			.pName = "main",
			.pSpecializationInfo = NULL
		}
	};

	VkPipelineDynamicStateCreateInfo vk_dynamic_state_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.dynamicStateCount = 0,
		.pDynamicStates = NULL
	};

	VkVertexInputBindingDescription vk_vertex_bindings[] =
	{
		{
			.binding = 0,
			.stride = sizeof(vk_depth_vertex_data_t),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
		},
		{
			.binding = 1,
			.stride = sizeof(vk_model_instance_data_t),
			.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE
		}
	};

	VkVertexInputAttributeDescription vk_vertex_attributes[] =
	{
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(vk_depth_vertex_data_t, position)
		},
		{
			.location = 1,
			.binding = 1,
			.format = VK_FORMAT_R32G32B32A32_SFLOAT,
			.offset = offsetof(vk_model_instance_data_t, transform) + sizeof(vec4) * 0
		},
		{
			.location = 2,
			.binding = 1,
			.format = VK_FORMAT_R32G32B32A32_SFLOAT,
			.offset = offsetof(vk_model_instance_data_t, transform) + sizeof(vec4) * 1
		},
		{
			.location = 3,
			.binding = 1,
			.format = VK_FORMAT_R32G32B32A32_SFLOAT,
			.offset = offsetof(vk_model_instance_data_t, transform) + sizeof(vec4) * 2
		},
		{
			.location = 4,
			.binding = 1,
			.format = VK_FORMAT_R32G32B32A32_SFLOAT,
			.offset = offsetof(vk_model_instance_data_t, transform) + sizeof(vec4) * 3
		}
	};

	VkPipelineVertexInputStateCreateInfo vk_vertex_input_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.vertexBindingDescriptionCount = MACRO_ARRAY_LEN(vk_vertex_bindings),
		.pVertexBindingDescriptions = vk_vertex_bindings,
		.vertexAttributeDescriptionCount = MACRO_ARRAY_LEN(vk_vertex_attributes),
		.pVertexAttributeDescriptions = vk_vertex_attributes
	};

	VkPipelineInputAssemblyStateCreateInfo vk_input_assembly_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE
	};

	VkViewport vk_viewport =
	{
		.x = 0.0f,
		.y = 0.0f,
		.width = vk->vk_depth_map_size,
		.height = vk->vk_depth_map_size,
		.minDepth = 0.0f,
		.maxDepth = 1.0f
	};

	VkRect2D vk_scissor =
	{
		.offset = { 0, 0 },
		.extent = { vk_viewport.width, vk_viewport.height }
	};

	VkPipelineViewportStateCreateInfo vk_viewport_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.viewportCount = 1,
		.pViewports = &vk_viewport,
		.scissorCount = 1,
		.pScissors = &vk_scissor
	};

	VkPipelineRasterizationStateCreateInfo vk_rasterization_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_TRUE,
		.depthBiasConstantFactor = 1.25f,
		.depthBiasClamp = 0.0f,
		.depthBiasSlopeFactor = 1.75f,
		.lineWidth = 1.0f
	};

	VkPipelineMultisampleStateCreateInfo vk_multisample_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		.sampleShadingEnable = VK_FALSE,
		.minSampleShading = 0.0f,
		.pSampleMask = NULL,
		.alphaToCoverageEnable = VK_FALSE,
		.alphaToOneEnable = VK_FALSE
	};

	VkPipelineDepthStencilStateCreateInfo vk_depth_stencil_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
		.front = {0},
		.back = {0},
		.minDepthBounds = 0.0f,
		.maxDepthBounds = 1.0f
	};

	VkPipelineColorBlendAttachmentState vk_color_blend_attachment =
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

	VkPipelineColorBlendStateCreateInfo vk_color_blend_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.logicOpEnable = VK_FALSE,
		.logicOp = VK_LOGIC_OP_COPY,
		.attachmentCount = 1,
		.pAttachments = &vk_color_blend_attachment,
		.blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f }
	};


	VkPushConstantRange vk_push_constants[] =
	{
		{
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
			.offset = 0,
			.size = sizeof(vk_depth_vert_constant_data_t)
		}
	};

	VkPipelineLayoutCreateInfo vk_pipeline_layout_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.setLayoutCount = 0,
		.pSetLayouts = NULL,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = vk_push_constants
	};

	VkResult vk_result = vk->vk_table.vkCreatePipelineLayout(vk->vk_device,
		&vk_pipeline_layout_info, NULL, &vk->vk_shadow.depth.pipeline_layout);
	hard_assert_eq(vk_result, VK_SUCCESS);


	VkGraphicsPipelineCreateInfo vk_pipeline_info =
	{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.stageCount = MACRO_ARRAY_LEN(vk_shader_stages),
		.pStages = vk_shader_stages,
		.pVertexInputState = &vk_vertex_input_info,
		.pInputAssemblyState = &vk_input_assembly_info,
		.pTessellationState = NULL,
		.pViewportState = &vk_viewport_info,
		.pRasterizationState = &vk_rasterization_info,
		.pMultisampleState = &vk_multisample_info,
		.pDepthStencilState = &vk_depth_stencil_info,
		.pColorBlendState = &vk_color_blend_info,
		.pDynamicState = &vk_dynamic_state_info,
		.layout = vk->vk_shadow.depth.pipeline_layout,
		.renderPass = vk->vk_shadow.render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	static const char* vk_pipeline_cache_path = "cache/vk_shadow_pipeline.bin";
	VkPipelineCache vk_pipeline_cache = vk_init_pipeline_cache(vk, vk_pipeline_cache_path);

	vk_result = vk->vk_table.vkCreateGraphicsPipelines(vk->vk_device,
		vk_pipeline_cache, 1, &vk_pipeline_info, NULL, &vk->vk_shadow.depth.pipeline);
	hard_assert_eq(vk_result, VK_SUCCESS);

	vk_free_pipeline_cache(vk, vk_pipeline_cache_path, vk_pipeline_cache);

	for(uint32_t i = 0; i < MACRO_ARRAY_LEN(vk_shader_stages); ++i)
	{
		vk_destroy_shader(vk, vk_shader_stages[i].module);
	}
}


private void
vk_free_shadow_pipeline(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->vk_table.vkDestroyPipeline(vk->vk_device, vk->vk_shadow.depth.pipeline, NULL);
	vk->vk_table.vkDestroyPipelineLayout(vk->vk_device, vk->vk_shadow.depth.pipeline_layout, NULL);
}


private void
vk_init_shadow_consts(
	vk_t vk
	)
{
	assert_not_null(vk);
}


private void
vk_free_shadow_consts(
	vk_t vk
	)
{
	assert_not_null(vk);
}


private void
vk_init_shadow(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_shadow_pass(vk);
	vk_init_shadow_pipeline(vk);
	vk_init_shadow_consts(vk);
}


private void
vk_free_shadow(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_shadow_consts(vk);
	vk_free_shadow_pipeline(vk);
	vk_free_shadow_pass(vk);
}


private void
vk_init_scene_render_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkAttachmentDescription vk_attachments[] =
	{
		{
			.flags = 0,
			.format = vk->vk_scene.multisampled_image.format,
			.samples = vk->vk_samples,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		},
		{
			.flags = 0,
			.format = vk->vk_scene.depth_image.format,
			.samples = vk->vk_samples,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
		},
		{
			.flags = 0,
			.format = vk->vk_scene.multisampled_image.format,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
		}
	};

	VkAttachmentReference vk_color_attachment =
	{
		.attachment = 0,
		.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
	};

	VkAttachmentReference vk_depth_attachment =
	{
		.attachment = 1,
		.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
	};

	VkAttachmentReference vk_multisampled_attachment =
	{
		.attachment = 2,
		.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
	};

	VkSubpassDescription vk_subpasses[] =
	{
		{
			.flags = 0,
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount = 0,
			.pInputAttachments = NULL,
			.colorAttachmentCount = 1,
			.pColorAttachments = &vk_color_attachment,
			.pResolveAttachments = &vk_multisampled_attachment,
			.pDepthStencilAttachment = &vk_depth_attachment,
			.preserveAttachmentCount = 0,
			.pPreserveAttachments = NULL
		}
	};

	VkSubpassDependency vk_subpass_dependencies[] =
	{
		{
			.srcSubpass = VK_SUBPASS_EXTERNAL,
			.dstSubpass = 0,
			.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
				VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
				VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
				VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.dependencyFlags = 0
		}
	};

	VkRenderPassCreateInfo vk_render_pass_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.attachmentCount = MACRO_ARRAY_LEN(vk_attachments),
		.pAttachments = vk_attachments,
		.subpassCount = MACRO_ARRAY_LEN(vk_subpasses),
		.pSubpasses = vk_subpasses,
		.dependencyCount = MACRO_ARRAY_LEN(vk_subpass_dependencies),
		.pDependencies = vk_subpass_dependencies
	};

	VkResult vk_result = vk->vk_table.vkCreateRenderPass(vk->vk_device,
		&vk_render_pass_info, NULL, &vk->vk_scene.render_pass);
	hard_assert_eq(vk_result, VK_SUCCESS);
}


private void
vk_free_scene_render_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->vk_table.vkDestroyRenderPass(vk->vk_device, vk->vk_scene.render_pass, NULL);
}


private void
vk_init_scene_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->vk_scene.depth_image.type = VK_IMAGE_TYPE_DEPTH_STENCIL;
	vk_init_image(vk, &vk->vk_scene.depth_image);

	vk->vk_scene.multisampled_image.type = VK_IMAGE_TYPE_MULTISAMPLED;
	vk_init_image(vk, &vk->vk_scene.multisampled_image);

	vk_init_scene_render_pass(vk);
}


private void
vk_free_scene_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_scene_render_pass(vk);
	vk_free_image(vk, &vk->vk_scene.multisampled_image);
	vk_free_image(vk, &vk->vk_scene.depth_image);
}


private void
vk_init_skybox_pipeline(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkPipelineShaderStageCreateInfo vk_shader_stages[] =
	{
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vk_create_shader(vk, "shaders/skybox.vert.spv"),
			.pName = "main",
			.pSpecializationInfo = NULL
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = vk_create_shader(vk, "shaders/skybox.frag.spv"),
			.pName = "main",
			.pSpecializationInfo = NULL
		}
	};

	VkDynamicState vk_dynamic_states[] =
	{
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};

	VkPipelineDynamicStateCreateInfo vk_dynamic_state_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.dynamicStateCount = MACRO_ARRAY_LEN(vk_dynamic_states),
		.pDynamicStates = vk_dynamic_states
	};

	VkVertexInputBindingDescription vk_vertex_bindings[] =
	{
		{
			.binding = 0,
			.stride = sizeof(vk_skybox_vertex_data_t),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
		}
	};

	VkVertexInputAttributeDescription vk_vertex_attributes[] =
	{
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(vk_skybox_vertex_data_t, position)
		}
	};

	VkPipelineVertexInputStateCreateInfo vk_vertex_input_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.vertexBindingDescriptionCount = MACRO_ARRAY_LEN(vk_vertex_bindings),
		.pVertexBindingDescriptions = vk_vertex_bindings,
		.vertexAttributeDescriptionCount = MACRO_ARRAY_LEN(vk_vertex_attributes),
		.pVertexAttributeDescriptions = vk_vertex_attributes
	};

	VkPipelineInputAssemblyStateCreateInfo vk_input_assembly_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE
	};

	VkPipelineViewportStateCreateInfo vk_viewport_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.viewportCount = 1,
		.pViewports = NULL,
		.scissorCount = 1,
		.pScissors = NULL
	};

	VkPipelineRasterizationStateCreateInfo vk_rasterization_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.depthBiasConstantFactor = 0.0f,
		.depthBiasClamp = 0.0f,
		.depthBiasSlopeFactor = 0.0f,
		.lineWidth = 1.0f
	};

	VkPipelineMultisampleStateCreateInfo vk_multisample_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.rasterizationSamples = vk->vk_samples,
		.sampleShadingEnable = vk->vk_sample_shading,
		.minSampleShading = 0.2f,
		.pSampleMask = NULL,
		.alphaToCoverageEnable = VK_FALSE,
		.alphaToOneEnable = VK_FALSE
	};

	VkPipelineDepthStencilStateCreateInfo vk_depth_stencil_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.depthTestEnable = VK_FALSE,
		.depthWriteEnable = VK_FALSE,
		.depthCompareOp = VK_COMPARE_OP_NEVER,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
		.front = {0},
		.back = {0},
		.minDepthBounds = 0.0f,
		.maxDepthBounds = 1.0f
	};

	VkPipelineColorBlendAttachmentState vk_color_blend_attachment =
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

	VkPipelineColorBlendStateCreateInfo vk_color_blend_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.logicOpEnable = VK_FALSE,
		.logicOp = VK_LOGIC_OP_COPY,
		.attachmentCount = 1,
		.pAttachments = &vk_color_blend_attachment,
		.blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f }
	};

	VkDescriptorSetLayoutBinding vk_set_layout_bindings[] =
	{
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
			.pImmutableSamplers = NULL
		}
	};

	VkDescriptorSetLayoutCreateInfo vk_set_layout_info =
	{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.bindingCount = MACRO_ARRAY_LEN(vk_set_layout_bindings),
		.pBindings = vk_set_layout_bindings
	};

	VkResult vk_result = vk->vk_table.vkCreateDescriptorSetLayout(vk->vk_device,
		&vk_set_layout_info, NULL, &vk->vk_scene.skybox.set_layout);
	hard_assert_eq(vk_result, VK_SUCCESS);


	VkPushConstantRange vk_push_constants[] =
	{
		{
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
			.offset = 0,
			.size = sizeof(vk_skybox_constant_data_t)
		}
	};

	VkPipelineLayoutCreateInfo vk_pipeline_layout_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.setLayoutCount = 1,
		.pSetLayouts = &vk->vk_scene.skybox.set_layout,
		.pushConstantRangeCount = MACRO_ARRAY_LEN(vk_push_constants),
		.pPushConstantRanges = vk_push_constants
	};

	vk_result = vk->vk_table.vkCreatePipelineLayout(vk->vk_device,
		&vk_pipeline_layout_info, NULL, &vk->vk_scene.skybox.pipeline_layout);
	hard_assert_eq(vk_result, VK_SUCCESS);


	VkGraphicsPipelineCreateInfo vk_pipeline_info =
	{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.stageCount = MACRO_ARRAY_LEN(vk_shader_stages),
		.pStages = vk_shader_stages,
		.pVertexInputState = &vk_vertex_input_info,
		.pInputAssemblyState = &vk_input_assembly_info,
		.pTessellationState = NULL,
		.pViewportState = &vk_viewport_info,
		.pRasterizationState = &vk_rasterization_info,
		.pMultisampleState = &vk_multisample_info,
		.pDepthStencilState = &vk_depth_stencil_info,
		.pColorBlendState = &vk_color_blend_info,
		.pDynamicState = &vk_dynamic_state_info,
		.layout = vk->vk_scene.skybox.pipeline_layout,
		.renderPass = vk->vk_scene.render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	static const char* vk_pipeline_cache_path = "cache/vk_skybox_pipeline.bin";
	VkPipelineCache vk_pipeline_cache = vk_init_pipeline_cache(vk, vk_pipeline_cache_path);

	vk_result = vk->vk_table.vkCreateGraphicsPipelines(vk->vk_device,
		vk_pipeline_cache, 1, &vk_pipeline_info, NULL, &vk->vk_scene.skybox.pipeline);
	hard_assert_eq(vk_result, VK_SUCCESS);

	vk_free_pipeline_cache(vk, vk_pipeline_cache_path, vk_pipeline_cache);

	for(uint32_t i = 0; i < MACRO_ARRAY_LEN(vk_shader_stages); ++i)
	{
		vk_destroy_shader(vk, vk_shader_stages[i].module);
	}
}


private void
vk_free_skybox_pipeline(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->vk_table.vkDestroyPipeline(vk->vk_device, vk->vk_scene.skybox.pipeline, NULL);
	vk->vk_table.vkDestroyPipelineLayout(vk->vk_device, vk->vk_scene.skybox.pipeline_layout, NULL);
	vk->vk_table.vkDestroyDescriptorSetLayout(vk->vk_device, vk->vk_scene.skybox.set_layout, NULL);
}


private void
vk_init_skybox_consts(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->vk_scene.skybox.image =
	(vk_image_t)
	{
		.path = simulation_get_skybox_path(vk->simulation),
		.type = VK_IMAGE_TYPE_TEXTURE_CUBE
	};
	vk_init_image(vk, &vk->vk_scene.skybox.image);

	VkDescriptorSetAllocateInfo vk_set_info =
	{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.pNext = NULL,
		.descriptorPool = vk->vk_descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &vk->vk_scene.skybox.set_layout
	};

	VkResult vk_result = vk->vk_table.vkAllocateDescriptorSets(
		vk->vk_device, &vk_set_info, &vk->vk_scene.skybox.set);
	hard_assert_eq(vk_result, VK_SUCCESS);


	VkDescriptorImageInfo vk_image_info =
	{
		.sampler = vk->vk_image_sampler,
		.imageView = vk->vk_scene.skybox.image.view,
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
	};

	VkWriteDescriptorSet vk_write_set =
	{
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.pNext = NULL,
		.dstSet = vk->vk_scene.skybox.set,
		.dstBinding = 0,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.pImageInfo = &vk_image_info,
		.pBufferInfo = NULL,
		.pTexelBufferView = NULL
	};

	vk->vk_table.vkUpdateDescriptorSets(vk->vk_device, 1, &vk_write_set, 0, NULL);


	vk_init_vertex_buffer(vk, sizeof(vk_skybox_vertex_data), &vk->vk_scene.skybox.vertex_buffer);

	vk_copy_to_buffer(vk, &vk->vk_scene.skybox.vertex_buffer,
		vk_skybox_vertex_data, sizeof(vk_skybox_vertex_data));

	vk_init_index_buffer(vk, sizeof(vk_skybox_index_data), &vk->vk_scene.skybox.index_buffer);

	vk_copy_to_buffer(vk, &vk->vk_scene.skybox.index_buffer,
		vk_skybox_index_data, sizeof(vk_skybox_index_data));
}


private void
vk_free_skybox_consts(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_buffer(vk, &vk->vk_scene.skybox.index_buffer);
	vk_free_buffer(vk, &vk->vk_scene.skybox.vertex_buffer);

	vk_free_image(vk, &vk->vk_scene.skybox.image);
}


private void
vk_init_mesh_pipeline(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkPipelineShaderStageCreateInfo vk_shader_stages[] =
	{
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vk_create_shader(vk, "shaders/mesh.vert.spv"),
			.pName = "main",
			.pSpecializationInfo = NULL
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = vk_create_shader(vk, "shaders/mesh.frag.spv"),
			.pName = "main",
			.pSpecializationInfo = NULL
		}
	};

	VkDynamicState vk_dynamic_states[] =
	{
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};

	VkPipelineDynamicStateCreateInfo vk_dynamic_state_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.dynamicStateCount = MACRO_ARRAY_LEN(vk_dynamic_states),
		.pDynamicStates = vk_dynamic_states
	};

	VkVertexInputBindingDescription vk_vertex_bindings[] =
	{
		{
			.binding = 0,
			.stride = sizeof(vk_mesh_vertex_data_t),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
		},
		{
			.binding = 1,
			.stride = sizeof(vk_model_instance_data_t),
			.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE
		}
	};

	VkVertexInputAttributeDescription vk_vertex_attributes[] =
	{
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(vk_mesh_vertex_data_t, position)
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(vk_mesh_vertex_data_t, normal)
		},
		{
			.location = 2,
			.binding = 0,
			.format = VK_FORMAT_R32G32_SFLOAT,
			.offset = offsetof(vk_mesh_vertex_data_t, coords)
		},
		{
			.location = 3,
			.binding = 1,
			.format = VK_FORMAT_R32G32B32A32_SFLOAT,
			.offset = offsetof(vk_model_instance_data_t, transform) + sizeof(vec4) * 0
		},
		{
			.location = 4,
			.binding = 1,
			.format = VK_FORMAT_R32G32B32A32_SFLOAT,
			.offset = offsetof(vk_model_instance_data_t, transform) + sizeof(vec4) * 1
		},
		{
			.location = 5,
			.binding = 1,
			.format = VK_FORMAT_R32G32B32A32_SFLOAT,
			.offset = offsetof(vk_model_instance_data_t, transform) + sizeof(vec4) * 2
		},
		{
			.location = 6,
			.binding = 1,
			.format = VK_FORMAT_R32G32B32A32_SFLOAT,
			.offset = offsetof(vk_model_instance_data_t, transform) + sizeof(vec4) * 3
		}
	};

	VkPipelineVertexInputStateCreateInfo vk_vertex_input_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.vertexBindingDescriptionCount = MACRO_ARRAY_LEN(vk_vertex_bindings),
		.pVertexBindingDescriptions = vk_vertex_bindings,
		.vertexAttributeDescriptionCount = MACRO_ARRAY_LEN(vk_vertex_attributes),
		.pVertexAttributeDescriptions = vk_vertex_attributes
	};

	VkPipelineInputAssemblyStateCreateInfo vk_input_assembly_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE
	};

	VkPipelineViewportStateCreateInfo vk_viewport_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.viewportCount = 1,
		.pViewports = NULL,
		.scissorCount = 1,
		.pScissors = NULL
	};

	VkPipelineRasterizationStateCreateInfo vk_rasterization_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.depthBiasConstantFactor = 0.0f,
		.depthBiasClamp = 0.0f,
		.depthBiasSlopeFactor = 0.0f,
		.lineWidth = 1.0f
	};

	VkPipelineMultisampleStateCreateInfo vk_multisample_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.rasterizationSamples = vk->vk_samples,
		.sampleShadingEnable = vk->vk_sample_shading,
		.minSampleShading = 0.2f,
		.pSampleMask = NULL,
		.alphaToCoverageEnable = VK_FALSE,
		.alphaToOneEnable = VK_FALSE
	};

	VkPipelineDepthStencilStateCreateInfo vk_depth_stencil_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_GREATER,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
		.front = {0},
		.back = {0},
		.minDepthBounds = 0.0f,
		.maxDepthBounds = 1.0f
	};

	VkPipelineColorBlendAttachmentState vk_color_blend_attachment =
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

	VkPipelineColorBlendStateCreateInfo vk_color_blend_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.logicOpEnable = VK_FALSE,
		.logicOp = VK_LOGIC_OP_COPY,
		.attachmentCount = 1,
		.pAttachments = &vk_color_blend_attachment,
		.blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f }
	};

	VkDescriptorSetLayoutBinding vk_set_layout_bindings[] =
	{
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
			.pImmutableSamplers = NULL
		}
	};

	VkDescriptorSetLayoutCreateInfo vk_set_layout_info =
	{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.bindingCount = MACRO_ARRAY_LEN(vk_set_layout_bindings),
		.pBindings = vk_set_layout_bindings
	};

	VkResult vk_result = vk->vk_table.vkCreateDescriptorSetLayout(vk->vk_device,
		&vk_set_layout_info, NULL, &vk->vk_scene.mesh.ubo_set_layout);
	hard_assert_eq(vk_result, VK_SUCCESS);


	vk_set_layout_bindings[0] =
	(VkDescriptorSetLayoutBinding)
	{
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.pImmutableSamplers = NULL
	};

	vk_result = vk->vk_table.vkCreateDescriptorSetLayout(vk->vk_device,
		&vk_set_layout_info, NULL, &vk->vk_scene.mesh.texture_set_layout);
	hard_assert_eq(vk_result, VK_SUCCESS);


	vk_set_layout_bindings[0] =
	(VkDescriptorSetLayoutBinding)
	{
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.pImmutableSamplers = NULL
	};

	vk_result = vk->vk_table.vkCreateDescriptorSetLayout(vk->vk_device,
		&vk_set_layout_info, NULL, &vk->vk_scene.mesh.depth_map_set_layout);
	hard_assert_eq(vk_result, VK_SUCCESS);


	VkPushConstantRange vk_push_constants[] =
	{
		{
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
			.offset = 0,
			.size = sizeof(vk_mesh_frag_constant_data_t)
		}
	};

	VkDescriptorSetLayout vk_set_layouts[] =
	{
		vk->vk_scene.mesh.ubo_set_layout,
		vk->vk_scene.mesh.texture_set_layout,
		vk->vk_scene.mesh.depth_map_set_layout
	};

	VkPipelineLayoutCreateInfo vk_pipeline_layout_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.setLayoutCount = MACRO_ARRAY_LEN(vk_set_layouts),
		.pSetLayouts = vk_set_layouts,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = vk_push_constants
	};

	vk_result = vk->vk_table.vkCreatePipelineLayout(vk->vk_device,
		&vk_pipeline_layout_info, NULL, &vk->vk_scene.mesh.pipeline_layout);
	hard_assert_eq(vk_result, VK_SUCCESS);


	VkGraphicsPipelineCreateInfo vk_pipeline_info =
	{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.stageCount = MACRO_ARRAY_LEN(vk_shader_stages),
		.pStages = vk_shader_stages,
		.pVertexInputState = &vk_vertex_input_info,
		.pInputAssemblyState = &vk_input_assembly_info,
		.pTessellationState = NULL,
		.pViewportState = &vk_viewport_info,
		.pRasterizationState = &vk_rasterization_info,
		.pMultisampleState = &vk_multisample_info,
		.pDepthStencilState = &vk_depth_stencil_info,
		.pColorBlendState = &vk_color_blend_info,
		.pDynamicState = &vk_dynamic_state_info,
		.layout = vk->vk_scene.mesh.pipeline_layout,
		.renderPass = vk->vk_scene.render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	static const char* vk_pipeline_cache_path = "cache/vk_mesh_pipeline.bin";
	VkPipelineCache vk_pipeline_cache = vk_init_pipeline_cache(vk, vk_pipeline_cache_path);

	vk_result = vk->vk_table.vkCreateGraphicsPipelines(vk->vk_device,
		vk_pipeline_cache, 1, &vk_pipeline_info, NULL, &vk->vk_scene.mesh.pipeline);
	hard_assert_eq(vk_result, VK_SUCCESS);

	vk_free_pipeline_cache(vk, vk_pipeline_cache_path, vk_pipeline_cache);

	for(uint32_t i = 0; i < MACRO_ARRAY_LEN(vk_shader_stages); ++i)
	{
		vk_destroy_shader(vk, vk_shader_stages[i].module);
	}
}


private void
vk_free_mesh_pipeline(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->vk_table.vkDestroyPipeline(vk->vk_device, vk->vk_scene.mesh.pipeline, NULL);
	vk->vk_table.vkDestroyPipelineLayout(vk->vk_device, vk->vk_scene.mesh.pipeline_layout, NULL);
	vk->vk_table.vkDestroyDescriptorSetLayout(vk->vk_device, vk->vk_scene.mesh.depth_map_set_layout, NULL);
	vk->vk_table.vkDestroyDescriptorSetLayout(vk->vk_device, vk->vk_scene.mesh.texture_set_layout, NULL);
	vk->vk_table.vkDestroyDescriptorSetLayout(vk->vk_device, vk->vk_scene.mesh.ubo_set_layout, NULL);
}


private void
vk_init_mesh_consts(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_ubo_buffer(vk, sizeof(vk_mesh_vert_ubo_data_t),
		&vk->vk_scene.mesh.ubo_buffer, &vk->vk_scene.mesh.ubo_set,
		vk->vk_scene.mesh.ubo_set_layout);
}


private void
vk_free_mesh_consts(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_buffer(vk, &vk->vk_scene.mesh.ubo_buffer);
}


private void
vk_init_depth_map_pipeline(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkPipelineShaderStageCreateInfo vk_shader_stages[] =
	{
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vk_create_shader(vk, "shaders/depth_map.vert.spv"),
			.pName = "main",
			.pSpecializationInfo = NULL
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = vk_create_shader(vk, "shaders/depth_map.frag.spv"),
			.pName = "main",
			.pSpecializationInfo = NULL
		}
	};

	VkDynamicState vk_dynamic_states[] =
	{
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};

	VkPipelineDynamicStateCreateInfo vk_dynamic_state_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.dynamicStateCount = MACRO_ARRAY_LEN(vk_dynamic_states),
		.pDynamicStates = vk_dynamic_states
	};

	VkPipelineVertexInputStateCreateInfo vk_vertex_input_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.vertexBindingDescriptionCount = 0,
		.pVertexBindingDescriptions = NULL,
		.vertexAttributeDescriptionCount = 0,
		.pVertexAttributeDescriptions = NULL
	};

	VkPipelineInputAssemblyStateCreateInfo vk_input_assembly_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE
	};

	VkPipelineViewportStateCreateInfo vk_viewport_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.viewportCount = 1,
		.pViewports = NULL,
		.scissorCount = 1,
		.pScissors = NULL
	};

	VkPipelineRasterizationStateCreateInfo vk_rasterization_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.depthBiasConstantFactor = 0.0f,
		.depthBiasClamp = 0.0f,
		.depthBiasSlopeFactor = 0.0f,
		.lineWidth = 1.0f
	};

	VkPipelineMultisampleStateCreateInfo vk_multisample_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.rasterizationSamples = vk->vk_samples,
		.sampleShadingEnable = !!vk->vk_sample_shading,
		.minSampleShading = 0.2f,
		.pSampleMask = NULL,
		.alphaToCoverageEnable = VK_FALSE,
		.alphaToOneEnable = VK_FALSE
	};

	VkPipelineDepthStencilStateCreateInfo vk_depth_stencil_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.depthTestEnable = VK_FALSE,
		.depthWriteEnable = VK_FALSE,
		.depthCompareOp = VK_COMPARE_OP_NEVER,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
		.front = {0},
		.back = {0},
		.minDepthBounds = 0.0f,
		.maxDepthBounds = 1.0f
	};

	VkPipelineColorBlendAttachmentState vk_color_blend_attachment =
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

	VkPipelineColorBlendStateCreateInfo vk_color_blend_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.logicOpEnable = VK_FALSE,
		.logicOp = VK_LOGIC_OP_COPY,
		.attachmentCount = 1,
		.pAttachments = &vk_color_blend_attachment,
		.blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f }
	};

	VkDescriptorSetLayoutBinding vk_set_layout_bindings[] =
	{
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
			.pImmutableSamplers = NULL
		}
	};

	VkDescriptorSetLayoutCreateInfo vk_set_layout_info =
	{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.bindingCount = MACRO_ARRAY_LEN(vk_set_layout_bindings),
		.pBindings = vk_set_layout_bindings
	};

	VkResult vk_result = vk->vk_table.vkCreateDescriptorSetLayout(vk->vk_device,
		&vk_set_layout_info, NULL, &vk->vk_scene.depth_map.set_layout);
	hard_assert_eq(vk_result, VK_SUCCESS);


	VkDescriptorSetLayout vk_set_layouts[] =
	{
		vk->vk_scene.depth_map.set_layout
	};

	VkPipelineLayoutCreateInfo vk_pipeline_layout_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.setLayoutCount = MACRO_ARRAY_LEN(vk_set_layouts),
		.pSetLayouts = vk_set_layouts,
		.pushConstantRangeCount = 0,
		.pPushConstantRanges = NULL
	};

	vk_result = vk->vk_table.vkCreatePipelineLayout(vk->vk_device,
		&vk_pipeline_layout_info, NULL, &vk->vk_scene.depth_map.pipeline_layout);
	hard_assert_eq(vk_result, VK_SUCCESS);


	VkGraphicsPipelineCreateInfo vk_pipeline_info =
	{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.stageCount = MACRO_ARRAY_LEN(vk_shader_stages),
		.pStages = vk_shader_stages,
		.pVertexInputState = &vk_vertex_input_info,
		.pInputAssemblyState = &vk_input_assembly_info,
		.pTessellationState = NULL,
		.pViewportState = &vk_viewport_info,
		.pRasterizationState = &vk_rasterization_info,
		.pMultisampleState = &vk_multisample_info,
		.pDepthStencilState = &vk_depth_stencil_info,
		.pColorBlendState = &vk_color_blend_info,
		.pDynamicState = &vk_dynamic_state_info,
		.layout = vk->vk_scene.depth_map.pipeline_layout,
		.renderPass = vk->vk_scene.render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	static const char* vk_pipeline_cache_path = "cache/vk_depth_map_pipeline.bin";
	VkPipelineCache vk_pipeline_cache = vk_init_pipeline_cache(vk, vk_pipeline_cache_path);

	vk_result = vk->vk_table.vkCreateGraphicsPipelines(vk->vk_device,
		vk_pipeline_cache, 1, &vk_pipeline_info, NULL, &vk->vk_scene.depth_map.pipeline);
	hard_assert_eq(vk_result, VK_SUCCESS);

	vk_free_pipeline_cache(vk, vk_pipeline_cache_path, vk_pipeline_cache);

	for(uint32_t i = 0; i < MACRO_ARRAY_LEN(vk_shader_stages); ++i)
	{
		vk_destroy_shader(vk, vk_shader_stages[i].module);
	}
}


private void
vk_free_depth_map_pipeline(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->vk_table.vkDestroyPipeline(vk->vk_device, vk->vk_scene.depth_map.pipeline, NULL);
	vk->vk_table.vkDestroyPipelineLayout(vk->vk_device, vk->vk_scene.depth_map.pipeline_layout, NULL);
	vk->vk_table.vkDestroyDescriptorSetLayout(vk->vk_device, vk->vk_scene.depth_map.set_layout, NULL);
}


private void
vk_init_depth_map_consts(
	vk_t vk
	)
{
	assert_not_null(vk);
}


private void
vk_free_depth_map_consts(
	vk_t vk
	)
{
	assert_not_null(vk);
}


private void
vk_init_scene(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_scene_pass(vk);
	vk_init_skybox_pipeline(vk);
	vk_init_skybox_consts(vk);
	vk_init_mesh_pipeline(vk);
	vk_init_mesh_consts(vk);
	vk_init_depth_map_pipeline(vk);
	vk_init_depth_map_consts(vk);
}


private void
vk_free_scene(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_depth_map_consts(vk);
	vk_free_depth_map_pipeline(vk);
	vk_free_mesh_consts(vk);
	vk_free_mesh_pipeline(vk);
	vk_free_skybox_consts(vk);
	vk_free_skybox_pipeline(vk);
	vk_free_scene_pass(vk);
}


private void
vk_init_pipelines(
	vk_t vk
	)
{
	assert_not_null(vk);

	simulation_model_info_t info = simulation_get_model_info(vk->simulation);

	VkDescriptorPoolSize vk_pool_sizes[] =
	{
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1
		},
		{
			.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = info.material_count + 2 + VK_MAX_FRAMES
		}
	};

	VkDescriptorPoolCreateInfo vk_pool_info =
	{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.maxSets = info.material_count + 3 + VK_MAX_FRAMES,
		.poolSizeCount = MACRO_ARRAY_LEN(vk_pool_sizes),
		.pPoolSizes = vk_pool_sizes
	};

	VkResult vk_result = vk->vk_table.vkCreateDescriptorPool(
		vk->vk_device, &vk_pool_info, NULL, &vk->vk_descriptor_pool);
	hard_assert_eq(vk_result, VK_SUCCESS);

	vk_init_samplers(vk);
	vk_init_shadow(vk);
	vk_init_scene(vk);
}


private void
vk_free_pipelines(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_scene(vk);
	vk_free_shadow(vk);
	vk_free_samplers(vk);

	vk->vk_table.vkDestroyDescriptorPool(vk->vk_device, vk->vk_descriptor_pool, NULL);
}


private void
vk_init_extent_images(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_image(vk, &vk->vk_scene.depth_image);
	vk_init_image(vk, &vk->vk_scene.multisampled_image);
}


private void
vk_free_extent_images(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_image(vk, &vk->vk_scene.depth_image);
	vk_free_image(vk, &vk->vk_scene.multisampled_image);
}


private void
vk_init_models(
	vk_t vk
	)
{
	assert_not_null(vk);

	simulation_model_info_t info = simulation_get_model_info(vk->simulation);
	vk->vk_model_count = info.model_count;
	vk->vk_material_count = info.material_count;

	VkDescriptorImageInfo vk_descriptor_images[vk->vk_material_count];
	VkDescriptorImageInfo* vk_descriptor_image = vk_descriptor_images;

	VkWriteDescriptorSet vk_descriptor_writes[vk->vk_material_count];
	VkWriteDescriptorSet* vk_descriptor_write = vk_descriptor_writes;

	VkDescriptorSet vk_sets[vk->vk_material_count];
	VkDescriptorSet* vk_set = vk_sets;

	VkDescriptorSetLayout vk_set_layouts[vk->vk_material_count];
	VkDescriptorSetLayout* vk_set_layout = vk_set_layouts;
	VkDescriptorSetLayout* vk_set_layout_end =
		vk_set_layout + vk->vk_material_count;

	while(vk_set_layout < vk_set_layout_end)
	{
		*(vk_set_layout++) = vk->vk_scene.mesh.texture_set_layout;
	}

	VkDescriptorSetAllocateInfo vk_set_info =
	{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.pNext = NULL,
		.descriptorPool = vk->vk_descriptor_pool,
		.descriptorSetCount = vk->vk_material_count,
		.pSetLayouts = vk_set_layouts
	};

	VkResult vk_result = vk->vk_table.vkAllocateDescriptorSets(
		vk->vk_device, &vk_set_info, vk_sets);
	hard_assert_eq(vk_result, VK_SUCCESS);


	vk->vk_materials = alloc_malloc(sizeof(*vk->vk_materials) * vk->vk_material_count);
	assert_ptr(vk->vk_materials, sizeof(*vk->vk_materials) * vk->vk_material_count);

	vk->vk_models = alloc_malloc(sizeof(*vk->vk_models) * vk->vk_model_count);
	assert_ptr(vk->vk_models, sizeof(*vk->vk_models) * vk->vk_model_count);

	vk_material_t* vk_material = vk->vk_materials;
	vk_model_t* vk_model = vk->vk_models;

	for(uint32_t i = 0; i < vk->vk_model_count; ++i)
	{
		model_t* model = info.models[i];

		vk_model->meshes = alloc_malloc(sizeof(*vk_model->meshes) * model->mesh_count);
		assert_ptr(vk_model->meshes, sizeof(*vk_model->meshes) * model->mesh_count);
		vk_model->mesh_count = model->mesh_count;

		vk_mesh_t* vk_mesh = vk_model->meshes;

		mesh_t* mesh = model->meshes;
		mesh_t* mesh_end = mesh + model->mesh_count;

		while(mesh < mesh_end)
		{
			vk_mesh->material_idx = vk_material - vk->vk_materials + mesh->material_idx;
			vk_mesh->vertex_count = mesh->vertex_count;
			vk_mesh->index_count = mesh->index_count;

			vk_init_vertex_buffer(vk, sizeof(vk_depth_vertex_data_t) *
				mesh->vertex_count, &vk_mesh->depth_vertex_buffer);
			vk_copy_to_buffer(vk, &vk_mesh->depth_vertex_buffer,
				mesh->vertices, sizeof(*mesh->vertices) * mesh->vertex_count);

			vk_mesh_vertex_data_t* vk_vertex_data =
				alloc_malloc(sizeof(*vk_vertex_data) * mesh->vertex_count);
			assert_ptr(vk_vertex_data, sizeof(*vk_vertex_data) * mesh->vertex_count);

			vk_mesh_vertex_data_t* vk_data = vk_vertex_data;
			vk_mesh_vertex_data_t* vk_data_end = vk_data + mesh->vertex_count;

			vec3* vertex = mesh->vertices;
			vec3* normal = mesh->normals;
			vec2* coord = mesh->coords;

			while(vk_data < vk_data_end)
			{
				glm_vec3_copy(*vertex, vk_data->position);
				glm_vec3_copy(*normal, vk_data->normal);
				glm_vec2_copy(*coord, vk_data->coords);

				++vk_data;
				++vertex;
				++normal;
				++coord;
			}

			vk_init_vertex_buffer(vk, sizeof(*vk_vertex_data) *
				mesh->vertex_count, &vk_mesh->mesh_vertex_buffer);
			vk_copy_to_buffer(vk, &vk_mesh->mesh_vertex_buffer,
				vk_vertex_data, sizeof(*vk_vertex_data) * mesh->vertex_count);

			alloc_free(vk_vertex_data, sizeof(*vk_vertex_data) * mesh->vertex_count);

			vk_init_index_buffer(vk, sizeof(*mesh->indexes) *
				mesh->index_count, &vk_mesh->index_buffer);
			vk_copy_to_buffer(vk, &vk_mesh->index_buffer,
				mesh->indexes, sizeof(*mesh->indexes) * mesh->index_count);

			++vk_mesh;
			++mesh;
		}


		material_t* material = model->materials;
		material_t* material_end = material + model->material_count;

		while(material < material_end)
		{
			vk_material->texture =
			(vk_image_t)
			{
				.type = VK_IMAGE_TYPE_TEXTURE_2D
			};

			if(!str_is_empty(material->texture))
			{
				vk_material->texture.path = str_init_copy(material->texture);
			}
			else
			{
				vk_material->texture.path = str_init_copy_cstr("assets/blank.png");
			}

			vk_init_image(vk, &vk_material->texture);

			glm_vec4_copy(material->ambient, vk_material->ambient);
			glm_vec4_copy(material->diffuse, vk_material->diffuse);

			*vk_descriptor_image =
			(VkDescriptorImageInfo)
			{
				.sampler = vk->vk_image_sampler,
				.imageView = vk_material->texture.view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
			};

			*vk_descriptor_write =
			(VkWriteDescriptorSet)
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.pNext = NULL,
				.dstSet = *vk_set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = vk_descriptor_image,
				.pBufferInfo = NULL,
				.pTexelBufferView = NULL
			};

			vk_material->set = *vk_set;

			++vk_material;
			++material;
			++vk_descriptor_image;
			++vk_descriptor_write;
			++vk_set;
		}

		vk_init_vertex_buffer(vk, sizeof(vk_model_instance_data_t) *
			VK_MAX_INSTANCES, &vk_model->instance_buffer);

		++vk_model;
	}

	vk->vk_table.vkUpdateDescriptorSets(vk->vk_device,
		vk->vk_material_count, vk_descriptor_writes, 0, NULL);
}


private void
vk_free_models(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_model_t* vk_model = vk->vk_models;
	vk_model_t* vk_model_end = vk_model + vk->vk_model_count;

	while(vk_model < vk_model_end)
	{
		vk_free_buffer(vk, &vk_model->instance_buffer);

		vk_mesh_t* vk_mesh = vk_model->meshes;
		vk_mesh_t* vk_mesh_end = vk_mesh + vk_model->mesh_count;

		while(vk_mesh < vk_mesh_end)
		{
			vk_free_buffer(vk, &vk_mesh->index_buffer);
			vk_free_buffer(vk, &vk_mesh->mesh_vertex_buffer);
			vk_free_buffer(vk, &vk_mesh->depth_vertex_buffer);

			++vk_mesh;
		}

		alloc_free(vk_model->meshes, sizeof(*vk_model->meshes) * vk_model->mesh_count);

		++vk_model;
	}

	alloc_free(vk->vk_models, sizeof(*vk->vk_models) * vk->vk_model_count);

	vk_material_t* vk_material = vk->vk_materials;
	vk_material_t* vk_material_end = vk_material + vk->vk_material_count;

	while(vk_material < vk_material_end)
	{
		vk_free_image(vk, &vk_material->texture);
		str_free(vk_material->texture.path);
		++vk_material;
	}

	alloc_free(vk->vk_materials, sizeof(*vk->vk_materials) * vk->vk_material_count);
}


private void
vk_free_swapchain(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->vk_table.vkDestroySwapchainKHR(vk->vk_device, vk->vk_swapchain, NULL);
}


private void
vk_init_swapchain(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkSwapchainCreateInfoKHR vk_swapchain_info =
	{
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.pNext = NULL,
		.flags = 0,
		.surface = vk->vk_surface,
		.minImageCount = vk->vk_image_count,
		.imageFormat = vk->vk_format,
		.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
		.imageExtent = vk->vk_extent,
		.imageArrayLayers = 1,
		.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
		.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 0,
		.pQueueFamilyIndices = NULL,
		.preTransform = vk->vk_transform,
		.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		.presentMode = vk->vk_present_mode,
		.clipped = VK_TRUE,
		.oldSwapchain = vk->vk_swapchain
	};

	VkSwapchainKHR vk_swapchain;
	VkResult vk_result = vk->vk_table.vkCreateSwapchainKHR(
		vk->vk_device, &vk_swapchain_info, NULL, &vk_swapchain);
	hard_assert_eq(vk_result, VK_SUCCESS);

	vk_free_swapchain(vk);
	vk->vk_swapchain = vk_swapchain;
}


private void
vk_init_frames(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkCommandBuffer vk_command_buffers[vk->vk_image_count];
	VkCommandBuffer* vk_command_buffer = vk_command_buffers;

	VkCommandBufferAllocateInfo vk_command_buffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = NULL,
		.commandPool = vk->vk_command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = vk->vk_image_count
	};

	VkResult vk_result = vk->vk_table.vkAllocateCommandBuffers(
		vk->vk_device, &vk_command_buffer_info, vk_command_buffers);
	hard_assert_eq(vk_result, VK_SUCCESS);


	VkSemaphoreCreateInfo vk_semaphore_info =
	{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0
	};

	VkFenceCreateInfo vk_fence_info =
	{
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.pNext = NULL,
		.flags = VK_FENCE_CREATE_SIGNALED_BIT
	};

	vk_barrier_t* vk_barrier = vk->vk_barriers;
	vk_barrier_t* vk_barrier_end = vk_barrier + vk->vk_image_count;

	while(vk_barrier < vk_barrier_end)
	{
		VkSemaphore* vk_semaphore = vk_barrier->semaphores;
		VkSemaphore* vk_semaphore_end = vk_semaphore + MACRO_ARRAY_LEN(vk_barrier->semaphores);

		while(vk_semaphore < vk_semaphore_end)
		{
			vk_result = vk->vk_table.vkCreateSemaphore(
				vk->vk_device, &vk_semaphore_info, NULL, vk_semaphore++);
			hard_assert_eq(vk_result, VK_SUCCESS);
		}

		VkFence* vk_fence = vk_barrier->fences;
		VkFence* vk_fence_end = vk_fence + MACRO_ARRAY_LEN(vk_barrier->fences);

		while(vk_fence < vk_fence_end)
		{
			vk_result = vk->vk_table.vkCreateFence(
				vk->vk_device, &vk_fence_info, NULL, vk_fence++);
			hard_assert_eq(vk_result, VK_SUCCESS);
		}

		vk_barrier->command_buffer = *vk_command_buffer;

		++vk_barrier;
		++vk_command_buffer;
	}

	vk->vk_barrier = vk->vk_barriers;
}


private void
vk_free_frames(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkCommandBuffer vk_command_buffers[vk->vk_image_count];
	VkCommandBuffer* vk_command_buffer = vk_command_buffers;

	vk_barrier_t* vk_barrier = vk->vk_barriers;
	vk_barrier_t* vk_barrier_end = vk_barrier + vk->vk_image_count;

	while(vk_barrier < vk_barrier_end)
	{
		*vk_command_buffer = vk_barrier->command_buffer;

		VkFence* vk_fence = vk_barrier->fences;
		VkFence* vk_fence_end = vk_fence + MACRO_ARRAY_LEN(vk_barrier->fences);

		while(vk_fence < vk_fence_end)
		{
			vk->vk_table.vkDestroyFence(vk->vk_device, *vk_fence, NULL);
			++vk_fence;
		}

		VkSemaphore* vk_semaphore = vk_barrier->semaphores;
		VkSemaphore* vk_semaphore_end = vk_semaphore + MACRO_ARRAY_LEN(vk_barrier->semaphores);

		while(vk_semaphore < vk_semaphore_end)
		{
			vk->vk_table.vkDestroySemaphore(vk->vk_device, *vk_semaphore, NULL);
			++vk_semaphore;
		}

		++vk_barrier;
		++vk_command_buffer;
	}

	vk->vk_table.vkFreeCommandBuffers(vk->vk_device,
		vk->vk_command_pool, vk->vk_image_count, vk_command_buffers);
}


private void
vk_init_framebuffers(
	vk_t vk
	)
{
	assert_not_null(vk);

	uint32_t image_count;
	VkResult vk_result = vk->vk_table.vkGetSwapchainImagesKHR(
		vk->vk_device, vk->vk_swapchain, &image_count, NULL);
	hard_assert_eq(vk_result, VK_SUCCESS);

	assert_lt(image_count, VK_MAX_FRAMES);
	assert_ge(image_count, vk->vk_image_count);

	bool vk_reinit_frames = vk->vk_image_count != image_count;
	vk->vk_image_count = image_count;
	if(vk_reinit_frames)
	{
		vk_free_frames(vk);
		vk_init_frames(vk);
	}

	VkImage vk_images[image_count];
	vk_result = vk->vk_table.vkGetSwapchainImagesKHR(
		vk->vk_device, vk->vk_swapchain, &image_count, vk_images);
	hard_assert_eq(vk_result, VK_SUCCESS);


	vk_frame_t* vk_frame = vk->vk_frames;
	vk_frame_t* vk_frame_end = vk_frame + vk->vk_image_count;

	VkImage* vk_image = vk_images;

	while(vk_frame < vk_frame_end)
	{
		vk_frame->scene.image = *vk_image;

		VkImageViewCreateInfo vk_image_view_info =
		{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.image = vk_frame->scene.image,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = vk->vk_format,
			.components =
			{
				.r = VK_COMPONENT_SWIZZLE_IDENTITY,
				.g = VK_COMPONENT_SWIZZLE_IDENTITY,
				.b = VK_COMPONENT_SWIZZLE_IDENTITY,
				.a = VK_COMPONENT_SWIZZLE_IDENTITY
			},
			.subresourceRange =
			{
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			}
		};

		vk_result = vk->vk_table.vkCreateImageView(
			vk->vk_device, &vk_image_view_info, NULL, &vk_frame->scene.image_view);
		hard_assert_eq(vk_result, VK_SUCCESS);

		VkImageView vk_attachments[] =
		{
			vk->vk_scene.multisampled_image.view,
			vk->vk_scene.depth_image.view,
			vk_frame->scene.image_view
		};

		VkFramebufferCreateInfo vk_framebuffer_info =
		{
			.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.renderPass = vk->vk_scene.render_pass,
			.attachmentCount = MACRO_ARRAY_LEN(vk_attachments),
			.pAttachments = vk_attachments,
			.width = vk->vk_extent.width,
			.height = vk->vk_extent.height,
			.layers = 1
		};

		vk_result = vk->vk_table.vkCreateFramebuffer(
			vk->vk_device, &vk_framebuffer_info, NULL, &vk_frame->scene.framebuffer);
		hard_assert_eq(vk_result, VK_SUCCESS);


		vk_frame->shadow.image.type = VK_IMAGE_TYPE_DEPTH_MAP;
		vk_init_image(vk, &vk_frame->shadow.image);

		vk_framebuffer_info.renderPass = vk->vk_shadow.render_pass;
		vk_framebuffer_info.attachmentCount = 1;
		vk_framebuffer_info.pAttachments = &vk_frame->shadow.image.view;
		vk_framebuffer_info.width = vk_frame->shadow.image.width;
		vk_framebuffer_info.height = vk_frame->shadow.image.height;

		vk_result = vk->vk_table.vkCreateFramebuffer(
			vk->vk_device, &vk_framebuffer_info, NULL, &vk_frame->shadow.framebuffer);


		VkDescriptorSetAllocateInfo vk_set_info =
		{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.pNext = NULL,
			.descriptorPool = vk->vk_descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &vk->vk_scene.mesh.depth_map_set_layout
		};

		VkResult vk_result = vk->vk_table.vkAllocateDescriptorSets(
			vk->vk_device, &vk_set_info, &vk_frame->shadow.set);
		hard_assert_eq(vk_result, VK_SUCCESS);

		VkDescriptorImageInfo vk_image_info =
		{
			.sampler = vk->vk_depth_sampler,
			.imageView = vk_frame->shadow.image.view,
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
		};

		VkWriteDescriptorSet vk_write_set =
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.pNext = NULL,
			.dstSet = vk_frame->shadow.set,
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &vk_image_info,
			.pBufferInfo = NULL,
			.pTexelBufferView = NULL
		};

		vk->vk_table.vkUpdateDescriptorSets(vk->vk_device, 1, &vk_write_set, 0, NULL);


		++vk_frame;
		++vk_image;
	}
}


private void
vk_free_framebuffers(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_frame_t* vk_frame = vk->vk_frames;
	vk_frame_t* vk_frame_end = vk_frame + vk->vk_image_count;

	while(vk_frame < vk_frame_end)
	{
		vk_free_image(vk, &vk_frame->shadow.image);
		vk->vk_table.vkDestroyFramebuffer(vk->vk_device, vk_frame->shadow.framebuffer, NULL);

		vk->vk_table.vkDestroyFramebuffer(vk->vk_device, vk_frame->scene.framebuffer, NULL);
		vk->vk_table.vkDestroyImageView(vk->vk_device, vk_frame->scene.image_view, NULL);

		++vk_frame;
	}
}


private void
vk_device_wait_idle(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkResult vk_result = vk->vk_table.vkDeviceWaitIdle(vk->vk_device);
	hard_assert_eq(vk_result, VK_SUCCESS);
}


private void
vk_recreate_swapchain(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_device_wait_idle(vk);

	vk_free_extent_images(vk);
	vk_get_extent(vk);
	vk_init_extent_images(vk);

	vk_free_framebuffers(vk);
	vk_init_swapchain(vk);
	vk_init_framebuffers(vk);

	vk_store_bool(&vk->window_resize_bool, false);
}


private void
vk_draw_shadow(
	vk_t vk,
	vk_frame_t* vk_frame,
	simulation_transform_t* transform,
	vk_entities_per_model_t* vk_entity_data
	)
{
	assert_not_null(vk);
	assert_not_null(vk_frame);

	VkClearValue vk_clear_values[] =
	{
		{
			.depthStencil = { 1.0f, 0 }
		}
	};

	VkRenderPassBeginInfo vk_render_pass_begin_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.pNext = NULL,
		.renderPass = vk->vk_shadow.render_pass,
		.framebuffer = vk_frame->shadow.framebuffer,
		.renderArea =
		{
			.offset = { 0, 0 },
			.extent = { vk_frame->shadow.image.width, vk_frame->shadow.image.height }
		},
		.clearValueCount = MACRO_ARRAY_LEN(vk_clear_values),
		.pClearValues = vk_clear_values
	};

	vk->vk_table.vkCmdBeginRenderPass(vk->vk_barrier->command_buffer,
		&vk_render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

	vk->vk_table.vkCmdBindPipeline(vk->vk_barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, vk->vk_shadow.depth.pipeline);

	vk_depth_vert_constant_data_t vk_depth_vert_constant_data;
	glm_mat4_copy(transform->light_transform, vk_depth_vert_constant_data.transform);

	vk->vk_table.vkCmdPushConstants(vk->vk_barrier->command_buffer,
		vk->vk_shadow.depth.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
		0, sizeof(vk_depth_vert_constant_data), &vk_depth_vert_constant_data);

	vk_entities_per_model_t* vk_entities_per_model = vk_entity_data;
	vk_entities_per_model_t* vk_entities_per_model_end =
		vk_entities_per_model + vk->vk_model_count;

	vk_model_t* vk_model = vk->vk_models;

	while(vk_entities_per_model < vk_entities_per_model_end)
	{
		if(vk_entities_per_model->entities_used != 0)
		{
			hard_assert_le(vk_entities_per_model->entities_used, VK_MAX_INSTANCES);

			vk->vk_table.vkCmdBindVertexBuffers(vk->vk_barrier->command_buffer,
				1, 1, &vk_model->instance_buffer.buffer, (VkDeviceSize[]){0});

			vk_mesh_t* vk_mesh = vk_model->meshes;
			vk_mesh_t* vk_mesh_end = vk_mesh + vk_model->mesh_count;

			while(vk_mesh < vk_mesh_end)
			{
				vk->vk_table.vkCmdBindVertexBuffers(vk->vk_barrier->command_buffer,
					0, 1, &vk_mesh->depth_vertex_buffer.buffer, (VkDeviceSize[]){0});

				vk->vk_table.vkCmdBindIndexBuffer(vk->vk_barrier->command_buffer,
					vk_mesh->index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

				vk->vk_table.vkCmdDrawIndexed(vk->vk_barrier->command_buffer,
					vk_mesh->index_count, vk_entities_per_model->entities_used, 0, 0, 0);

				++vk_mesh;
			}
		}

		++vk_entities_per_model;
		++vk_model;
	}

	vk->vk_table.vkCmdEndRenderPass(vk->vk_barrier->command_buffer);
}


private void
vk_draw_mesh(
	vk_t vk,
	vk_frame_t* vk_frame,
	simulation_transform_t* transform,
	vk_entities_per_model_t* vk_entity_data
	)
{
	assert_not_null(vk);
	assert_not_null(vk_frame);

	VkClearValue vk_clear_values[] =
	{
		{
			.color = {{ 0.0f, 0.0f, 0.0f, 1.0f }}
		},
		{
			.depthStencil = { 0.0f, 0 }
		}
	};

	VkRenderPassBeginInfo vk_render_pass_begin_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.pNext = NULL,
		.renderPass = vk->vk_scene.render_pass,
		.framebuffer = vk_frame->scene.framebuffer,
		.renderArea =
		{
			.offset = { 0, 0 },
			.extent = vk->vk_extent
		},
		.clearValueCount = MACRO_ARRAY_LEN(vk_clear_values),
		.pClearValues = vk_clear_values
	};

	vk->vk_table.vkCmdBeginRenderPass(vk->vk_barrier->command_buffer,
		&vk_render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

	vk->vk_table.vkCmdSetViewport(vk->vk_barrier->command_buffer, 0, 1, &vk->vk_viewport);
	vk->vk_table.vkCmdSetScissor(vk->vk_barrier->command_buffer, 0, 1, &vk->vk_scissor);

	vk_mesh_vert_ubo_data_t vk_mesh_vert_ubo_data;
	glm_mat4_copy(transform->projection, vk_mesh_vert_ubo_data.projection);
	glm_mat4_copy(transform->view, vk_mesh_vert_ubo_data.view);
	glm_mat4_copy(transform->light_transform, vk_mesh_vert_ubo_data.light_transform);
	glm_vec4_copy(transform->light_position, vk_mesh_vert_ubo_data.light_position);

	vk_copy_to_buffer(vk, &vk->vk_scene.mesh.ubo_buffer,
		&vk_mesh_vert_ubo_data, sizeof(vk_mesh_vert_ubo_data));

	vk_skybox_constant_data_t vk_skybox_constant_data;
	glm_mat4_copy(vk_mesh_vert_ubo_data.projection, vk_skybox_constant_data.transform);

	mat4 view;
	glm_mat4_copy(vk_mesh_vert_ubo_data.view, view);
	view[3][0] = 0.0f;
	view[3][1] = 0.0f;
	view[3][2] = 0.0f;
	glm_mat4_mul(vk_skybox_constant_data.transform, view, vk_skybox_constant_data.transform);

	vk->vk_table.vkCmdBindPipeline(vk->vk_barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, vk->vk_scene.skybox.pipeline);

	vk->vk_table.vkCmdPushConstants(vk->vk_barrier->command_buffer,
		vk->vk_scene.skybox.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
		0, sizeof(vk_skybox_constant_data), &vk_skybox_constant_data);

	vk->vk_table.vkCmdBindVertexBuffers(vk->vk_barrier->command_buffer,
		0, 1, &vk->vk_scene.skybox.vertex_buffer.buffer, (VkDeviceSize[]){0});

	vk->vk_table.vkCmdBindIndexBuffer(vk->vk_barrier->command_buffer,
		vk->vk_scene.skybox.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT16);

	vk->vk_table.vkCmdBindDescriptorSets(vk->vk_barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, vk->vk_scene.skybox.pipeline_layout,
		0, 1, &vk->vk_scene.skybox.set, 0, NULL);

	vk->vk_table.vkCmdDrawIndexed(vk->vk_barrier->command_buffer,
		MACRO_ARRAY_LEN(vk_skybox_index_data), 1, 0, 0, 0);


	vk->vk_table.vkCmdBindPipeline(vk->vk_barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, vk->vk_scene.mesh.pipeline);

	vk->vk_table.vkCmdBindDescriptorSets(vk->vk_barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, vk->vk_scene.mesh.pipeline_layout,
		0, 1, &vk->vk_scene.mesh.ubo_set, 0, NULL);

	vk->vk_table.vkCmdBindDescriptorSets(vk->vk_barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, vk->vk_scene.mesh.pipeline_layout,
		2, 1, &vk_frame->shadow.set, 0, NULL);

	vk_entities_per_model_t* vk_entities_per_model = vk_entity_data;
	vk_entities_per_model_t* vk_entities_per_model_end =
		vk_entities_per_model + vk->vk_model_count;

	vk_model_t* vk_model = vk->vk_models;

	while(vk_entities_per_model < vk_entities_per_model_end)
	{
		if(vk_entities_per_model->entities_used != 0)
		{
			hard_assert_le(vk_entities_per_model->entities_used, VK_MAX_INSTANCES);

			vk->vk_table.vkCmdBindVertexBuffers(vk->vk_barrier->command_buffer,
				1, 1, &vk_model->instance_buffer.buffer, (VkDeviceSize[]){0});

			vk_mesh_t* vk_mesh = vk_model->meshes;
			vk_mesh_t* vk_mesh_end = vk_mesh + vk_model->mesh_count;

			while(vk_mesh < vk_mesh_end)
			{
				vk_mesh_frag_constant_data_t vk_mesh_frag_constant_data;
				glm_vec4_copy(vk->vk_materials[vk_mesh->material_idx].ambient,
					vk_mesh_frag_constant_data.ambient);
				glm_vec4_copy(vk->vk_materials[vk_mesh->material_idx].diffuse,
					vk_mesh_frag_constant_data.diffuse);

				vk->vk_table.vkCmdPushConstants(vk->vk_barrier->command_buffer,
					vk->vk_scene.mesh.pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
					0, sizeof(vk_mesh_frag_constant_data), &vk_mesh_frag_constant_data);

				vk->vk_table.vkCmdBindVertexBuffers(vk->vk_barrier->command_buffer,
					0, 1, &vk_mesh->mesh_vertex_buffer.buffer, (VkDeviceSize[]){0});

				vk->vk_table.vkCmdBindIndexBuffer(vk->vk_barrier->command_buffer,
					vk_mesh->index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

				vk->vk_table.vkCmdBindDescriptorSets(vk->vk_barrier->command_buffer,
					VK_PIPELINE_BIND_POINT_GRAPHICS, vk->vk_scene.mesh.pipeline_layout,
					1, 1, &vk->vk_materials[vk_mesh->material_idx].set, 0, NULL);

				vk->vk_table.vkCmdDrawIndexed(vk->vk_barrier->command_buffer,
					vk_mesh->index_count, vk_entities_per_model->entities_used, 0, 0, 0);

				++vk_mesh;
			}
		}

		alloc_free(vk_entities_per_model->entities,
			sizeof(*vk_entities_per_model->entities) * vk_entities_per_model->entities_size);

		++vk_entities_per_model;
		++vk_model;
	}

	if(vk->vk_preview_depth_map)
	{
		vk->vk_table.vkCmdBindPipeline(vk->vk_barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, vk->vk_scene.depth_map.pipeline);

		vk->vk_table.vkCmdBindDescriptorSets(vk->vk_barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, vk->vk_scene.depth_map.pipeline_layout,
			0, 1, &vk_frame->shadow.set, 0, NULL);

		vk->vk_table.vkCmdDraw(vk->vk_barrier->command_buffer, 6, 1, 0, 0);
	}

	vk->vk_table.vkCmdEndRenderPass(vk->vk_barrier->command_buffer);
}


private void
vk_draw(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkResult vk_result = vk->vk_table.vkWaitForFences(vk->vk_device, 1,
		vk->vk_barrier->fences + VK_BARRIER_FENCE_IN_FLIGHT, VK_TRUE, UINT64_MAX);
	hard_assert_eq(vk_result, VK_SUCCESS);

	uint32_t image_idx;
	vk_result = vk->vk_table.vkAcquireNextImageKHR(vk->vk_device, vk->vk_swapchain, UINT64_MAX,
		vk->vk_barrier->semaphores[VK_BARRIER_SEMAPHORE_IMAGE_AVAILABLE], VK_NULL_HANDLE, &image_idx);
	if(vk_result == VK_ERROR_OUT_OF_DATE_KHR)
	{
		vk_recreate_swapchain(vk);
		return;
	}

	vk_result = vk->vk_table.vkResetFences(vk->vk_device, 1,
		vk->vk_barrier->fences + VK_BARRIER_FENCE_IN_FLIGHT);
	hard_assert_eq(vk_result, VK_SUCCESS);

	vk_frame_t* vk_frame = vk->vk_frames + image_idx;


	vk_result = vk->vk_table.vkResetCommandBuffer(vk->vk_barrier->command_buffer, 0);
	hard_assert_eq(vk_result, VK_SUCCESS);

	VkCommandBufferBeginInfo vk_command_buffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = NULL,
		.flags = 0,
		.pInheritanceInfo = NULL
	};

	vk_result = vk->vk_table.vkBeginCommandBuffer(
		vk->vk_barrier->command_buffer, &vk_command_buffer_info);
	hard_assert_eq(vk_result, VK_SUCCESS);

	simulation_transform_t transform = simulation_get_transform(
		vk->simulation, vk->vk_extent.width, vk->vk_extent.height);

	uint32_t entity_count;
	simulation_entity_data_t* entity_data =
		simulation_get_entity_data(vk->simulation, &entity_count);

	simulation_entity_data_t* entity = entity_data;
	simulation_entity_data_t* entity_end = entity + entity_count;

	vk_entities_per_model_t* vk_entity_data = alloc_calloc(sizeof(*vk_entity_data) * vk->vk_model_count);
	assert_ptr(vk_entity_data, sizeof(*vk_entity_data) * vk->vk_model_count);

	while(entity < entity_end)
	{
		vk_entities_per_model_t* vk_entities = vk_entity_data + entity->model_idx;

		if(vk_entities->entities_used >= vk_entities->entities_size)
		{
			uint32_t new_size = (vk_entities->entities_size << 1) | 1;

			vk_entities->entities = alloc_recalloc(
				vk_entities->entities,
				sizeof(*vk_entities->entities) * vk_entities->entities_size,
				sizeof(*vk_entities->entities) * new_size
				);
			assert_ptr(vk_entities->entities, sizeof(*vk_entities->entities) * new_size);

			vk_entities->entities_size = new_size;
		}

		vk_entities->entities[vk_entities->entities_used++] = entity;

		++entity;
	}

	vk_entities_per_model_t* vk_entities_per_model = vk_entity_data;
	vk_entities_per_model_t* vk_entities_per_model_end =
		vk_entities_per_model + vk->vk_model_count;

	vk_model_t* vk_model = vk->vk_models;

	while(vk_entities_per_model < vk_entities_per_model_end)
	{
		if(vk_entities_per_model->entities_used != 0)
		{
			hard_assert_le(vk_entities_per_model->entities_used, VK_MAX_INSTANCES);

			simulation_entity_data_t** vk_entity = vk_entities_per_model->entities;
			simulation_entity_data_t** vk_entity_end = vk_entity + vk_entities_per_model->entities_used;

			uint64_t vk_instance_data_size =
				sizeof(vk_model_instance_data_t) * vk_entities_per_model->entities_used;
			vk_model_instance_data_t* vk_instance_data = alloc_malloc(vk_instance_data_size);
			assert_ptr(vk_instance_data, vk_instance_data_size);

			vk_model_instance_data_t* vk_instance = vk_instance_data;

			while(vk_entity < vk_entity_end)
			{
				glm_mat4_copy((*vk_entity)->transform, vk_instance->transform);

				++vk_entity;
				++vk_instance;
			}

			vk_copy_to_buffer(vk, &vk_model->instance_buffer, vk_instance_data, vk_instance_data_size);
			alloc_free(vk_instance_data, vk_instance_data_size);
		}

		++vk_entities_per_model;
		++vk_model;
	}

	vk_draw_shadow(vk, vk_frame, &transform, vk_entity_data);
	vk_draw_mesh(vk, vk_frame, &transform, vk_entity_data);

	alloc_free(vk_entity_data, sizeof(*vk_entity_data) * vk->vk_model_count);

	simulation_free_entity_data(entity_data, entity_count);

	vk_result = vk->vk_table.vkEndCommandBuffer(vk->vk_barrier->command_buffer);
	hard_assert_eq(vk_result, VK_SUCCESS);


	VkPipelineStageFlags vk_wait_stages[] =
	{
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
	};

	VkSubmitInfo vk_submit_info =
	{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = NULL,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = vk->vk_barrier->semaphores + VK_BARRIER_SEMAPHORE_IMAGE_AVAILABLE,
		.pWaitDstStageMask = vk_wait_stages,
		.commandBufferCount = 1,
		.pCommandBuffers = &vk->vk_barrier->command_buffer,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = vk->vk_barrier->semaphores + VK_BARRIER_SEMAPHORE_RENDER_FINISHED
	};

	vk_result = vk->vk_table.vkQueueSubmit(vk->vk_queue, 1,
		&vk_submit_info, vk->vk_barrier->fences[VK_BARRIER_FENCE_IN_FLIGHT]);
	hard_assert_eq(vk_result, VK_SUCCESS);

	VkPresentInfoKHR vk_present_info =
	{
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.pNext = NULL,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = vk->vk_barrier->semaphores + VK_BARRIER_SEMAPHORE_RENDER_FINISHED,
		.swapchainCount = 1,
		.pSwapchains = &vk->vk_swapchain,
		.pImageIndices = &image_idx,
		.pResults = NULL
	};

	vk_result = vk->vk_table.vkQueuePresentKHR(vk->vk_queue, &vk_present_info);
	if(
		vk_result == VK_ERROR_OUT_OF_DATE_KHR ||
		vk_result == VK_SUBOPTIMAL_KHR ||
		vk_load_bool(&vk->window_resize_bool)
		)
	{
		vk_recreate_swapchain(vk);
	}
	else
	{
		hard_assert_eq(vk_result, VK_SUCCESS);
	}

	if(++vk->vk_barrier >= vk->vk_barriers + vk->vk_image_count)
	{
		vk->vk_barrier = vk->vk_barriers;
	}
}


private void
vk_thread_fn(
	vk_t vk
	)
{
	assert_not_null(vk);

	while(vk_load_bool(&vk->vk_running))
	{
		vk_draw(vk);
	}
}


private void
vk_init_thread(
	vk_t vk
	)
{
	assert_not_null(vk);

	atomic_init(&vk->vk_running, true);

	thread_data_t vk_thread_data =
	{
		.fn = (void*) vk_thread_fn,
		.data = vk
	};
	thread_init(&vk->vk_thread, vk_thread_data);
}


private void
vk_free_thread(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_store_bool(&vk->vk_running, false);
	thread_join(vk->vk_thread);

	thread_free(&vk->vk_thread);
}


private void
vk_init_vk(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_instance(vk);
	vk_init_surface(vk);
	vk_init_device(vk);
	vk_init_pipelines(vk);
	vk_init_models(vk);
	vk_init_swapchain(vk);
	vk_init_frames(vk);
	vk_init_framebuffers(vk);

	vk_init_thread(vk);
}


private void
vk_free_vk(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_thread(vk);

	vk_device_wait_idle(vk);

	vk_free_staging_buffer(vk);

	vk_free_framebuffers(vk);
	vk_free_frames(vk);
	vk_free_swapchain(vk);
	vk_free_models(vk);
	vk_free_pipelines(vk);
	vk_free_device(vk);
	vk_free_surface(vk);
	vk_free_instance(vk);
}


private void
vk_window_close_once_fn(
	vk_t vk,
	window_close_event_data_t* event_data
	)
{
	assert_not_null(vk);

	vk->window_close_once_listener = NULL;
	window_close(vk->window);
}


private void
vk_window_free_once_fn(
	vk_t vk,
	window_free_event_data_t* event_data
	)
{
	assert_not_null(vk);

	vk_free_vk(vk);

	window_event_table_t* table = window_get_event_table(vk->window);

	event_target_del(&table->key_up_target, vk->window_key_up_listener);
	event_target_del(&table->key_down_target, vk->window_key_down_listener);
	event_target_del(&table->mouse_move_target, vk->window_mouse_move_listener);
	event_target_del(&table->mouse_up_target, vk->window_mouse_up_listener);
	event_target_del(&table->mouse_down_target, vk->window_mouse_down_listener);
	event_target_del(&table->resize_target, vk->window_resize_listener);
	event_target_del_once(&table->close_target, vk->window_close_once_listener);

	simulation_stop(vk->simulation);
}


private void
vk_window_init_once_fn(
	vk_t vk,
	window_init_event_data_t* event_data
	)
{
	assert_not_null(vk);

	window_show(vk->window);

	vk_init_vk(vk);
}


private void
vk_window_resize_fn(
	vk_t vk,
	window_resize_event_data_t* event_data
	)
{
	assert_not_null(vk);

	if(!vk_exchange_bool(&vk->window_resize_bool, false, true))
	{
		sync_cond_wake(&vk->window_resize_cond);
	}
}


private void
vk_window_mouse_down_fn(
	vk_t vk,
	window_mouse_down_event_data_t* event_data
	)
{
	assert_not_null(vk);

	if(event_data->button == WINDOW_BUTTON_LEFT && event_data->clicks == 1)
	{
		vk->window_mouse_holding = true;
	}
}


private void
vk_window_mouse_up_fn(
	vk_t vk,
	window_mouse_up_event_data_t* event_data
	)
{
	assert_not_null(vk);

	if(event_data->button == WINDOW_BUTTON_LEFT)
	{
		vk->window_mouse_holding = false;
	}
}


private void
vk_window_mouse_move_fn(
	vk_t vk,
	window_mouse_move_event_data_t* event_data
	)
{
	assert_not_null(vk);

	bool modify_angle = vk->window_mouse_holding;

	if(!modify_angle)
	{
		window_info_t info;
		window_get_info(vk->window, &info);

		modify_angle = info.fullscreen && info.rel_mouse_in_fullscreen;
	}

	if(modify_angle)
	{
		vec3 angles =
		{
			event_data->rel_pos.y * VK_WINDOW_SENSITIVITY,
			-event_data->rel_pos.x * VK_WINDOW_SENSITIVITY,
			0.0f
		};
		simulation_modify_angle(vk->simulation, angles);
	}
}


private void
vk_window_key_down_fn(
	vk_t vk,
	window_key_down_event_data_t* event_data
	)
{
	assert_not_null(vk);

	if(
		(
			(event_data->mods & WINDOW_MOD_CTRL_BIT) &&
			(
				event_data->key == WINDOW_KEY_W ||
				event_data->key == WINDOW_KEY_R
				)
			) ||
			event_data->key == WINDOW_KEY_ESCAPE
		)
	{
		window_close(vk->window);
	}
	else if(event_data->key == WINDOW_KEY_F11)
	{
		window_toggle_fullscreen(vk->window);
	}
}


private void
vk_window_key_up_fn(
	vk_t vk,
	window_key_up_event_data_t* event_data
	)
{
	assert_not_null(vk);

	vec3 pos = { 0.0f, 0.0f, 0.0f };


	switch(event_data->key)
	{

	case WINDOW_KEY_W:
	{
		pos[2] = VK_WINDOW_SPEED;
		break;
	}

	case WINDOW_KEY_S:
	{
		pos[2] = -VK_WINDOW_SPEED;
		break;
	}

	case WINDOW_KEY_A:
	{
		pos[0] = VK_WINDOW_SPEED;
		break;
	}

	case WINDOW_KEY_D:
	{
		pos[0] = -VK_WINDOW_SPEED;
		break;
	}

	default: break;

	}


	simulation_modify_position(vk->simulation, pos);
}


private void
vk_window_thread_fn(
	vk_t vk
	)
{
	assert_not_null(vk);

	sigset_t set;
	sigemptyset(&set);
	sigaddset(&set, SIGINT);
	pthread_sigmask(SIG_UNBLOCK, &set, NULL);

	window_manager_run(vk->window_manager);
}


private void
vk_init_window(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->window_manager = window_manager_init();
	vk->window = window_init();

	window_history_t history =
	{
		.extent =
		{
			.x = -1,
			.y = -1,
			.w = vk->window_default_width,
			.h = vk->window_default_height
		},
		.fullscreen = true,
		.rel_mouse_in_fullscreen = true
	};
	window_manager_add(vk->window_manager, vk->window, "Thesis", &history);

	atomic_init(&vk->window_resize_bool, false);
	sync_mtx_init(&vk->window_resize_mtx);
	sync_cond_init(&vk->window_resize_cond);

	vk->window_mouse_holding = false;

	window_event_table_t* table = window_get_event_table(vk->window);

	event_listener_data_t close_once_data =
	{
		.fn = (void*) vk_window_close_once_fn,
		.data = vk
	};
	vk->window_close_once_listener = event_target_once(&table->close_target, close_once_data);

	event_listener_data_t free_once_data =
	{
		.fn = (void*) vk_window_free_once_fn,
		.data = vk
	};
	event_target_once(&table->free_target, free_once_data);

	event_listener_data_t init_once_data =
	{
		.fn = (void*) vk_window_init_once_fn,
		.data = vk
	};
	event_target_once(&table->init_target, init_once_data);

	event_listener_data_t resize_data =
	{
		.fn = (void*) vk_window_resize_fn,
		.data = vk
	};
	vk->window_resize_listener = event_target_add(&table->resize_target, resize_data);

	event_listener_data_t mouse_down_data =
	{
		.fn = (void*) vk_window_mouse_down_fn,
		.data = vk
	};
	vk->window_mouse_down_listener = event_target_add(&table->mouse_down_target, mouse_down_data);

	event_listener_data_t mouse_up_data =
	{
		.fn = (void*) vk_window_mouse_up_fn,
		.data = vk
	};
	vk->window_mouse_up_listener = event_target_add(&table->mouse_up_target, mouse_up_data);

	event_listener_data_t mouse_move_data =
	{
		.fn = (void*) vk_window_mouse_move_fn,
		.data = vk
	};
	vk->window_mouse_move_listener = event_target_add(&table->mouse_move_target, mouse_move_data);

	event_listener_data_t key_down_data =
	{
		.fn = (void*) vk_window_key_down_fn,
		.data = vk
	};
	vk->window_key_down_listener = event_target_add(&table->key_down_target, key_down_data);

	event_listener_data_t key_up_data =
	{
		.fn = (void*) vk_window_key_up_fn,
		.data = vk
	};
	vk->window_key_up_listener = event_target_add(&table->key_up_target, key_up_data);

	thread_data_t thread_data =
	{
		.fn = (void*) vk_window_thread_fn,
		.data = vk
	};
	thread_init(&vk->window_thread, thread_data);
}


private void
vk_free_window(
	vk_t vk
	)
{
	assert_not_null(vk);

	thread_free(&vk->window_thread);

	sync_cond_free(&vk->window_resize_cond);
	sync_mtx_free(&vk->window_resize_mtx);

	window_manager_free(vk->window_manager);
}


private void
vk_free(
	vk_t vk,
	simulation_free_event_data_t* event_data
	)
{
	assert_not_null(vk);

	window_manager_stop_running(vk->window_manager);
	thread_join(vk->window_thread);

	vk_free_window(vk);

	alloc_free(vk, sizeof(*vk));
}


vk_t
vk_init(
	simulation_t simulation
	)
{
	assert_not_null(simulation);

	vk_t vk = alloc_calloc(sizeof(*vk));
	assert_not_null(vk);

	vk_init_options(vk);

	vk->simulation = simulation;

	simulation_event_table_t* table = simulation_get_event_table(vk->simulation);

	event_listener_data_t free_data =
	{
		.fn = (void*) vk_free,
		.data = vk
	};
	event_target_once(&table->free_target, free_data);

	vk_init_window(vk);

	return vk;
}
