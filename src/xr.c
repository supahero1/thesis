/* skip */
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
#include <thesis/debug.h>
#include <thesis/threads.h>
#include <thesis/alloc_ext.h>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <volk.h>

#define XR_USE_PLATFORM_WAYLAND
#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr_platform.h>

#define VULKAN_MAX_IMAGES 8

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef struct graphics_frame
{
	VkImage image;
	VkImageView view;
	VkFramebuffer frame_buffer;
	VkCommandBuffer command_buffer;
}
graphics_frame_t;

typedef enum graphics_barrier_semaphore
{
	GRAPHICS_BARRIER_SEMAPHORE_IMAGE_AVAILABLE,
	GRAPHICS_BARRIER_SEMAPHORE_RENDER_FINISHED,
	MACRO_ENUM_BITS(GRAPHICS_BARRIER_SEMAPHORE)
}
graphics_barrier_semaphore_t;

typedef enum graphics_barrier_fence
{
	GRAPHICS_BARRIER_FENCE_IN_FLIGHT,
	MACRO_ENUM_BITS(GRAPHICS_BARRIER_FENCE)
}
graphics_barrier_fence_t;

typedef struct graphics_barrier
{
	VkSemaphore semaphores[GRAPHICS_BARRIER_SEMAPHORE__COUNT];
	VkFence fences[GRAPHICS_BARRIER_FENCE__COUNT];
}
graphics_barrier_t;

typedef enum graphics_image_type
{
	GRAPHICS_IMAGE_TYPE_DEPTH_STENCIL,
	GRAPHICS_IMAGE_TYPE_MULTISAMPLED,
	GRAPHICS_IMAGE_TYPE_TEXTURE,
	MACRO_ENUM_BITS(GRAPHICS_IMAGE_TYPE)
}
graphics_image_type_t;

typedef struct graphics_image
{
	const char* path;

	uint32_t width;
	uint32_t height;
	uint32_t layers;

	VkFormat format;
	graphics_image_type_t type;

	VkImage image;
	VkImageView view;
	VkDeviceMemory memory;

	VkImageAspectFlags aspect;
	VkImageUsageFlags usage;
	VkSampleCountFlagBits samples;
}
graphics_image_t;

typedef struct graphics_vertex_input
{
	pair_t pos;
	pair_t tex_coord;
}
graphics_vertex_input_t;

typedef struct graphics_vertex_consts
{
	pair_t window_size;
}
graphics_vertex_consts_t;

struct graphics
{
	window_t window;

	graphics_event_table_t event_table;

#ifndef NDEBUG
	XrDebugUtilsMessengerEXT xr_debug_messenger;
#endif

	XrInstance xr_instance;
	XrSystemId xr_system;
	XrSession xr_session;

	PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;

#ifndef NDEBUG
	VkDebugUtilsMessengerEXT vk_debug_messenger;
#endif

	VkInstance vk_instance;

	VkSurfaceKHR vk_surface;
	VkSurfaceCapabilitiesKHR vk_surface_capabilities;

	VkPhysicalDevice vk_physical_device;
	uint32_t vk_queue_id;

	VkDevice vk_device;
	struct VolkDeviceTable vk_table;
	VkQueue vk_queue;

	VkSampleCountFlagBits vk_samples;
	VkPhysicalDeviceLimits vk_device_limits;
	VkPhysicalDeviceMemoryProperties vk_memory_properties;


	VkCommandPool vk_command_pool;
	VkCommandBuffer vk_command_buffer;
	VkFence vk_fence;

	VkExtent2D extent;
	uint32_t min_image_count;
	VkSurfaceTransformFlagBitsKHR transform;
	VkPresentModeKHR present_mode;

	VkSwapchainKHR swapchain;
	uint32_t image_count;
	graphics_frame_t frames[VULKAN_MAX_IMAGES];

	graphics_barrier_t barriers[VULKAN_MAX_IMAGES];
	graphics_barrier_t* barrier;

	VkViewport viewport;
	VkRect2D scissor;

	VkBuffer copy_buffer;
	VkDeviceMemory copy_buffer_memory;

	VkSampler sampler;

	graphics_image_t depth_buffer;
	graphics_image_t multisampling_buffer;

	graphics_image_t textures[1];

	VkBuffer vertex_input_buffer;
	VkDeviceMemory vertex_input_memory;

	VkBuffer instance_input_buffer;
	VkDeviceMemory instance_input_memory;

	VkBuffer draw_count_buffer;
	VkDeviceMemory draw_count_memory;

	graphics_vertex_consts_t consts;

	VkDescriptorSetLayout descriptors;
	VkRenderPass render_pass;
	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;
	VkDescriptorPool descriptor_pool;
	VkDescriptorSet descriptor_set;

	graphics_draw_data_t* draw_data_buffer;
	uint32_t draw_data_used;
	uint32_t draw_data_size;

	thread_t thread;
	uint32_t frame_number;
	_Atomic uint8_t should_run;

	sync_mtx_t resize_mtx;
	sync_cond_t resize_cond;
	bool resized;

	sync_mtx_t window_mtx;
};


private const char* graphics_xr_instance_extensions[] =
{
#ifndef NDEBUG
	XR_EXT_DEBUG_UTILS_EXTENSION_NAME,
#endif
	XR_KHR_VULKAN_ENABLE_EXTENSION_NAME,
	XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME,
	XR_EXT_HAND_TRACKING_EXTENSION_NAME,
};

private const char* graphics_xr_instance_layers[] =
{
#ifndef NDEBUG
	"XR_APILAYER_LUNARG_core_validation",
#endif
};

private const char* graphics_vk_instance_extensions[] =
{
#ifndef NDEBUG
	VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
#endif
};

private const char* graphics_vk_instance_layers[] =
{
#ifndef NDEBUG
	"VK_LAYER_KHRONOS_validation"
#endif
};

private const char* graphics_vk_device_extensions[] =
{
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	VK_KHR_MULTIVIEW_EXTENSION_NAME,
};

private const char* graphics_vk_device_layers[] =
{
#ifndef NDEBUG
	"VK_LAYER_KHRONOS_validation"
#endif
};


#ifndef NDEBUG

private XRAPI_ATTR XrBool32 XRAPI_CALL
graphics_xr_debug_callback(
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
graphics_vk_debug_callback(
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
graphics_xr_get_func(
	graphics_t graphics,
	const char* name
	)
{
	assert_not_null(graphics);
	assert_not_null(name);

	PFN_xrVoidFunction func;
	XrResult xr_result = xrGetInstanceProcAddr(graphics->xr_instance, name, &func);
	assert_eq(xr_result, XR_SUCCESS);
	assert_not_null(func);

	return func;
}


private void*
graphics_vk_get_func(
	graphics_t graphics,
	const char* name
	)
{
	assert_not_null(graphics);
	assert_not_null(name);

	void* func = graphics->vkGetInstanceProcAddr(graphics->vk_instance, name);
	assert_not_null(func);

	return func;
}


private void
graphics_free_str_array(
	const char** start,
	const char** end
	)
{
	assert_not_null(start);
	assert_not_null(end);

	while(start < end)
	{
		free((void*) *(start++));
	}
}


private const char**
graphics_xr_get_instance_extensions(
	graphics_t graphics,
	const char** extension
	)
{
	assert_not_null(graphics);
	assert_not_null(extension);

	uint32_t xr_instance_extension_count = 0;
	XrResult xr_result = xrEnumerateInstanceExtensionProperties(
		NULL, 0, &xr_instance_extension_count, NULL);

	XrExtensionProperties xr_instance_extensions[xr_instance_extension_count];

	XrExtensionProperties* xr_instance_extension = xr_instance_extensions;
	XrExtensionProperties* xr_instance_extension_end =
		xr_instance_extension + xr_instance_extension_count;

	while(xr_instance_extension < xr_instance_extension_end)
	{
		*(xr_instance_extension++) = (XrExtensionProperties){XR_TYPE_EXTENSION_PROPERTIES};
	}

	xr_result = xrEnumerateInstanceExtensionProperties(NULL,
		xr_instance_extension_count, &xr_instance_extension_count, xr_instance_extensions);
	assert_eq(xr_result, XR_SUCCESS);

	puts("XR instance extensions:");

	for(
		xr_instance_extension = xr_instance_extensions;
		xr_instance_extension < xr_instance_extension_end;
		xr_instance_extension++
		)
	{
		printf("- %s\n", xr_instance_extension->extensionName);
	}

	puts("");

	const char* const* graphics_xr_instance_extension = graphics_xr_instance_extensions;
	const char* const* graphics_xr_instance_extension_end =
		graphics_xr_instance_extension + MACRO_ARRAY_LEN(graphics_xr_instance_extensions);

	while(graphics_xr_instance_extension < graphics_xr_instance_extension_end)
	{
		bool found = false;
		const char* extension_name = *(graphics_xr_instance_extension++);

		xr_instance_extension = xr_instance_extensions;
		while(xr_instance_extension < xr_instance_extension_end)
		{
			if(strcmp(extension_name, xr_instance_extension->extensionName) == 0)
			{
				found = true;
				break;
			}

			xr_instance_extension++;
		}

		assert_true(found, fprintf(stderr, "XR instance extension %s not found\n", extension_name));
		*(extension++) = strdup(extension_name);
	}

	return extension;
}


private const char**
graphics_xr_get_instance_layers(
	graphics_t graphics,
	const char** layer
	)
{
	assert_not_null(graphics);
	assert_not_null(layer);

	uint32_t xr_instance_layer_count = 0;
	XrResult xr_result = xrEnumerateApiLayerProperties(0, &xr_instance_layer_count, NULL);
	assert_eq(xr_result, XR_SUCCESS);

	XrApiLayerProperties xr_instance_layers[xr_instance_layer_count];

	XrApiLayerProperties* xr_instance_layer = xr_instance_layers;
	XrApiLayerProperties* xr_instance_layer_end =
		xr_instance_layer + xr_instance_layer_count;

	while(xr_instance_layer < xr_instance_layer_end)
	{
		*(xr_instance_layer++) = (XrApiLayerProperties){XR_TYPE_API_LAYER_PROPERTIES};
	}

	xr_result = xrEnumerateApiLayerProperties(
		xr_instance_layer_count, &xr_instance_layer_count, xr_instance_layers);
	assert_eq(xr_result, XR_SUCCESS);

	puts("XR instance layers:");

	for(
		xr_instance_layer = xr_instance_layers;
		xr_instance_layer < xr_instance_layer_end;
		xr_instance_layer++
		)
	{
		printf("- %s\n", xr_instance_layer->layerName);
	}

	puts("");

	const char* const* graphics_xr_instance_layer = graphics_xr_instance_layers;
	const char* const* graphics_xr_instance_layer_end =
		graphics_xr_instance_layer + MACRO_ARRAY_LEN(graphics_xr_instance_layers);

	while(graphics_xr_instance_layer < graphics_xr_instance_layer_end)
	{
		bool found = false;
		const char* layer_name = *(graphics_xr_instance_layer++);

		xr_instance_layer = xr_instance_layers;
		while(xr_instance_layer < xr_instance_layer_end)
		{
			if(strcmp(layer_name, xr_instance_layer->layerName) == 0)
			{
				found = true;
				break;
			}

			xr_instance_layer++;
		}

		assert_true(found, fprintf(stderr, "XR instance layer %s not found\n", layer_name));
		*(layer++) = strdup(layer_name);
	}

	return layer;
}


private void
graphics_init_xr_instance(
	graphics_t graphics
	)
{
	assert_not_null(graphics);

	const char* xr_instance_extensions[64];
	const char** xr_instance_extension =
		graphics_xr_get_instance_extensions(graphics, xr_instance_extensions);
	assert_lt(xr_instance_extension,
		xr_instance_extensions + MACRO_ARRAY_LEN(xr_instance_extensions));

	const char* xr_instance_layers[64];
	const char** xr_instance_layer =
		graphics_xr_get_instance_layers(graphics, xr_instance_layers);
	assert_lt(xr_instance_layer,
		xr_instance_layers + MACRO_ARRAY_LEN(xr_instance_layers));

	XrInstanceCreateInfo xr_instance_info =
	{
		.type = XR_TYPE_INSTANCE_CREATE_INFO,
		.next = NULL,
		.createFlags = 0,
		.applicationInfo =
		{
			.apiVersion = XR_CURRENT_API_VERSION
		},
		.enabledExtensionCount = xr_instance_extension - xr_instance_extensions,
		.enabledExtensionNames = xr_instance_extensions,
		.enabledApiLayerCount = xr_instance_layer - xr_instance_layers,
		.enabledApiLayerNames = xr_instance_layers,
	};

	strcpy(xr_instance_info.applicationInfo.applicationName, "Thesis");

#ifndef NDEBUG
	XrDebugUtilsMessengerCreateInfoEXT xr_debug_info =
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
		.userCallback = graphics_xr_debug_callback,
		.userData = NULL
	};

	xr_instance_info.next = &xr_debug_info;
#endif

	XrResult xr_result = xrCreateInstance(&xr_instance_info, &graphics->xr_instance);
	assert_eq(xr_result, XR_SUCCESS);

	graphics_free_str_array(xr_instance_extensions, xr_instance_extension);
	graphics_free_str_array(xr_instance_layers, xr_instance_layer);

#ifndef NDEBUG
	PFN_xrCreateDebugUtilsMessengerEXT xrCreateDebugUtilsMessengerEXT =
		graphics_xr_get_func(graphics, "xrCreateDebugUtilsMessengerEXT");

	xr_result = xrCreateDebugUtilsMessengerEXT(
		graphics->xr_instance, &xr_debug_info, &graphics->xr_debug_messenger);
	assert_eq(xr_result, XR_SUCCESS);
#endif

	XrInstanceProperties xr_instance_properties = {XR_TYPE_INSTANCE_PROPERTIES};
	xr_result = xrGetInstanceProperties(graphics->xr_instance, &xr_instance_properties);
	assert_eq(xr_result, XR_SUCCESS);

	printf(
		"XR runtime: '%s' ver. %u.%u.%u\n",
		xr_instance_properties.runtimeName,
		XR_VERSION_MAJOR(xr_instance_properties.runtimeVersion),
		XR_VERSION_MINOR(xr_instance_properties.runtimeVersion),
		XR_VERSION_PATCH(xr_instance_properties.runtimeVersion)
		);

	XrSystemGetInfo xr_system_info =
	{
		.type = XR_TYPE_SYSTEM_GET_INFO,
		.next = NULL,
		.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY
	};
	xr_result = xrGetSystem(graphics->xr_instance, &xr_system_info, &graphics->xr_system);
	assert_eq(xr_result, XR_SUCCESS);

	XrSystemHandTrackingPropertiesEXT xr_hand_tracking_properties =
	{
		.type = XR_TYPE_SYSTEM_HAND_TRACKING_PROPERTIES_EXT,
		.next = NULL,
		.supportsHandTracking = false
	};

	XrSystemProperties xr_system_properties = {
		.type = XR_TYPE_SYSTEM_PROPERTIES,
		.next = &xr_hand_tracking_properties
	};

	xr_result = xrGetSystemProperties(graphics->xr_instance,
		graphics->xr_system, &xr_system_properties);
	assert_eq(xr_result, XR_SUCCESS);

	assert_true(xr_hand_tracking_properties.supportsHandTracking);

	printf("XR system: %s\n", xr_system_properties.systemName);
	printf("XR system vendor: %u\n", xr_system_properties.vendorId);
	printf(
		"XR system properties:\n"
		"\tmaxSwapchainImageHeight: %u\n"
		"\tmaxSwapchainImageWidth: %u\n"
		"\tmaxLayerCount: %u\n"
		"\torientationTracking: %d\n"
		"\tpositionTracking: %d\n",
		xr_system_properties.graphicsProperties.maxSwapchainImageHeight,
		xr_system_properties.graphicsProperties.maxSwapchainImageWidth,
		xr_system_properties.graphicsProperties.maxLayerCount,
		xr_system_properties.trackingProperties.orientationTracking,
		xr_system_properties.trackingProperties.positionTracking
		);
}


private void
graphics_free_xr_instance(
	graphics_t graphics
	)
{
	assert_not_null(graphics);

#ifndef NDEBUG
	PFN_xrDestroyDebugUtilsMessengerEXT xrDestroyDebugUtilsMessengerEXT =
		graphics_xr_get_func(graphics, "xrDestroyDebugUtilsMessengerEXT");

	xrDestroyDebugUtilsMessengerEXT(graphics->xr_debug_messenger);
#endif

	xrDestroyInstance(graphics->xr_instance);
}


private const char**
graphics_vk_get_instance_extensions(
	graphics_t graphics,
	const char** extension
	)
{
	assert_not_null(graphics);
	assert_not_null(extension);

	uint32_t vk_instance_extension_count = 0;
	VkResult vk_result = vkEnumerateInstanceExtensionProperties(
		NULL, &vk_instance_extension_count, NULL);
	assert_eq(vk_result, VK_SUCCESS);

	VkExtensionProperties vk_instance_extensions[vk_instance_extension_count];

	VkExtensionProperties* vk_instance_extension = vk_instance_extensions;
	VkExtensionProperties* vk_instance_extension_end =
		vk_instance_extension + vk_instance_extension_count;

	while(vk_instance_extension < vk_instance_extension_end)
	{
		*(vk_instance_extension++) = (VkExtensionProperties){0};
	}

	vk_result = vkEnumerateInstanceExtensionProperties(
		NULL, &vk_instance_extension_count, vk_instance_extensions);
	assert_eq(vk_result, VK_SUCCESS);

	puts("VK instance extensions:");

	for(
		vk_instance_extension = vk_instance_extensions;
		vk_instance_extension < vk_instance_extension_end;
		vk_instance_extension++
		)
	{
		printf("- %s\n", vk_instance_extension->extensionName);
	}

	puts("");

	const char* const* graphics_vk_instance_extension = graphics_vk_instance_extensions;
	const char* const* graphics_vk_instance_extension_end =
		graphics_vk_instance_extension + MACRO_ARRAY_LEN(graphics_vk_instance_extensions);

	while(graphics_vk_instance_extension < graphics_vk_instance_extension_end)
	{
		bool found = false;
		const char* extension_name = *(graphics_vk_instance_extension++);

		vk_instance_extension = vk_instance_extensions;
		while(vk_instance_extension < vk_instance_extension_end)
		{
			if(strcmp(extension_name, vk_instance_extension->extensionName) == 0)
			{
				found = true;
				break;
			}

			vk_instance_extension++;
		}

		assert_true(found, fprintf(stderr, "VK instance extension %s not found\n", extension_name));
		*(extension++) = strdup(extension_name);
	}

	uint32_t sdl_instance_extension_count = 0;
	const char* const* sdl_instance_extensions =
		window_get_vulkan_extensions(&sdl_instance_extension_count);
	assert_ptr(sdl_instance_extensions, sdl_instance_extension_count);

	const char* const* sdl_instance_extension = sdl_instance_extensions;
	const char* const* sdl_instance_extension_end =
		sdl_instance_extension + sdl_instance_extension_count;

	while(sdl_instance_extension < sdl_instance_extension_end)
	{
		bool found = false;
		const char* extension_name = *(sdl_instance_extension++);

		vk_instance_extension = vk_instance_extensions;
		while(vk_instance_extension < vk_instance_extension_end)
		{
			if(strcmp(extension_name, vk_instance_extension->extensionName) == 0)
			{
				found = true;
				break;
			}

			vk_instance_extension++;
		}

		assert_true(found, fprintf(stderr, "SDL VK instance extension %s not found\n", extension_name));
		*(extension++) = strdup(extension_name);
	}

	return extension;
}


private const char**
graphics_vk_get_instance_layers(
	graphics_t graphics,
	const char** layer
	)
{
	assert_not_null(graphics);
	assert_not_null(layer);

	uint32_t vk_instance_layer_count = 0;
	VkResult vk_result = vkEnumerateInstanceLayerProperties(&vk_instance_layer_count, NULL);
	assert_eq(vk_result, VK_SUCCESS);

	VkLayerProperties vk_instance_layers[vk_instance_layer_count];

	VkLayerProperties* vk_instance_layer = vk_instance_layers;
	VkLayerProperties* vk_instance_layer_end =
		vk_instance_layer + vk_instance_layer_count;

	while(vk_instance_layer < vk_instance_layer_end)
	{
		*(vk_instance_layer++) = (VkLayerProperties){0};
	}

	vk_result = vkEnumerateInstanceLayerProperties(&vk_instance_layer_count, vk_instance_layers);
	assert_eq(vk_result, VK_SUCCESS);

	puts("VK instance layers:");

	for(
		vk_instance_layer = vk_instance_layers;
		vk_instance_layer < vk_instance_layer_end;
		vk_instance_layer++
		)
	{
		printf("- %s\n", vk_instance_layer->layerName);
	}

	puts("");

	const char* const* graphics_vk_instance_layer = graphics_vk_instance_layers;
	const char* const* graphics_vk_instance_layer_end =
		graphics_vk_instance_layer + MACRO_ARRAY_LEN(graphics_vk_instance_layers);

	while(graphics_vk_instance_layer < graphics_vk_instance_layer_end)
	{
		bool found = false;
		const char* layer_name = *(graphics_vk_instance_layer++);

		vk_instance_layer = vk_instance_layers;
		while(vk_instance_layer < vk_instance_layer_end)
		{
			if(strcmp(layer_name, vk_instance_layer->layerName) == 0)
			{
				found = true;
				break;
			}

			vk_instance_layer++;
		}

		assert_true(found, fprintf(stderr, "VK instance layer %s not found\n", layer_name));
		*(layer++) = strdup(layer_name);
	}

	return layer;
}


private void
graphics_init_vk_instance(
	graphics_t graphics
	)
{
	assert_not_null(graphics);

	graphics->vkGetInstanceProcAddr = window_get_vulkan_proc_addr_fn();
	assert_not_null(graphics->vkGetInstanceProcAddr);

	volkInitializeCustom(graphics->vkGetInstanceProcAddr);

	XrGraphicsRequirementsVulkanKHR xr_graphics_requirements =
	{
		.type = XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR,
		.next = NULL,
		.minApiVersionSupported = XR_MAKE_VERSION(1, 4, 309),
		.maxApiVersionSupported = XR_MAKE_VERSION(1, 4, 309)
	};

	PFN_xrGetVulkanGraphicsRequirementsKHR xrGetVulkanGraphicsRequirementsKHR =
		graphics_xr_get_func(graphics, "xrGetVulkanGraphicsRequirementsKHR");

	XrResult xr_result = xrGetVulkanGraphicsRequirementsKHR(
		graphics->xr_instance, graphics->xr_system, &xr_graphics_requirements);
	assert_eq(xr_result, XR_SUCCESS);

	const char* vk_instance_extensions[64];
	const char** vk_instance_extension =
		graphics_vk_get_instance_extensions(graphics, vk_instance_extensions);
	assert_lt(vk_instance_extension,
		vk_instance_extensions + MACRO_ARRAY_LEN(vk_instance_extensions));

	const char* vk_instance_layers[64];
	const char** vk_instance_layer =
		graphics_vk_get_instance_layers(graphics, vk_instance_layers);
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
		.pfnUserCallback = graphics_vk_debug_callback,
		.pUserData = NULL
	};

	vk_instance_info.pNext = &vk_debug_info;
#endif

	XrVulkanInstanceCreateInfoKHR xr_vk_instance_info =
	{
		.type = XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR,
		.next = NULL,
		.systemId = graphics->xr_system,
		.createFlags = 0,
		.pfnGetInstanceProcAddr = vkGetInstanceProcAddr,
		.vulkanCreateInfo = &vk_instance_info,
		.vulkanAllocator = NULL
	};

	PFN_xrCreateVulkanInstanceKHR xrCreateVulkanInstanceKHR =
		graphics_xr_get_func(graphics, "xrCreateVulkanInstanceKHR");

	VkResult vk_result;
	xr_result = xrCreateVulkanInstanceKHR(graphics->xr_instance,
		&xr_vk_instance_info, &graphics->vk_instance, &vk_result);
	assert_eq(xr_result, XR_SUCCESS);
	assert_eq(vk_result, VK_SUCCESS);

	graphics_free_str_array(vk_instance_extensions, vk_instance_extension);
	graphics_free_str_array(vk_instance_layers, vk_instance_layer);

#ifndef NDEBUG
	PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT =
		graphics_vk_get_func(graphics, "vkCreateDebugUtilsMessengerEXT");

	vk_result = vkCreateDebugUtilsMessengerEXT(graphics->vk_instance,
		&vk_debug_info, NULL, &graphics->vk_debug_messenger);
	assert_eq(vk_result, VK_SUCCESS);
#endif

	volkLoadInstanceOnly(graphics->vk_instance);
}


private void
graphics_free_vk_instance(
	graphics_t graphics
	)
{
	assert_not_null(graphics);

#ifndef NDEBUG
	/* Volk loaded the function already */
	vkDestroyDebugUtilsMessengerEXT(
		graphics->vk_instance, graphics->vk_debug_messenger, NULL);
#endif

	vkDestroyInstance(graphics->vk_instance, NULL);

	volkFinalize();
}


private void
graphics_init_vk_physical_device(
	graphics_t graphics
	)
{
	assert_not_null(graphics);

	XrVulkanGraphicsDeviceGetInfoKHR xr_vk_device_info =
	{
		.type = XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR,
		.next = NULL,
		.systemId = graphics->xr_system,
		.vulkanInstance = graphics->vk_instance
	};

	PFN_xrGetVulkanGraphicsDevice2KHR xrGetVulkanGraphicsDevice2KHR =
		graphics_xr_get_func(graphics, "xrGetVulkanGraphicsDevice2KHR");

	XrResult xr_result = xrGetVulkanGraphicsDevice2KHR(
		graphics->xr_instance, &xr_vk_device_info, &graphics->vk_physical_device);
	assert_eq(xr_result, XR_SUCCESS);

	uint32_t vk_queue_family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(
		graphics->vk_physical_device, &vk_queue_family_count, NULL);
	assert_gt(vk_queue_family_count, 0);

	VkQueueFamilyProperties vk_queue_family_properties[vk_queue_family_count];

	vkGetPhysicalDeviceQueueFamilyProperties(
		graphics->vk_physical_device, &vk_queue_family_count, vk_queue_family_properties);
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
	graphics->vk_queue_id = vk_queue_family_property - vk_queue_family_properties;
}


private void
graphics_free_vk_physical_device(
	graphics_t graphics
	)
{
	assert_not_null(graphics);

	/* Physical devices are not freed */
}


private const char**
graphics_vk_get_device_extensions(
	graphics_t graphics,
	const char** extension
	)
{
	assert_not_null(graphics);
	assert_not_null(extension);

	uint32_t vk_device_extension_count = 0;
	vkEnumerateDeviceExtensionProperties(
		graphics->vk_physical_device, NULL, &vk_device_extension_count, NULL);

	VkExtensionProperties vk_device_extensions[vk_device_extension_count];

	VkExtensionProperties* vk_device_extension = vk_device_extensions;
	VkExtensionProperties* vk_device_extension_end =
		vk_device_extension + vk_device_extension_count;

	vkEnumerateDeviceExtensionProperties(graphics->vk_physical_device,
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

	const char* const* graphics_vk_device_extension = graphics_vk_device_extensions;
	const char* const* graphics_vk_device_extension_end =
		graphics_vk_device_extension + MACRO_ARRAY_LEN(graphics_vk_device_extensions);

	while(graphics_vk_device_extension < graphics_vk_device_extension_end)
	{
		bool found = false;
		const char* extension_name = *(graphics_vk_device_extension++);

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
		graphics_xr_get_func(graphics, "xrGetVulkanDeviceExtensionsKHR");

	XrResult xr_result = xrGetVulkanDeviceExtensionsKHR(graphics->xr_instance,
		graphics->xr_system, 0, &xr_vk_device_extension_count, NULL);
	assert_eq(xr_result, XR_SUCCESS);

	char xr_vk_device_extensions[xr_vk_device_extension_count + 1];
	xr_vk_device_extensions[xr_vk_device_extension_count] = '\0';

	xr_result = xrGetVulkanDeviceExtensionsKHR(graphics->xr_instance, graphics->xr_system,
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
graphics_vk_get_device_layers(
	graphics_t graphics,
	const char** layer
	)
{
	assert_not_null(graphics);
	assert_not_null(layer);

	uint32_t vk_device_layer_count = 0;
	vkEnumerateDeviceLayerProperties(
		graphics->vk_physical_device, &vk_device_layer_count, NULL);

	VkLayerProperties vk_device_layers[vk_device_layer_count];

	VkLayerProperties* vk_device_layer = vk_device_layers;
	VkLayerProperties* vk_device_layer_end = vk_device_layer + vk_device_layer_count;

	vkEnumerateDeviceLayerProperties(graphics->vk_physical_device,
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

	const char* const* graphics_vk_device_layer = graphics_vk_device_layers;
	const char* const* graphics_vk_device_layer_end =
		graphics_vk_device_layer + MACRO_ARRAY_LEN(graphics_vk_device_layers);

	while(graphics_vk_device_layer < graphics_vk_device_layer_end)
	{
		bool found = false;
		const char* layer_name = *(graphics_vk_device_layer++);

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
graphics_init_vk_logical_device(
	graphics_t graphics
	)
{
	assert_not_null(graphics);

	const char* vk_device_extensions[64];
	const char** vk_device_extension =
		graphics_vk_get_device_extensions(graphics, vk_device_extensions);
	assert_lt(vk_device_extension,
		vk_device_extensions + MACRO_ARRAY_LEN(vk_device_extensions));

	const char* vk_device_layers[64];
	const char** vk_device_layer =
		graphics_vk_get_device_layers(graphics, vk_device_layers);
	assert_lt(vk_device_layer,
		vk_device_layers + MACRO_ARRAY_LEN(vk_device_layers));

	float vk_queue_priority = 1.0f;

	VkDeviceQueueCreateInfo vk_queue_info =
	{
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.queueFamilyIndex = graphics->vk_queue_id,
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
		.systemId = graphics->xr_system,
		.createFlags = 0,
		.pfnGetInstanceProcAddr = graphics->vkGetInstanceProcAddr,
		.vulkanPhysicalDevice = graphics->vk_physical_device,
		.vulkanCreateInfo = &vk_device_info,
		.vulkanAllocator = NULL
	};

	PFN_xrCreateVulkanDeviceKHR xrCreateVulkanDeviceKHR =
		graphics_xr_get_func(graphics, "xrCreateVulkanDeviceKHR");

	VkResult vk_result;
	XrResult xr_result = xrCreateVulkanDeviceKHR(
		graphics->xr_instance, &xr_vk_device_info, &graphics->vk_device, &vk_result);
	assert_eq(xr_result, XR_SUCCESS);
	assert_eq(vk_result, VK_SUCCESS);

	graphics_free_str_array(vk_device_extensions, vk_device_extension);
	graphics_free_str_array(vk_device_layers, vk_device_layer);

	volkLoadDeviceTable(&graphics->vk_table, graphics->vk_device);

	graphics->vk_table.vkGetDeviceQueue(graphics->vk_device,
		graphics->vk_queue_id, 0, &graphics->vk_queue);

	// vkGetPhysicalDeviceMemoryProperties(
	// 	graphics->vk_physical_device, &graphics->vk_memory_properties);
}


private void
graphics_free_vk_logical_device(
	graphics_t graphics
	)
{
	assert_not_null(graphics);

	graphics->vk_table.vkDestroyDevice(graphics->vk_device, NULL);
}


private void
graphics_init_xr_session(
	graphics_t graphics
	)
{
	assert_not_null(graphics);

	XrGraphicsBindingVulkanKHR xr_vk_binding =
	{
		.type = XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR,
		.next = NULL,
		.instance = graphics->vk_instance,
		.physicalDevice = graphics->vk_physical_device,
		.device = graphics->vk_device,
		.queueFamilyIndex = graphics->vk_queue_id,
		.queueIndex = 0
	};

	XrSessionCreateInfo xr_session_info =
	{
		.type = XR_TYPE_SESSION_CREATE_INFO,
		.next = &xr_vk_binding,
		.createFlags = 0,
		.systemId = graphics->xr_system
	};

	XrResult xr_result = xrCreateSession(
		graphics->xr_instance, &xr_session_info, &graphics->xr_session);
	assert_eq(xr_result, XR_SUCCESS);
}


private void
graphics_free_xr_session(
	graphics_t graphics
	)
{
	assert_not_null(graphics);

	XrResult xr_result = xrDestroySession(graphics->xr_session);
	assert_eq(xr_result, XR_SUCCESS);
}


private void
graphics_init_xr(
	graphics_t graphics,
	window_init_event_data_t* event_data
	)
{
	assert_not_null(graphics);

	graphics_init_xr_instance(graphics);
	graphics_init_vk_instance(graphics);
	graphics_init_vk_physical_device(graphics);
	graphics_init_vk_logical_device(graphics);
	graphics_init_xr_session(graphics);
}


private void
graphics_free_xr(
	graphics_t graphics
	)
{
	assert_not_null(graphics);

	graphics_free_xr_session(graphics);
	graphics_free_vk_logical_device(graphics);
	graphics_free_vk_physical_device(graphics);
	graphics_free_vk_instance(graphics);
	graphics_free_xr_instance(graphics);
}


private void
graphics_free(
	graphics_t graphics,
	window_free_event_data_t* event_data
	)
{
	assert_not_null(graphics);

	graphics_free_xr(graphics);

	event_target_free(&graphics->event_table.draw_target);

	alloc_free(graphics, sizeof(*graphics));
}


graphics_t
graphics_init(
	window_t window
	)
{
	assert_not_null(window);

	graphics_t graphics = alloc_malloc(sizeof(*graphics));
	assert_ptr(graphics, sizeof(*graphics));

	graphics->window = window;
	window_event_table_t* table = window_get_event_table(window);

	event_listener_data_t init_data =
	{
		.fn = (event_fn_t) graphics_init_xr,
		.data = graphics
	};
	event_target_once(&table->init_target, init_data);

	event_listener_data_t free_data =
	{
		.fn = (event_fn_t) graphics_free,
		.data = graphics
	};
	(void) event_target_once(&table->free_target, free_data);

	event_target_init(&graphics->event_table.draw_target);

	return graphics;
}


