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

#include <thesis/debug.h>
#include <thesis/threads.h>
#include <thesis/alloc_ext.h>
#include <thesis/graphics.h>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <volk.h>

#define XR_USE_PLATFORM_WAYLAND
#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr_platform.h>

#define VULKAN_MAX_IMAGES 8

#include <stdio.h>
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
	XrDebugUtilsMessengerEXT xr_messenger;
#endif

	XrInstance xr_instance;
	XrSystemId xr_system;
	XrSession xr_session;

	struct VolkDeviceTable table;

	VkDebugUtilsMessengerEXT messenger;

	VkInstance instance;
	VkSurfaceKHR surface;
	VkSurfaceCapabilitiesKHR surface_capabilities;

	uint32_t queue_id;
	VkSampleCountFlagBits samples;
	VkPhysicalDeviceLimits device_limits;

	VkPhysicalDevice physical_device;
	VkDevice device;
	VkQueue queue;
	VkPhysicalDeviceMemoryProperties memory_properties;

	VkCommandPool command_pool;
	VkCommandBuffer command_buffer;
	VkFence fence;

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
	return XR_FALSE;
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
	return VK_FALSE;
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


private const char**
graphics_xr_get_instance_extensions(
	graphics_t graphics,
	const char** extension
	)
{
	assert_not_null(graphics);
	assert_not_null(extension);

	uint32_t xr_instance_properties_count = 0;
	XrResult xr_result = xrEnumerateInstanceExtensionProperties(
		NULL, 0, &xr_instance_properties_count, NULL);

	XrExtensionProperties xr_instance_properties[xr_instance_properties_count];

	XrExtensionProperties* xr_instance_property = xr_instance_properties;
	XrExtensionProperties* xr_instance_property_end =
		xr_instance_property + xr_instance_properties_count;

	while(xr_instance_property < xr_instance_property_end)
	{
		*(xr_instance_property++) = (XrExtensionProperties){XR_TYPE_EXTENSION_PROPERTIES};
	}

	xr_result = xrEnumerateInstanceExtensionProperties(NULL,
		xr_instance_properties_count, &xr_instance_properties_count, xr_instance_properties);
	assert_eq(xr_result, XR_SUCCESS);

	const char* const* graphics_xr_instance_extension = graphics_xr_instance_extensions;
	const char* const* graphics_xr_instance_extension_end =
		graphics_xr_instance_extension + MACRO_ARRAY_LEN(graphics_xr_instance_extensions);

	while(graphics_xr_instance_extension < graphics_xr_instance_extension_end)
	{
		bool found = false;
		const char* extension_name = *(graphics_xr_instance_extension++);

		xr_instance_property = xr_instance_properties;
		while(xr_instance_property < xr_instance_property_end)
		{
			if(strcmp(extension_name, xr_instance_property->extensionName) == 0)
			{
				found = true;
				break;
			}

			xr_instance_property++;
		}

		assert_true(found, fprintf(stderr, "XR extension %s not found\n", extension_name));
		*(extension++) = extension_name;
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

	uint32_t xr_instance_layers_count = 0;
	XrResult xr_result = xrEnumerateApiLayerProperties(0, &xr_instance_layers_count, NULL);
	assert_eq(xr_result, XR_SUCCESS);

	XrApiLayerProperties xr_instance_layers_properties[xr_instance_layers_count];

	XrApiLayerProperties* xr_instance_layer_property = xr_instance_layers_properties;
	XrApiLayerProperties* xr_instance_layer_property_end =
		xr_instance_layer_property + xr_instance_layers_count;

	while(xr_instance_layer_property < xr_instance_layer_property_end)
	{
		*(xr_instance_layer_property++) = (XrApiLayerProperties){XR_TYPE_API_LAYER_PROPERTIES};
	}

	xr_result = xrEnumerateApiLayerProperties(
		xr_instance_layers_count, &xr_instance_layers_count, xr_instance_layers_properties);
	assert_eq(xr_result, XR_SUCCESS);

	const char* const* graphics_xr_instance_layer = graphics_xr_instance_layers;
	const char* const* graphics_xr_instance_layer_end =
		graphics_xr_instance_layer + MACRO_ARRAY_LEN(graphics_xr_instance_layers);

	while(graphics_xr_instance_layer < graphics_xr_instance_layer_end)
	{
		bool found = false;
		const char* layer_name = *(graphics_xr_instance_layer++);

		xr_instance_layer_property = xr_instance_layers_properties;
		while(xr_instance_layer_property < xr_instance_layer_property_end)
		{
			if(strcmp(layer_name, xr_instance_layer_property->layerName) == 0)
			{
				found = true;
				break;
			}

			xr_instance_layer_property++;
		}

		assert_true(found, fprintf(stderr, "XR layer %s not found\n", layer_name));
		*(layer++) = layer_name;
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

#ifndef NDEBUG
	PFN_xrCreateDebugUtilsMessengerEXT xrCreateDebugUtilsMessengerEXT =
		graphics_xr_get_func(graphics, "xrCreateDebugUtilsMessengerEXT");

	xr_result = xrCreateDebugUtilsMessengerEXT(
		graphics->xr_instance, &xr_debug_info, &graphics->xr_messenger);
	assert_eq(xr_result, XR_SUCCESS);
#endif

	XrInstanceProperties xr_instance_properties = {XR_TYPE_INSTANCE_PROPERTIES};
	xr_result = xrGetInstanceProperties(graphics->xr_instance, &xr_instance_properties);
	assert_eq(xr_result, XR_SUCCESS);

	printf("XR instance: %s\n", xr_instance_properties.runtimeName);
	printf("XR runtime version: %u.%u.%u\n",
		XR_VERSION_MAJOR(xr_instance_properties.runtimeVersion),
		XR_VERSION_MINOR(xr_instance_properties.runtimeVersion),
		XR_VERSION_PATCH(xr_instance_properties.runtimeVersion));

	XrSystemGetInfo xr_system_info =
	{
		.type = XR_TYPE_SYSTEM_GET_INFO,
		.next = NULL,
		.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY
	};
	xr_result = xrGetSystem(graphics->xr_instance, &xr_system_info, &graphics->xr_system);
	assert_eq(xr_result, XR_SUCCESS);

	XrSystemProperties xr_system_properties = {XR_TYPE_SYSTEM_PROPERTIES};
	xr_result = xrGetSystemProperties(graphics->xr_instance,
		graphics->xr_system, &xr_system_properties);
	assert_eq(xr_result, XR_SUCCESS);

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

	xrDestroyDebugUtilsMessengerEXT(graphics->xr_messenger);
#endif

	xrDestroyInstance(graphics->xr_instance);
}


private void
graphics_init_xr(
	graphics_t graphics
	)
{
	assert_not_null(graphics);

	graphics_init_xr_instance(graphics);
}


private void
graphics_free_xr(
	graphics_t graphics
	)
{
	assert_not_null(graphics);

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

	event_listener_data_t free_data =
	{
		.fn = (event_fn_t) graphics_free,
		.data = graphics
	};
	(void) event_target_once(&table->free_target, free_data);

	event_target_init(&graphics->event_table.draw_target);

	graphics_init_xr(graphics);

	return graphics;
}


