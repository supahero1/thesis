/*
 *   Copyright 2025 Franciszek Balcerak
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by vklicable law or agreed to in writing, software
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
#include <thesis/threads.h>
#include <thesis/alloc_ext.h>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <volk.h>

#include <signal.h>
#include <string.h>


struct vk
{
	simulation_t simulation;

	event_listener_t* window_close_once_listener;
	event_listener_t* window_resize_listener;
	event_listener_t* window_key_down_listener;

	window_manager_t window_manager;
	window_t window;
	thread_t window_thread;

	bool window_resize_bool;
	sync_mtx_t window_resize_mtx;
	sync_cond_t window_resize_cond;



	PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;

#ifndef NDEBUG
	VkDebugUtilsMessengerEXT vk_debug_messenger;
#endif

	VkInstance vk_instance;

	VkSurfaceKHR vk_surface;
	VkSurfaceCapabilitiesKHR vk_surface_capabilities;

	uint32_t vk_queue_id;
	VkSampleCountFlagBits vk_samples;
	VkPhysicalDeviceLimits vk_limits;

	VkPhysicalDevice vk_physical_device;
	VkDevice vk_device;

	struct VolkDeviceTable vk_table;

	VkQueue vk_queue;
	VkPhysicalDeviceMemoryProperties vk_memory_properties;

	VkExtent2D vk_extent;
	uint32_t vk_min_image_count;
	VkSurfaceTransformFlagBitsKHR vk_transform;
	VkPresentModeKHR vk_present_mode;

	VkCommandPool vk_command_pool;
	VkCommandBuffer vk_command_buffer;
	VkFence vk_fence;
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


private void*
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

	puts("\n");


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

	puts("\n");

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

	puts("\n");

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
	VkSampleCountFlagBits samples;
	VkPhysicalDeviceLimits limits;
}
vk_device_score;


private bool
vk_get_device_features(
	vk_t vk,
	VkPhysicalDevice device,
	vk_device_score* device_score
	)
{
	VkPhysicalDeviceFeatures features;
	vkGetPhysicalDeviceFeatures(device, &features);

	if(!features.sampleRateShading)
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
	vk_device_score* device_score
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
	vk_device_score* device_score
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
	vk_device_score* device_score
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

	puts("\n");

	return true;
}


private bool
vk_get_device_swapchain(
	vk_t vk,
	VkPhysicalDevice device,
	vk_device_score* device_score
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
		if(format->format == VK_FORMAT_B8G8R8A8_SRGB &&
			format->colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
		{
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
	vk_device_score* device_score
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

	if(properties.limits.maxImageDimension2D < 1024)
	{
		hard_assert_log("%u\n", properties.limits.maxImageDimension2D);
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


private vk_device_score
vk_get_device_score(
	vk_t vk,
	VkPhysicalDevice device
	)
{
	assert_not_null(vk);
	assert_not_null(device);

	vk_device_score device_score = {0};

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
			while(!vk->window_resize_bool)
			{
				sync_cond_wait(&vk->window_resize_cond, &vk->window_resize_mtx);
			}

			vk->window_resize_bool = false;
		sync_mtx_unlock(&vk->window_resize_mtx);
	}

	width = MACRO_CLAMP(
		width,
		vk->vk_surface_capabilities.maxImageExtent.width,
		vk->vk_surface_capabilities.minImageExtent.width
		);

	height = MACRO_CLAMP(
		height,
		vk->vk_surface_capabilities.maxImageExtent.height,
		vk->vk_surface_capabilities.minImageExtent.height
		);

	vk->vk_extent =
	(VkExtent2D)
	{
		.width = width,
		.height = height
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
	vk_device_score best_device_score = {0};

	do
	{
		vk_device_score this_device_score = vk_get_device_score(vk, *vk_physical_device);

		if(this_device_score.score > best_device_score.score)
		{
			best_device = *vk_physical_device;
			best_device_score = this_device_score;
		}
	}
	while(++vk_physical_device != vk_physical_device_end);

	hard_assert_not_null(best_device);

	vk->vk_queue_id = best_device_score.queue_id;
	vk->vk_samples = best_device_score.samples;
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
		.samplerAnisotropy = VK_TRUE
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
	vk->vk_min_image_count = MACRO_CLAMP(
		2,
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


private void
vk_init_vk(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_instance(vk);
	vk_init_surface(vk);
	vk_init_device(vk);
}


private void
vk_free_vk(
	vk_t vk
	)
{
	assert_not_null(vk);

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

	event_target_del(&table->key_down_target, vk->window_key_down_listener);
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

	vk_init_vk(vk);

	// window_toggle_fullscreen(vk->window);
	window_show(vk->window);
}


private void
vk_window_resize_fn(
	vk_t vk,
	window_resize_event_data_t* event_data
	)
{
	assert_not_null(vk);

	if(!vk->window_resize_bool)
	{
		vk->window_resize_bool = true;
		sync_cond_wake(&vk->window_resize_cond);
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
		event_data->key == WINDOW_KEY_ESCAPE ||
		event_data->key == WINDOW_KEY_F11
		)
	{
		window_close(vk->window);
	}
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
	window_manager_add(vk->window_manager, vk->window, "Thesis", NULL);

	vk->window_resize_bool = false;
	sync_mtx_init(&vk->window_resize_mtx);
	sync_cond_init(&vk->window_resize_cond);

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

	event_listener_data_t key_down_data =
	{
		.fn = (void*) vk_window_key_down_fn,
		.data = vk
	};
	vk->window_key_down_listener = event_target_add(&table->key_down_target, key_down_data);


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

	vk_t vk = alloc_malloc(sizeof(*vk));
	assert_not_null(vk);

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
