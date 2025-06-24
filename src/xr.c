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
#include <thesis/openxr.h>
#include <thesis/shared.h>
#include <thesis/options.h>
#include <thesis/threads.h>
#include <thesis/alloc_ext.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


struct xr
{
	simulation_t simulation;

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


private void
xr_init_options(
	xr_t xr
	)
{
	assert_not_null(xr);

	puts("\nXR options:");

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
xr_xr_get_func(
	xr_t xr,
	const char* name
	)
{
	assert_not_null(xr);
	assert_not_null(name);

	PFN_xrVoidFunction func;
	XrResult xr_result = xrGetInstanceProcAddr(xr->xr_instance, name, &func);
	assert_eq(xr_result, XR_SUCCESS);
	assert_not_null(func);

	return func;
}


private void*
xr_vk_get_func(
	xr_t xr,
	const char* name
	)
{
	assert_not_null(xr);
	assert_not_null(name);

	void* func = xr->vkGetInstanceProcAddr(xr->vk_instance, name);
	assert_not_null(func);

	return func;
}


private void
xr_free_str_array(
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
xr_xr_get_instance_extensions(
	xr_t xr,
	const char** extension
	)
{
	assert_not_null(xr);
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

	const char* const* xr_xr_instance_extension = xr_xr_instance_extensions;
	const char* const* xr_xr_instance_extension_end =
		xr_xr_instance_extension + MACRO_ARRAY_LEN(xr_xr_instance_extensions);

	while(xr_xr_instance_extension < xr_xr_instance_extension_end)
	{
		bool found = false;
		const char* extension_name = *(xr_xr_instance_extension++);

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
xr_xr_get_instance_layers(
	xr_t xr,
	const char** layer
	)
{
	assert_not_null(xr);
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

	const char* const* xr_xr_instance_layer = xr_xr_instance_layers;
	const char* const* xr_xr_instance_layer_end =
		xr_xr_instance_layer + MACRO_ARRAY_LEN(xr_xr_instance_layers);

	while(xr_xr_instance_layer < xr_xr_instance_layer_end)
	{
		bool found = false;
		const char* layer_name = *(xr_xr_instance_layer++);

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
xr_init_xr_instance(
	xr_t xr
	)
{
	assert_not_null(xr);

	const char* xr_instance_extensions[64];
	const char** xr_instance_extension =
		xr_xr_get_instance_extensions(xr, xr_instance_extensions);
	assert_lt(xr_instance_extension,
		xr_instance_extensions + MACRO_ARRAY_LEN(xr_instance_extensions));

	const char* xr_instance_layers[64];
	const char** xr_instance_layer =
		xr_xr_get_instance_layers(xr, xr_instance_layers);
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
		.userCallback = xr_xr_debug_callback,
		.userData = NULL
	};

	xr_instance_info.next = &xr_debug_info;
#endif

	XrResult xr_result = xrCreateInstance(&xr_instance_info, &xr->xr_instance);
	assert_eq(xr_result, XR_SUCCESS);

	xr_free_str_array(xr_instance_extensions, xr_instance_extension);
	xr_free_str_array(xr_instance_layers, xr_instance_layer);

#ifndef NDEBUG
	PFN_xrCreateDebugUtilsMessengerEXT xrCreateDebugUtilsMessengerEXT =
		xr_xr_get_func(xr, "xrCreateDebugUtilsMessengerEXT");

	xr_result = xrCreateDebugUtilsMessengerEXT(
		xr->xr_instance, &xr_debug_info, &xr->xr_debug_messenger);
	assert_eq(xr_result, XR_SUCCESS);
#endif

	XrInstanceProperties xr_instance_properties = {XR_TYPE_INSTANCE_PROPERTIES};
	xr_result = xrGetInstanceProperties(xr->xr_instance, &xr_instance_properties);
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
	xr_result = xrGetSystem(xr->xr_instance, &xr_system_info, &xr->xr_system);
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

	xr_result = xrGetSystemProperties(xr->xr_instance,
		xr->xr_system, &xr_system_properties);
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
xr_free_xr_instance(
	xr_t xr
	)
{
	assert_not_null(xr);

#ifndef NDEBUG
	PFN_xrDestroyDebugUtilsMessengerEXT xrDestroyDebugUtilsMessengerEXT =
		xr_xr_get_func(xr, "xrDestroyDebugUtilsMessengerEXT");

	xrDestroyDebugUtilsMessengerEXT(xr->xr_debug_messenger);
#endif

	xrDestroyInstance(xr->xr_instance);
}


private const char**
xr_vk_get_instance_extensions(
	xr_t xr,
	const char** extension
	)
{
	assert_not_null(xr);
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

	const char* const* xr_vk_instance_extension = xr_vk_instance_extensions;
	const char* const* xr_vk_instance_extension_end =
		xr_vk_instance_extension + MACRO_ARRAY_LEN(xr_vk_instance_extensions);

	while(xr_vk_instance_extension < xr_vk_instance_extension_end)
	{
		bool found = false;
		const char* extension_name = *(xr_vk_instance_extension++);

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
xr_vk_get_instance_layers(
	xr_t xr,
	const char** layer
	)
{
	assert_not_null(xr);
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

	const char* const* xr_vk_instance_layer = xr_vk_instance_layers;
	const char* const* xr_vk_instance_layer_end =
		xr_vk_instance_layer + MACRO_ARRAY_LEN(xr_vk_instance_layers);

	while(xr_vk_instance_layer < xr_vk_instance_layer_end)
	{
		bool found = false;
		const char* layer_name = *(xr_vk_instance_layer++);

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
xr_init_vk_instance(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr->vkGetInstanceProcAddr = window_get_vulkan_proc_addr_fn();
	assert_not_null(xr->vkGetInstanceProcAddr);

	volkInitializeCustom(xr->vkGetInstanceProcAddr);

	XrGraphicsRequirementsVulkanKHR xr_xr_requirements =
	{
		.type = XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR,
		.next = NULL,
		.minApiVersionSupported = XR_MAKE_VERSION(1, 4, 309),
		.maxApiVersionSupported = XR_MAKE_VERSION(1, 4, 309)
	};

	PFN_xrGetVulkanGraphicsRequirementsKHR xrGetVulkanGraphicsRequirementsKHR =
		xr_xr_get_func(xr, "xrGetVulkanGraphicsRequirementsKHR");

	XrResult xr_result = xrGetVulkanGraphicsRequirementsKHR(
		xr->xr_instance, xr->xr_system, &xr_xr_requirements);
	assert_eq(xr_result, XR_SUCCESS);

	const char* vk_instance_extensions[64];
	const char** vk_instance_extension =
		xr_vk_get_instance_extensions(xr, vk_instance_extensions);
	assert_lt(vk_instance_extension,
		vk_instance_extensions + MACRO_ARRAY_LEN(vk_instance_extensions));

	const char* vk_instance_layers[64];
	const char** vk_instance_layer =
		xr_vk_get_instance_layers(xr, vk_instance_layers);
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
		.pfnUserCallback = xr_vk_debug_callback,
		.pUserData = NULL
	};

	vk_instance_info.pNext = &vk_debug_info;
#endif

	XrVulkanInstanceCreateInfoKHR xr_vk_instance_info =
	{
		.type = XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR,
		.next = NULL,
		.systemId = xr->xr_system,
		.createFlags = 0,
		.pfnGetInstanceProcAddr = vkGetInstanceProcAddr,
		.vulkanCreateInfo = &vk_instance_info,
		.vulkanAllocator = NULL
	};

	PFN_xrCreateVulkanInstanceKHR xrCreateVulkanInstanceKHR =
		xr_xr_get_func(xr, "xrCreateVulkanInstanceKHR");

	VkResult vk_result;
	xr_result = xrCreateVulkanInstanceKHR(xr->xr_instance,
		&xr_vk_instance_info, &xr->vk_instance, &vk_result);
	assert_eq(xr_result, XR_SUCCESS);
	assert_eq(vk_result, VK_SUCCESS);

	xr_free_str_array(vk_instance_extensions, vk_instance_extension);
	xr_free_str_array(vk_instance_layers, vk_instance_layer);

#ifndef NDEBUG
	PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT =
		xr_vk_get_func(xr, "vkCreateDebugUtilsMessengerEXT");

	vk_result = vkCreateDebugUtilsMessengerEXT(xr->vk_instance,
		&vk_debug_info, NULL, &xr->vk_debug_messenger);
	assert_eq(vk_result, VK_SUCCESS);
#endif

	volkLoadInstanceOnly(xr->vk_instance);
}


private void
xr_free_vk_instance(
	xr_t xr
	)
{
	assert_not_null(xr);

#ifndef NDEBUG
	/* Volk loaded the function already */
	vkDestroyDebugUtilsMessengerEXT(
		xr->vk_instance, xr->vk_debug_messenger, NULL);
#endif

	vkDestroyInstance(xr->vk_instance, NULL);

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
		.systemId = xr->xr_system,
		.vulkanInstance = xr->vk_instance
	};

	PFN_xrGetVulkanGraphicsDevice2KHR xrGetVulkanGraphicsDevice2KHR =
		xr_xr_get_func(xr, "xrGetVulkanGraphicsDevice2KHR");

	XrResult xr_result = xrGetVulkanGraphicsDevice2KHR(
		xr->xr_instance, &xr_vk_device_info, &xr->vk_physical_device);
	assert_eq(xr_result, XR_SUCCESS);

	uint32_t vk_queue_family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(
		xr->vk_physical_device, &vk_queue_family_count, NULL);
	assert_gt(vk_queue_family_count, 0);

	VkQueueFamilyProperties vk_queue_family_properties[vk_queue_family_count];

	vkGetPhysicalDeviceQueueFamilyProperties(
		xr->vk_physical_device, &vk_queue_family_count, vk_queue_family_properties);
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
	xr->vk_queue_id = vk_queue_family_property - vk_queue_family_properties;
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
		xr->vk_physical_device, NULL, &vk_device_extension_count, NULL);

	VkExtensionProperties vk_device_extensions[vk_device_extension_count];

	VkExtensionProperties* vk_device_extension = vk_device_extensions;
	VkExtensionProperties* vk_device_extension_end =
		vk_device_extension + vk_device_extension_count;

	vkEnumerateDeviceExtensionProperties(xr->vk_physical_device,
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
		xr_xr_get_func(xr, "xrGetVulkanDeviceExtensionsKHR");

	XrResult xr_result = xrGetVulkanDeviceExtensionsKHR(xr->xr_instance,
		xr->xr_system, 0, &xr_vk_device_extension_count, NULL);
	assert_eq(xr_result, XR_SUCCESS);

	char xr_vk_device_extensions[xr_vk_device_extension_count + 1];
	xr_vk_device_extensions[xr_vk_device_extension_count] = '\0';

	xr_result = xrGetVulkanDeviceExtensionsKHR(xr->xr_instance, xr->xr_system,
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
		xr->vk_physical_device, &vk_device_layer_count, NULL);

	VkLayerProperties vk_device_layers[vk_device_layer_count];

	VkLayerProperties* vk_device_layer = vk_device_layers;
	VkLayerProperties* vk_device_layer_end = vk_device_layer + vk_device_layer_count;

	vkEnumerateDeviceLayerProperties(xr->vk_physical_device,
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
		.queueFamilyIndex = xr->vk_queue_id,
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
		.systemId = xr->xr_system,
		.createFlags = 0,
		.pfnGetInstanceProcAddr = xr->vkGetInstanceProcAddr,
		.vulkanPhysicalDevice = xr->vk_physical_device,
		.vulkanCreateInfo = &vk_device_info,
		.vulkanAllocator = NULL
	};

	PFN_xrCreateVulkanDeviceKHR xrCreateVulkanDeviceKHR =
		xr_xr_get_func(xr, "xrCreateVulkanDeviceKHR");

	VkResult vk_result;
	XrResult xr_result = xrCreateVulkanDeviceKHR(
		xr->xr_instance, &xr_vk_device_info, &xr->vk_device, &vk_result);
	assert_eq(xr_result, XR_SUCCESS);
	assert_eq(vk_result, VK_SUCCESS);

	xr_free_str_array(vk_device_extensions, vk_device_extension);
	xr_free_str_array(vk_device_layers, vk_device_layer);

	volkLoadDeviceTable(&xr->vk_table, xr->vk_device);

	xr->vk_table.vkGetDeviceQueue(xr->vk_device,
		xr->vk_queue_id, 0, &xr->vk_queue);

	// vkGetPhysicalDeviceMemoryProperties(
	// 	xr->vk_physical_device, &xr->vk_memory_properties);
}


private void
xr_free_vk_logical_device(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr->vk_table.vkDestroyDevice(xr->vk_device, NULL);
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
		.instance = xr->vk_instance,
		.physicalDevice = xr->vk_physical_device,
		.device = xr->vk_device,
		.queueFamilyIndex = xr->vk_queue_id,
		.queueIndex = 0
	};

	XrSessionCreateInfo xr_session_info =
	{
		.type = XR_TYPE_SESSION_CREATE_INFO,
		.next = &xr_vk_binding,
		.createFlags = 0,
		.systemId = xr->xr_system
	};

	XrResult xr_result = xrCreateSession(
		xr->xr_instance, &xr_session_info, &xr->xr_session);
	assert_eq(xr_result, XR_SUCCESS);
}


private void
xr_free_xr_session(
	xr_t xr
	)
{
	assert_not_null(xr);

	XrResult xr_result = xrDestroySession(xr->xr_session);
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


