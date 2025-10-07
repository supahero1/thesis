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
#include <thesis/time.h>
#include <thesis/debug.h>
#include <thesis/atomic.h>
#include <thesis/extent.h>
#include <thesis/shared.h>
#include <thesis/options.h>
#include <thesis/threads.h>
#include <thesis/alloc_ext.h>
#include <thesis/extent_3d.h>

#include <volk.h>

#define XR_USE_PLATFORM_WAYLAND
#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr_platform.h>

#include <signal.h>
#include <string.h>
#include <math.h>

#define VK_STATS_SIZE 64
#define VK_QUERY_SIZE 32
#define VK_MAX_IMAGES 8
#define VK_MAX_FRAMES 2
#define VK_MAX_INSTANCES 128
#define VK_POOL_SIZE 16
#define VK_COMMANDS 8

#define VK_WINDOW_WIDTH 1280
#define VK_WINDOW_HEIGHT 720
#define VK_WINDOW_SENSITIVITY 0.003f
#define VK_WINDOW_SPEED 500.0f

#define XR_COORDINATE_SCALE 1.0f


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
	VK_IMAGE_TYPE_MULTIVIEW,
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
	VK_IMAGE_TYPE_MULTIVIEW_BIT		= MACRO_POWER_OF_2(VK_IMAGE_TYPE_MULTIVIEW),

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
	mat4 projection[2];
	mat4 view[2];
	mat4 light_transform;
	vec4 light_direction;
	vec4 camera_position[2];
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
	mat4 transform[2];
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
		vk_frame_image_t map_ms;

		vk_frame_image_t position;
		vk_frame_image_t normal;
		VkDescriptorSet set;

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

typedef struct vk_timing
{
	VkCommandBuffer command_buffer;
	VkQueryPool pool;
	VkSemaphore semaphore;
	vk_buffer_t buffer;
	uint64_t* results;
	uint32_t* index_map;
	uint32_t count;
	uint32_t current;
	bool first_reset;
}
vk_timing_t;

typedef enum vk_barrier_timing_idx
{
	VK_BARRIER_TIMING_IDX_SHADOW,
	VK_BARRIER_TIMING_IDX_SCENE,
	VK_BARRIER_TIMING_IDX_SSAO,
	VK_BARRIER_TIMING_IDX_SSAO_BLUR,
	VK_BARRIER_TIMING_IDX_OUTPUT,
	MACRO_ENUM_BITS(VK_BARRIER_TIMING_IDX)
}
vk_barrier_timing_idx_t;

struct barrier
{
	VkSemaphore semaphore;
	VkFence fence;
	VkCommandBuffer command_buffer;
	vk_timing_t timing;
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
	pair_t pair;
	VkExtent2D extent;
	XrExtent2Di xr_extent;
	VkViewport viewport;
	VkRect2D scissor;
}
vk_extent_t;

typedef struct vk_command
{
	VkCommandBuffer buffer;
	VkFence fence;
	bool waited;

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

typedef struct xr_pose
{
	triplet_t position;
	triplet_t rotation;
}
xr_pose_t;

struct xr
{
	simulation_t simulation;
	stats_t stats;

	struct
	{
		bool xr_enable;
		str_t xr_runtime;
		bool xr_monado;

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
		float ssao_range_check;
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
	XrSessionState state;
	XrSpace space;

	XrViewConfigurationView* view_configs;
	uint32_t view_count;

	XrHandTrackerEXT hand_tracker_left;
	XrHandTrackerEXT hand_tracker_right;

	PFN_xrLocateHandJointsEXT xrLocateHandJointsEXT;

	XrHandJointLocationEXT hand_joints_left[XR_HAND_JOINT_COUNT_EXT];
	XrHandJointLocationEXT hand_joints_right[XR_HAND_JOINT_COUNT_EXT];
	bool hand_joints_left_active;
	bool hand_joints_right_active;

	XrTime predicted_display_time;

	thread_t thread;

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
		VkPhysicalDeviceMemoryProperties memory_properties;

		VkFormat format;
		VkSampleCountFlagBits samples;
		float anisotropy;
		VkPhysicalDeviceLimits limits;
		float timestamp_period;
		bool timing_enabled;

		XrSwapchain swapchain;
		ipair_t swapchain_extent;

		XrSwapchainImageVulkanKHR* images;
		VkImageView* image_views;
		uint32_t image_count;

		vk_extent_t screen_extent;
		vk_extent_t shadow_extent;
		vk_extent_t ssao_extent;

		VkCommandPool command_pool;
		vk_command_t commands[VK_COMMANDS];
		vk_command_t* command;

		vk_descriptor_set_pool_t** set_pools;
		uint32_t set_pool_count;

		vk_descriptor_set_layout_t sampler_set_layout;
		vk_descriptor_set_layout_t vert_ubo_set_layout;
		vk_descriptor_set_layout_t frag_ubo_set_layout;
		vk_descriptor_set_layout_t scene_set_layout;

		VkSampler depth_sampler;
		VkSampler image_sampler;

		struct
		{
			VkRenderPass render_pass;

			struct
			{
				VkPipelineLayout pipeline_layout;
				VkPipeline pipeline;
			};
		}
		shadow;

		struct
		{
			VkRenderPass render_pass;

			struct
			{
				VkPipelineLayout pipeline_layout;
				VkPipeline pipeline;
			};
		}
		scene;

		struct
		{
			VkRenderPass render_pass;

			struct
			{
				VkPipelineLayout pipeline_layout;
				VkPipeline pipeline;

				vk_frame_image_t noise;
				vk_frame_buffer_t kernel_ubo;
			};
		}
		ssao;

		struct
		{
			VkRenderPass render_pass;

			struct
			{
				VkPipelineLayout pipeline_layout;
				VkPipeline pipeline;
			};
		}
		ssao_blur;

		struct
		{
			VkRenderPass render_pass;

			struct
			{
				VkPipelineLayout pipeline_layout;
				VkPipeline pipeline;

				vk_frame_image_t sky;

				vk_buffer_t vertex_buffer;
				vk_buffer_t index_buffer;
			}
			skybox;

			struct
			{
				VkPipelineLayout pipeline_layout;
				VkPipeline pipeline;
			}
			compose;

			struct
			{
				struct
				{
					VkPipelineLayout pipeline_layout;
					VkPipeline pipeline;
				}
				depth;

				struct
				{
					VkPipelineLayout pipeline_layout;
					VkPipeline pipeline;
				}
				image;
			}
			preview;
		}
		output;

		vk_material_t* materials;
		vk_model_t* models;
		uint32_t material_count;
		uint32_t model_count;

		vk_frame_t frames[VK_MAX_IMAGES];
		vk_barrier_t barriers[VK_MAX_FRAMES];
		vk_barrier_t* barrier;
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


private _Atomic bool is_running;


private void
xr_signal_handler(
	int signum
	)
{
	(void) signum;

	atomic_store_rel(&is_running, false);
}


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

	xr->options.xr_runtime =
		options_get_str(global_options, "xr_runtime", "monado");
	printf("- xr_runtime: %s\n", (char*) xr->options.xr_runtime->str);

	str_t monado = str_init_move_cstr("monado");
	xr->options.xr_monado = str_cmp(xr->options.xr_runtime, monado);
	str_reset(monado);
	str_free(monado);

	xr->options.max_msaa_samples =
		options_get_i64(global_options, "vk_max_msaa_samples", 1, 64, 4);
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

	xr->options.ssao_range_check =
		options_get_f32(global_options, "vk_ssao_range_check", 0.0f, 16.0f, 4.0f);
	printf("- ssao_range_check: %.2f\n", xr->options.ssao_range_check);

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


private void
xr_init_stats(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr->stats = simulation_get_stats(xr->simulation);

	stats_add(xr->stats, "xr_barrier_timing_shadow", VK_STATS_SIZE);
	stats_add(xr->stats, "xr_barrier_timing_scene", VK_STATS_SIZE);
	stats_add(xr->stats, "xr_barrier_timing_ssao", VK_STATS_SIZE);
	stats_add(xr->stats, "xr_barrier_timing_ssao_blur", VK_STATS_SIZE);
	stats_add(xr->stats, "xr_barrier_timing_output", VK_STATS_SIZE);
	stats_add(xr->stats, "xr_command_record_time", VK_STATS_SIZE);
	stats_add(xr->stats, "xr_frame_time", VK_STATS_SIZE);
	stats_add(xr->stats, "xr_frame_delta_time", VK_STATS_SIZE);
}


private void
xr_free_stats(
	xr_t xr
	)
{
	assert_not_null(xr);

	stats_del(xr->stats, "xr_frame_delta_time");
	stats_del(xr->stats, "xr_frame_time");
	stats_del(xr->stats, "xr_command_record_time");
	stats_del(xr->stats, "xr_barrier_timing_output");
	stats_del(xr->stats, "xr_barrier_timing_ssao_blur");
	stats_del(xr->stats, "xr_barrier_timing_ssao");
	stats_del(xr->stats, "xr_barrier_timing_scene");
	stats_del(xr->stats, "xr_barrier_timing_shadow");
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

	puts("\nXR available instance extensions:");

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

	if(xr->options.xr_monado)
	{
		--instance_extension_end;
	}

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

	puts("\nXR available instance layers:");

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

	XrSystemProperties system_properties = {XR_TYPE_SYSTEM_PROPERTIES};

	if(!xr->options.xr_monado)
	{
		system_properties.next = &hand_tracking_properties;
	}

	result = xrGetSystemProperties(xr->instance, xr->system, &system_properties);
	hard_assert_eq(result, XR_SUCCESS);

	if(!xr->options.xr_monado)
	{
		hard_assert_true(hand_tracking_properties.supportsHandTracking);
	}

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

	puts("\nVK available instance extensions:");

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

	puts("\nVK available instance layers:");

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

	PFN_xrCreateVulkanInstanceKHR xrCreateVulkanInstanceKHR = xr_xr_load_func(xr, "xrCreateVulkanInstanceKHR");

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

	XrResult xr_result = xrGetVulkanGraphicsDevice2KHR(xr->instance, &xr_vk_device_info, &xr->vk.physical_device);
	assert_eq(xr_result, XR_SUCCESS);

	uint32_t vk_queue_family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(xr->vk.physical_device, &vk_queue_family_count, NULL);
	assert_gt(vk_queue_family_count, 0);

	VkQueueFamilyProperties vk_queue_family_properties[vk_queue_family_count];

	vkGetPhysicalDeviceQueueFamilyProperties(
		xr->vk.physical_device, &vk_queue_family_count, vk_queue_family_properties);
	assert_gt(vk_queue_family_count, 0);

	VkQueueFamilyProperties* vk_queue_family_property = vk_queue_family_properties;
	VkQueueFamilyProperties* vk_queue_family_property_end = vk_queue_family_property + vk_queue_family_count;

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

	uint32_t available_device_extension_count = 0;
	vkEnumerateDeviceExtensionProperties(xr->vk.physical_device, NULL, &available_device_extension_count, NULL);

	VkExtensionProperties available_device_extensions[available_device_extension_count];

	VkExtensionProperties* available_device_extension = available_device_extensions;
	VkExtensionProperties* available_device_extension_end =
		available_device_extension + available_device_extension_count;

	vkEnumerateDeviceExtensionProperties(xr->vk.physical_device,
		NULL, &available_device_extension_count, available_device_extensions);

	puts("\nXR VK available device extensions:");

	for(
		available_device_extension = available_device_extensions;
		available_device_extension < available_device_extension_end;
		available_device_extension++
		)
	{
		printf("- %s\n", available_device_extension->extensionName);
	}

	puts("");

	const char* const* device_extension = xr_vk_device_extensions;
	const char* const* device_extension_end = device_extension + MACRO_ARRAY_LEN(xr_vk_device_extensions);

	while(device_extension < device_extension_end)
	{
		bool found = false;
		const char* extension_name = *(device_extension++);

		available_device_extension = available_device_extensions;
		while(available_device_extension < available_device_extension_end)
		{
			if(strcmp(extension_name, available_device_extension->extensionName) == 0)
			{
				found = true;
				break;
			}

			available_device_extension++;
		}

		hard_assert_true(found, fprintf(stderr, "XR VK device extension %s not found\n", extension_name));
		printf("+ %s\n", extension_name);
		*(extension++) = cstr_init(extension_name);
	}

	uint32_t device_extension_count = 0;

	PFN_xrGetVulkanDeviceExtensionsKHR xrGetVulkanDeviceExtensionsKHR =
		xr_xr_load_func(xr, "xrGetVulkanDeviceExtensionsKHR");

	XrResult xr_result = xrGetVulkanDeviceExtensionsKHR(
		xr->instance, xr->system, 0, &device_extension_count, NULL);
	assert_eq(xr_result, XR_SUCCESS);

	char device_extensions_str[device_extension_count + 1];
	device_extensions_str[device_extension_count] = '\0';

	xr_result = xrGetVulkanDeviceExtensionsKHR(xr->instance, xr->system,
		device_extension_count, &device_extension_count, device_extensions_str);
	assert_eq(xr_result, XR_SUCCESS);

	char* strtok_r_state = NULL;
	const char* extension_name = strtok_r(device_extensions_str, " ", &strtok_r_state);

	while(extension_name)
	{
		bool found = false;

		available_device_extension = available_device_extensions;
		while(available_device_extension < available_device_extension_end)
		{
			if(strcmp(extension_name, available_device_extension->extensionName) == 0)
			{
				found = true;
				break;
			}

			available_device_extension++;
		}

		hard_assert_true(found, fprintf(stderr, "XR VK device extension %s not found\n", extension_name));
		printf("+ %s\n", extension_name);
		*(extension++) = cstr_init(extension_name);

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

	uint32_t available_device_layer_count = 0;
	vkEnumerateDeviceLayerProperties(xr->vk.physical_device, &available_device_layer_count, NULL);

	VkLayerProperties available_device_layers[available_device_layer_count];

	VkLayerProperties* available_device_layer = available_device_layers;
	VkLayerProperties* available_device_layer_end = available_device_layer + available_device_layer_count;

	vkEnumerateDeviceLayerProperties(xr->vk.physical_device, &available_device_layer_count, available_device_layers);

	puts("\nXR VK available device layers:");

	for(
		available_device_layer = available_device_layers;
		available_device_layer < available_device_layer_end;
		available_device_layer++
		)
	{
		printf("- %s\n", available_device_layer->layerName);
	}

	puts("");

	const char* const* device_layer = xr_vk_device_layers;
	const char* const* device_layer_end = device_layer + MACRO_ARRAY_LEN(xr_vk_device_layers);

	while(device_layer < device_layer_end)
	{
		bool found = false;
		const char* layer_name = *(device_layer++);

		available_device_layer = available_device_layers;
		while(available_device_layer < available_device_layer_end)
		{
			if(strcmp(layer_name, available_device_layer->layerName) == 0)
			{
				found = true;
				break;
			}

			available_device_layer++;
		}

		hard_assert_true(found, fprintf(stderr, "XR VK device layer %s not found\n", layer_name));
		printf("+ %s\n", layer_name);
		*(layer++) = cstr_init(layer_name);
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
	const char** vk_device_extension = xr_vk_get_device_extensions(xr, vk_device_extensions);
	assert_lt(vk_device_extension, vk_device_extensions + MACRO_ARRAY_LEN(vk_device_extensions));

	const char* vk_device_layers[64];
	const char** vk_device_layer = xr_vk_get_device_layers(xr, vk_device_layers);
	assert_lt(vk_device_layer, vk_device_layers + MACRO_ARRAY_LEN(vk_device_layers));

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

	VkPhysicalDeviceFeatures vk_device_features =
	{
		.samplerAnisotropy = !!xr->vk.anisotropy,
		.sampleRateShading = xr->options.sample_shading
	};

	VkPhysicalDeviceFeatures2 vk_device_features2 =
	{
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = &vk_multiview_features,
		.features = vk_device_features
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

	PFN_xrCreateVulkanDeviceKHR xrCreateVulkanDeviceKHR = xr_xr_load_func(xr, "xrCreateVulkanDeviceKHR");

	VkResult vk_result;
	XrResult xr_result = xrCreateVulkanDeviceKHR(xr->instance, &xr_vk_device_info, &xr->vk.device, &vk_result);
	assert_eq(xr_result, XR_SUCCESS);
	assert_eq(vk_result, VK_SUCCESS);

	shared_free_str_array(vk_device_extensions, vk_device_extension);
	shared_free_str_array(vk_device_layers, vk_device_layer);

	volkLoadDeviceTable(&xr->vk.table, xr->vk.device);

	xr->vk.table.vkGetDeviceQueue(xr->vk.device, xr->vk.queue_id, 0, &xr->vk.queue);

	vkGetPhysicalDeviceMemoryProperties(xr->vk.physical_device, &xr->vk.memory_properties);
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
xr_init_vk_device_properties(
	xr_t xr
	)
{
	assert_not_null(xr);

	VkPhysicalDeviceProperties device_properties;
	vkGetPhysicalDeviceProperties(xr->vk.physical_device, &device_properties);

	VkSampleCountFlags sample_count =
		device_properties.limits.framebufferColorSampleCounts &
		device_properties.limits.framebufferDepthSampleCounts;

	VkSampleCountFlagBits max_samples;
	if(sample_count >= VK_SAMPLE_COUNT_64_BIT)
	{
		max_samples = VK_SAMPLE_COUNT_64_BIT;
	}
	else if(sample_count >= VK_SAMPLE_COUNT_32_BIT)
	{
		max_samples = VK_SAMPLE_COUNT_32_BIT;
	}
	else if(sample_count >= VK_SAMPLE_COUNT_16_BIT)
	{
		max_samples = VK_SAMPLE_COUNT_16_BIT;
	}
	else if(sample_count >= VK_SAMPLE_COUNT_8_BIT)
	{
		max_samples = VK_SAMPLE_COUNT_8_BIT;
	}
	else if(sample_count >= VK_SAMPLE_COUNT_4_BIT)
	{
		max_samples = VK_SAMPLE_COUNT_4_BIT;
	}
	else if(sample_count >= VK_SAMPLE_COUNT_2_BIT)
	{
		max_samples = VK_SAMPLE_COUNT_2_BIT;
	}
	else
	{
		hard_assert_unreachable();
	}

	xr->vk.samples = MACRO_MIN(xr->options.max_msaa_samples, max_samples);
	xr->vk.anisotropy = MACRO_MIN(xr->options.max_anisotropy, device_properties.limits.maxSamplerAnisotropy);
	xr->vk.limits = device_properties.limits;
	xr->vk.timestamp_period = device_properties.limits.timestampPeriod;
	xr->vk.timing_enabled = device_properties.limits.timestampComputeAndGraphics == VK_TRUE;

	printf(
		"\nXR VK device properties:\n"
		"\tsamples: %u\n"
		"\tanisotropy: %.1f\n"
		"\ttimestamp_period: %.3f\n"
		"\ttiming_enabled: %d\n",
		xr->vk.samples,
		xr->vk.anisotropy,
		xr->vk.timestamp_period,
		xr->vk.timing_enabled
		);
}


private void
xr_init_vk_capabilities(
	xr_t xr
	)
{
	assert_not_null(xr);

	uint32_t format_count = 0;
	XrResult xr_result = xrEnumerateSwapchainFormats(xr->session, 0, &format_count, NULL);
	hard_assert_eq(xr_result, XR_SUCCESS);
	hard_assert_gt(format_count, 0);

	int64_t formats[format_count];
	xr_result = xrEnumerateSwapchainFormats(xr->session, format_count, &format_count, formats);
	hard_assert_eq(xr_result, XR_SUCCESS);

	int64_t* format = formats;
	int64_t* format_end = format + format_count;

	while(1)
	{
		if(
			*format == VK_FORMAT_R8G8B8A8_SRGB ||
			*format == VK_FORMAT_B8G8R8A8_SRGB
			)
		{
			xr->vk.format = *format;
			break;
		}

		if(++format == format_end)
		{
			hard_assert_unreachable();
		}
	}

	printf(
		"\nXR VK swapchain format: %u\n",
		xr->vk.format
		);
}


private void
xr_init_extent(
	xr_t xr,
	vk_extent_t* extent,
	uint32_t width,
	uint32_t height
	)
{
	assert_not_null(xr);
	assert_not_null(extent);

	extent->width = width;
	extent->height = height;

	extent->pair =
	(pair_t)
	{
		.x = width,
		.y = height
	};

	extent->extent =
	(VkExtent2D)
	{
		.width = width,
		.height = height
	};

	extent->xr_extent =
	(XrExtent2Di)
	{
		.width = width,
		.height = height
	};

	extent->viewport =
	(VkViewport)
	{
		.x = 0.0f,
		.y = 0.0f,
		.width = width,
		.height = height,
		.minDepth = 0.0f,
		.maxDepth = 1.0f
	};

	extent->scissor =
	(VkRect2D)
	{
		.offset = { 0, 0 },
		.extent = extent->extent
	};
}


private void
xr_init_xr_swapchain(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr->vk.swapchain_extent.w = xr->view_configs[0].recommendedImageRectWidth;
	xr->vk.swapchain_extent.h = xr->view_configs[0].recommendedImageRectHeight;

	XrSwapchainCreateInfo swapchain_info =
	{
		.type = XR_TYPE_SWAPCHAIN_CREATE_INFO,
		.next = NULL,
		.createFlags = 0,
		.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT,
		.format = xr->vk.format,
		.sampleCount = 1,
		.width = xr->vk.swapchain_extent.w,
		.height = xr->vk.swapchain_extent.h,
		.faceCount = 1,
		.arraySize = xr->view_count,
		.mipCount = 1
	};

	XrResult result = xrCreateSwapchain(xr->session, &swapchain_info, &xr->vk.swapchain);
	hard_assert_eq(result, XR_SUCCESS);

	result = xrEnumerateSwapchainImages(xr->vk.swapchain, 0, &xr->vk.image_count, NULL);
	hard_assert_eq(result, XR_SUCCESS);

	hard_assert_gt(xr->vk.image_count, 0);
	assert_le(xr->vk.image_count, VK_MAX_IMAGES);

	xr->vk.images = alloc_malloc(sizeof(*xr->vk.images) * xr->vk.image_count);
	assert_ptr(xr->vk.images, sizeof(*xr->vk.images) * xr->vk.image_count);

	for(uint32_t i = 0; i < xr->vk.image_count; ++i)
	{
		xr->vk.images[i] = (XrSwapchainImageVulkanKHR){XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR};
	}

	result = xrEnumerateSwapchainImages(xr->vk.swapchain,
		xr->vk.image_count, &xr->vk.image_count, (void*) xr->vk.images);
	hard_assert_eq(result, XR_SUCCESS);

	xr->vk.image_views = alloc_malloc(sizeof(*xr->vk.image_views) * xr->vk.image_count);
	assert_ptr(xr->vk.image_views, sizeof(*xr->vk.image_views) * xr->vk.image_count);

	for(uint32_t i = 0; i < xr->vk.image_count; ++i)
	{
		VkImageViewCreateInfo view_info =
		{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.image = xr->vk.images[i].image,
			.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
			.format = xr->vk.format,
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
				.layerCount = xr->view_count
			}
		};

		VkResult vk_result = xr->vk.table.vkCreateImageView(xr->vk.device, &view_info, NULL, &xr->vk.image_views[i]);
		hard_assert_eq(vk_result, VK_SUCCESS);
	}

	printf(
		"\nXR VK swapchain:\n"
		"\twidth: %u\n"
		"\theight: %u\n"
		"\tarray_size: %u\n"
		"\timage_count: %u\n",
		xr->vk.swapchain_extent.w,
		xr->vk.swapchain_extent.h,
		xr->view_count,
		xr->vk.image_count
		);

	uint32_t width = xr->vk.swapchain_extent.w;
	uint32_t height = xr->vk.swapchain_extent.h;

	xr_init_extent(xr, &xr->vk.screen_extent, width, height);
	xr_init_extent(xr, &xr->vk.shadow_extent, xr->options.shadow_map_size, xr->options.shadow_map_size);
	xr_init_extent(xr, &xr->vk.ssao_extent, width * xr->options.ssao_scale, height * xr->options.ssao_scale);
}


private void
xr_free_xr_swapchain(
	xr_t xr
	)
{
	assert_not_null(xr);

	for(uint32_t i = 0; i < xr->vk.image_count; ++i)
	{
		xr->vk.table.vkDestroyImageView(xr->vk.device, xr->vk.image_views[i], NULL);
	}
	alloc_free(xr->vk.image_views, sizeof(*xr->vk.image_views) * xr->vk.image_count);

	alloc_free(xr->vk.images, sizeof(*xr->vk.images) * xr->vk.image_count);

	XrResult result = xrDestroySwapchain(xr->vk.swapchain);
	hard_assert_eq(result, XR_SUCCESS);
}


private void
xr_init_commands(
	xr_t xr
	)
{
	assert_not_null(xr);

	VkCommandPoolCreateInfo command_pool_info =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.pNext = NULL,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = xr->vk.queue_id
	};

	VkResult result = xr->vk.table.vkCreateCommandPool(xr->vk.device, &command_pool_info, NULL, &xr->vk.command_pool);
	hard_assert_eq(result, VK_SUCCESS);


	VkCommandBuffer command_buffers[MACRO_ARRAY_LEN(xr->vk.commands)];
	VkCommandBuffer* command_buffer = command_buffers;

	VkCommandBufferAllocateInfo command_buffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = NULL,
		.commandPool = xr->vk.command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = MACRO_ARRAY_LEN(command_buffers)
	};

	result = xr->vk.table.vkAllocateCommandBuffers(xr->vk.device, &command_buffer_info, command_buffers);
	hard_assert_eq(result, VK_SUCCESS);

	VkFenceCreateInfo fence_info =
	{
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.pNext = NULL,
		.flags = VK_FENCE_CREATE_SIGNALED_BIT
	};

	vk_command_t* command = xr->vk.commands;
	vk_command_t* command_end = command + MACRO_ARRAY_LEN(xr->vk.commands);

	while(command != command_end)
	{
		command->buffer = *command_buffer;

		result = xr->vk.table.vkCreateFence(xr->vk.device, &fence_info, NULL, &command->fence);
		hard_assert_eq(result, VK_SUCCESS);

		command->waited = false;

		++command_buffer;
		++command;
	}

	xr->vk.command = xr->vk.commands;
}


private void
xr_free_commands(
	xr_t xr
	)
{
	assert_not_null(xr);

	VkCommandBuffer command_buffers[MACRO_ARRAY_LEN(xr->vk.commands)];
	VkCommandBuffer* command_buffer = command_buffers;

	vk_command_t* command = xr->vk.commands;
	vk_command_t* command_end = command + MACRO_ARRAY_LEN(xr->vk.commands);

	while(command != command_end)
	{
		xr->vk.table.vkDestroyFence(xr->vk.device, command->fence, NULL);

		*command_buffer = command->buffer;

		++command_buffer;
		++command;
	}

	xr->vk.table.vkFreeCommandBuffers(xr->vk.device, xr->vk.command_pool, MACRO_ARRAY_LEN(xr->vk.commands), command_buffers);

	xr->vk.table.vkDestroyCommandPool(xr->vk.device, xr->vk.command_pool, NULL);
}


private void
xr_wait_command(
	xr_t xr,
	vk_command_t* command
	)
{
	assert_not_null(xr);
	assert_not_null(command);

	if(command->waited)
	{
		return;
	}

	VkResult result = xr->vk.table.vkWaitForFences(xr->vk.device, 1, &command->fence, VK_TRUE, UINT64_MAX);
	hard_assert_eq(result, VK_SUCCESS);

	result = xr->vk.table.vkResetFences(xr->vk.device, 1, &command->fence);
	hard_assert_eq(result, VK_SUCCESS);

	command->waited = true;
}


private vk_command_t*
xr_get_command(
	xr_t xr
	)
{
	assert_not_null(xr);

	if(xr->vk.command >= xr->vk.commands + MACRO_ARRAY_LEN(xr->vk.commands))
	{
		xr->vk.command = xr->vk.commands;
	}

	vk_command_t* command = xr->vk.command;
	++xr->vk.command;

	if(!command->waited)
	{
		xr_wait_command(xr, command);
	}

	VkResult result = xr->vk.table.vkResetCommandBuffer(command->buffer, 0);
	hard_assert_eq(result, VK_SUCCESS);

	VkCommandBufferBeginInfo command_buffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = NULL,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		.pInheritanceInfo = NULL
	};

	result = xr->vk.table.vkBeginCommandBuffer(command->buffer, &command_buffer_info);
	hard_assert_eq(result, VK_SUCCESS);

	return command;
}


private void
xr_run_command(
	xr_t xr,
	vk_command_t* command
	)
{
	assert_not_null(xr);
	assert_not_null(command);

	VkResult result = xr->vk.table.vkEndCommandBuffer(command->buffer);
	hard_assert_eq(result, VK_SUCCESS);

	VkSubmitInfo submit_info =
	{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = NULL,
		.waitSemaphoreCount = 0,
		.pWaitSemaphores = NULL,
		.pWaitDstStageMask = NULL,
		.commandBufferCount = 1,
		.pCommandBuffers = &command->buffer,
		.signalSemaphoreCount = 0,
		.pSignalSemaphores = NULL
	};

	result = xr->vk.table.vkQueueSubmit(xr->vk.queue, 1, &submit_info, command->fence);
	hard_assert_eq(result, VK_SUCCESS);

	command->waited = false;
}


private void
xr_descriptor_set_pool_add(
	xr_t xr,
	vk_descriptor_set_pool_t* set_pool
	)
{
	assert_not_null(xr);
	assert_not_null(set_pool);

	vk_descriptor_pool_t* pool = alloc_malloc(sizeof(*pool));
	assert_not_null(pool);

	VkDescriptorPoolSize sizes[set_pool->size_count];
	VkDescriptorPoolSize* size = sizes;
	VkDescriptorPoolSize* size_end = size + set_pool->size_count;

	VkDescriptorPoolSize* set_size = set_pool->sizes;

	while(size < size_end)
	{
		*size =
		(VkDescriptorPoolSize)
		{
			.type = set_size->type,
			.descriptorCount = set_size->descriptorCount * VK_POOL_SIZE
		};

		++size;
		++set_size;
	}

	VkDescriptorPoolCreateInfo pool_info =
	{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.maxSets = VK_POOL_SIZE,
		.poolSizeCount = set_pool->size_count,
		.pPoolSizes = sizes
	};

	VkResult result = xr->vk.table.vkCreateDescriptorPool(xr->vk.device, &pool_info, NULL, &pool->pool);
	hard_assert_eq(result, VK_SUCCESS);

	pool->allocations = 0;

	pool->next = set_pool->head;
	pool->prev = NULL;

	pool->free_next = set_pool->free_head;
	pool->free_prev = NULL;

	if(set_pool->head)
	{
		set_pool->head->prev = pool;
	}
	set_pool->head = pool;

	if(set_pool->free_head)
	{
		set_pool->free_head->free_prev = pool;
	}
	set_pool->free_head = pool;
}


private void
xr_get_descriptor_sets(
	xr_t xr,
	vk_descriptor_set_layout_t dst_set_layout,
	VkDescriptorSet* sets,
	uint32_t count
	)
{
	assert_not_null(xr);
	assert_not_null(dst_set_layout.layout);
	assert_ptr(sets, count);

	if(!count)
	{
		return;
	}

	vk_descriptor_set_pool_t* set_pool = dst_set_layout.set_pool;
	assert_not_null(set_pool);
	assert_neq(dst_set_layout.multiplier, 0);

	VkDescriptorSetLayout set_layouts[VK_POOL_SIZE];

	while(count > 0)
	{
		vk_descriptor_pool_t* pool = set_pool->free_head;
		while(pool)
		{
			if((pool->allocations + dst_set_layout.multiplier) <= VK_POOL_SIZE)
			{
				break;
			}
			pool = pool->free_next;
		}

		if(!pool)
		{
			xr_descriptor_set_pool_add(xr, set_pool);
			continue;
		}

		uint32_t available_slots = VK_POOL_SIZE - pool->allocations;
		uint32_t can_alloc_sets = available_slots / dst_set_layout.multiplier;
		uint32_t alloc_count = MACRO_MIN(count, can_alloc_sets);

		for(uint32_t i = 0; i < alloc_count; ++i)
		{
			set_layouts[i] = dst_set_layout.layout;
		}

		VkDescriptorSetAllocateInfo alloc_info =
		{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.pNext = NULL,
			.descriptorPool = pool->pool,
			.descriptorSetCount = alloc_count,
			.pSetLayouts = set_layouts
		};

		VkResult result = xr->vk.table.vkAllocateDescriptorSets(xr->vk.device, &alloc_info, sets);
		hard_assert_eq(result, VK_SUCCESS);

		pool->allocations += alloc_count * dst_set_layout.multiplier;
		sets += alloc_count;
		count -= alloc_count;

		if(pool->allocations == VK_POOL_SIZE)
		{
			if(set_pool->free_head == pool)
			{
				set_pool->free_head = pool->free_next;
			}

			if(pool->free_next)
			{
				pool->free_next->free_prev = pool->free_prev;
			}
			pool->free_next = NULL;

			if(pool->free_prev)
			{
				pool->free_prev->free_next = pool->free_next;
			}
			pool->free_prev = NULL;
		}
	}
}


private void
xr_init_set_layout(
	xr_t xr,
	VkDescriptorPoolSize* sizes,
	uint32_t size_count,
	VkShaderStageFlags stage_flags,
	vk_descriptor_set_layout_t* set_layout
	)
{
	assert_not_null(xr);
	assert_not_null(sizes);
	assert_neq(size_count, 0);
	assert_not_null(set_layout);

	uint32_t type_map[11] = {0};
	uint32_t unique_types = 0;

	VkDescriptorPoolSize* size = sizes;
	VkDescriptorPoolSize* size_end = size + size_count;

	while(size < size_end)
	{
		if(type_map[size->type] == 0)
		{
			type_map[size->type] = ++unique_types;
		}

		++size;
	}

	VkDescriptorPoolSize* new_sizes = alloc_calloc(sizeof(*new_sizes) * unique_types);
	assert_not_null(new_sizes);

	VkDescriptorSetLayoutBinding bindings[size_count];
	VkDescriptorSetLayoutBinding* binding = bindings;

	size = sizes;

	for(uint32_t i = 0; i < size_count; ++i, ++binding, ++size)
	{
		*binding =
		(VkDescriptorSetLayoutBinding)
		{
			.binding = i,
			.descriptorType = size->type,
			.descriptorCount = size->descriptorCount,
			.stageFlags = stage_flags,
			.pImmutableSamplers = NULL
		};

		VkDescriptorPoolSize* new_size = new_sizes + type_map[size->type] - 1;
		new_size->type = size->type;
		new_size->descriptorCount += size->descriptorCount;
	}

	VkDescriptorSetLayoutCreateInfo set_layout_info =
	{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.bindingCount = size_count,
		.pBindings = bindings
	};

	VkResult result = xr->vk.table.vkCreateDescriptorSetLayout(
		xr->vk.device, &set_layout_info, NULL, &set_layout->layout);
	hard_assert_eq(result, VK_SUCCESS);

	sizes = new_sizes;
	size_count = unique_types;


	vk_descriptor_set_pool_t** set_pool_ptr = xr->vk.set_pools;
	vk_descriptor_set_pool_t** set_pool_ptr_end = set_pool_ptr + xr->vk.set_pool_count;

	while(set_pool_ptr < set_pool_ptr_end)
	{
		vk_descriptor_set_pool_t* set_pool = *set_pool_ptr;

		if(set_pool->size_count != size_count)
		{
			++set_pool_ptr;
			continue;
		}

		VkDescriptorPoolSize* set_pool_size = set_pool->sizes;
		VkDescriptorPoolSize* set_pool_size_end = set_pool_size + set_pool->size_count;

		size = sizes;
		uint32_t multiplier = 0;

		bool found = true;

		while(set_pool_size < set_pool_size_end)
		{
			if(
				set_pool_size->type == size->type
				)
			{
				uint32_t times = size->descriptorCount / set_pool_size->descriptorCount;
				if(!times)
				{
					found = false;
					break;
				}

				if(!multiplier)
				{
					multiplier = times;
				}
				else if(multiplier != times)
				{
					found = false;
					break;
				}
			}
			else
			{
				found = false;
				break;
			}

			++set_pool_size;
			++size;
		}

		if(found)
		{
			set_layout->multiplier = multiplier;
			set_layout->set_pool = set_pool;
			++set_pool->refs;

			alloc_free(sizes, sizeof(*sizes) * size_count);
			return;
		}

		++set_pool_ptr;
	}


	xr->vk.set_pools = alloc_remalloc(xr->vk.set_pools,
		sizeof(*xr->vk.set_pools) * xr->vk.set_pool_count,
		sizeof(*xr->vk.set_pools) * (xr->vk.set_pool_count + 1));
	assert_not_null(xr->vk.set_pools);

	set_pool_ptr = xr->vk.set_pools + xr->vk.set_pool_count++;
	*set_pool_ptr = alloc_malloc(sizeof(**set_pool_ptr));
	assert_not_null(*set_pool_ptr);

	vk_descriptor_set_pool_t* set_pool = *set_pool_ptr;
	*set_pool =
	(vk_descriptor_set_pool_t)
	{
		.head = NULL,
		.free_head = NULL,
		.sizes = sizes,
		.size_count = size_count,
		.refs = 1
	};

	set_layout->multiplier = 1;
	set_layout->set_pool = set_pool;
}


private void
xr_free_set_layout(
	xr_t xr,
	vk_descriptor_set_layout_t* set_layout
	)
{
	assert_not_null(xr);
	assert_not_null(set_layout);

	vk_descriptor_set_pool_t* set_pool = set_layout->set_pool;

	if(!--set_pool->refs)
	{
		vk_descriptor_pool_t* pool = set_pool->head;
		while(pool)
		{
			xr->vk.table.vkDestroyDescriptorPool(xr->vk.device, pool->pool, NULL);

			vk_descriptor_pool_t* next = pool->next;
			alloc_free(pool, sizeof(*pool));
			pool = next;
		}

		alloc_free(set_pool->sizes, sizeof(*set_pool->sizes) * set_pool->size_count);
	}

	xr->vk.table.vkDestroyDescriptorSetLayout(xr->vk.device, set_layout->layout, NULL);
}


private void
xr_init_vert_set_layout(
	xr_t xr,
	VkDescriptorPoolSize* sizes,
	uint32_t size_count,
	vk_descriptor_set_layout_t* set_layout
	)
{
	xr_init_set_layout(xr, sizes, size_count, VK_SHADER_STAGE_VERTEX_BIT, set_layout);
}


private void
xr_init_frag_set_layout(
	xr_t xr,
	VkDescriptorPoolSize* sizes,
	uint32_t size_count,
	vk_descriptor_set_layout_t* set_layout
	)
{
	xr_init_set_layout(xr, sizes, size_count, VK_SHADER_STAGE_FRAGMENT_BIT, set_layout);
}


private void
xr_init_set_layouts(
	xr_t xr
	)
{
	assert_not_null(xr);

	VkDescriptorPoolSize sampler_sizes[] =
	{
		{
			.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1
		}
	};

	xr_init_frag_set_layout(xr, sampler_sizes, MACRO_ARRAY_LEN(sampler_sizes), &xr->vk.sampler_set_layout);


	VkDescriptorPoolSize vert_ubo_sizes[] =
	{
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1
		}
	};

	xr_init_vert_set_layout(xr, vert_ubo_sizes, MACRO_ARRAY_LEN(vert_ubo_sizes), &xr->vk.vert_ubo_set_layout);


	VkDescriptorPoolSize frag_ubo_sizes[] =
	{
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1
		}
	};

	xr_init_frag_set_layout(xr, frag_ubo_sizes, MACRO_ARRAY_LEN(frag_ubo_sizes), &xr->vk.frag_ubo_set_layout);


	VkDescriptorPoolSize scene_sizes[] =
	{
		{
			.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1
		},
		{
			.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1
		}
	};

	xr_init_frag_set_layout(xr, scene_sizes, MACRO_ARRAY_LEN(scene_sizes), &xr->vk.scene_set_layout);
}


private void
xr_free_set_layouts(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_set_layout(xr, &xr->vk.scene_set_layout);
	xr_free_set_layout(xr, &xr->vk.frag_ubo_set_layout);
	xr_free_set_layout(xr, &xr->vk.vert_ubo_set_layout);
	xr_free_set_layout(xr, &xr->vk.sampler_set_layout);
}


private void
xr_init_sets(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_init_set_layouts(xr);
}


private void
xr_free_sets(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_set_layouts(xr);

	vk_descriptor_set_pool_t** set_pool_ptr = xr->vk.set_pools;
	vk_descriptor_set_pool_t** set_pool_ptr_end = set_pool_ptr + xr->vk.set_pool_count;

	while(set_pool_ptr < set_pool_ptr_end)
	{
		vk_descriptor_set_pool_t* set_pool = *set_pool_ptr;
		alloc_free(set_pool, sizeof(*set_pool));

		++set_pool_ptr;
	}

	alloc_free(xr->vk.set_pools, sizeof(*xr->vk.set_pools) * xr->vk.set_pool_count);
}


private uint32_t
xr_get_memory(
	xr_t xr,
	uint32_t bits,
	VkMemoryPropertyFlags flags
	)
{
	for(uint32_t i = 0; i < xr->vk.memory_properties.memoryTypeCount; ++i)
	{
		if(
			(bits & (1 << i)) &&
			(xr->vk.memory_properties.memoryTypes[i].propertyFlags & flags) == flags
			)
		{
			return i;
		}
	}

	hard_assert_unreachable();
}


private void
xr_init_buffer(
	xr_t xr,
	VkDeviceSize size,
	VkBufferUsageFlags usage,
	VkMemoryPropertyFlags flags,
	vk_buffer_t* buffer
	)
{
	assert_not_null(xr);
	assert_not_null(buffer);

	VkBufferCreateInfo buffer_info =
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

	VkResult result = xr->vk.table.vkCreateBuffer(xr->vk.device, &buffer_info, NULL, &buffer->buffer);
	hard_assert_eq(result, VK_SUCCESS);

	VkMemoryRequirements memory_requirements;
	xr->vk.table.vkGetBufferMemoryRequirements(xr->vk.device, buffer->buffer, &memory_requirements);

	uint32_t memory_type_index = xr_get_memory(xr, memory_requirements.memoryTypeBits, flags);

	VkMemoryAllocateInfo memory_info =
	{
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext = NULL,
		.allocationSize = memory_requirements.size,
		.memoryTypeIndex = memory_type_index
	};

	result = xr->vk.table.vkAllocateMemory(xr->vk.device, &memory_info, NULL, &buffer->memory);
	hard_assert_eq(result, VK_SUCCESS);

	result = xr->vk.table.vkBindBufferMemory(xr->vk.device, buffer->buffer, buffer->memory, 0);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
xr_free_buffer(
	xr_t xr,
	vk_buffer_t* buffer
	)
{
	assert_not_null(xr);

	xr->vk.table.vkFreeMemory(xr->vk.device, buffer->memory, NULL);
	xr->vk.table.vkDestroyBuffer(xr->vk.device, buffer->buffer, NULL);
}


private void
xr_init_staging_buffer(
	xr_t xr,
	vk_command_t* command,
	VkDeviceSize size
	)
{
	assert_not_null(xr);
	assert_not_null(command);

	VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	xr_init_buffer(xr, size, usage, flags, &command->staging_buffer);
}


private void
xr_free_staging_buffer(
	xr_t xr,
	vk_command_t* command
	)
{
	assert_not_null(xr);
	assert_not_null(command);

	xr_free_buffer(xr, &command->staging_buffer);
}


private void
xr_free_all_staging_buffers(
	xr_t xr
	)
{
	assert_not_null(xr);

	vk_command_t* command = xr->vk.commands;
	vk_command_t* command_end = command + MACRO_ARRAY_LEN(xr->vk.commands);

	while(command != command_end)
	{
		xr_free_staging_buffer(xr, command);

		++command;
	}
}


private void
xr_init_vertex_buffer(
	xr_t xr,
	VkDeviceSize size,
	vk_buffer_t* buffer
	)
{
	assert_not_null(xr);
	assert_not_null(buffer);

	VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	xr_init_buffer(xr, size, usage, flags, buffer);
}


private void
xr_init_index_buffer(
	xr_t xr,
	VkDeviceSize size,
	vk_buffer_t* buffer
	)
{
	assert_not_null(xr);
	assert_not_null(buffer);

	VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
	VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	xr_init_buffer(xr, size, usage, flags, buffer);
}


private void
xr_init_ubo_buffer(
	xr_t xr,
	VkDeviceSize size,
	vk_descriptor_set_layout_t set_layout,
	vk_frame_buffer_t* frame_buffer
	)
{
	assert_not_null(xr);
	assert_not_null(set_layout.layout);
	assert_not_null(frame_buffer);

	VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	xr_init_buffer(xr, size, usage, flags, &frame_buffer->buffer);

	if(!frame_buffer->set)
	{
		xr_get_descriptor_sets(xr, set_layout, &frame_buffer->set, 1);
	}

	VkDescriptorBufferInfo descriptor_buffer_info =
	{
		.buffer = frame_buffer->buffer.buffer,
		.offset = 0,
		.range = VK_WHOLE_SIZE
	};

	VkWriteDescriptorSet write_set =
	{
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.pNext = NULL,
		.dstSet = frame_buffer->set,
		.dstBinding = 0,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pImageInfo = NULL,
		.pBufferInfo = &descriptor_buffer_info,
		.pTexelBufferView = NULL
	};

	xr->vk.table.vkUpdateDescriptorSets(xr->vk.device, 1, &write_set, 0, NULL);
}


private void
xr_free_ubo_buffer(
	xr_t xr,
	vk_frame_buffer_t* frame_buffer
	)
{
	assert_not_null(xr);
	assert_not_null(frame_buffer);

	xr_free_buffer(xr, &frame_buffer->buffer);
}


private void
xr_init_vert_ubo_buffer(
	xr_t xr,
	VkDeviceSize size,
	vk_frame_buffer_t* frame_buffer
	)
{
	assert_not_null(xr);
	assert_not_null(frame_buffer);

	xr_init_ubo_buffer(xr, size, xr->vk.vert_ubo_set_layout, frame_buffer);
}


private void
xr_init_frag_ubo_buffer(
	xr_t xr,
	VkDeviceSize size,
	vk_frame_buffer_t* frame_buffer
	)
{
	assert_not_null(xr);
	assert_not_null(frame_buffer);

	xr_init_ubo_buffer(xr, size, xr->vk.frag_ubo_set_layout, frame_buffer);
}


private void
xr_copy_to_buffer(
	xr_t xr,
	vk_buffer_t* buffer,
	const void* data,
	VkDeviceSize size
	)
{
	assert_not_null(xr);
	assert_not_null(buffer);
	assert_ptr(data, size);

	if(!size)
	{
		return;
	}

	vk_command_t* command = xr_get_command(xr);

	xr_free_staging_buffer(xr, command);
	xr_init_staging_buffer(xr, command, size);

	void* mapped_data;
	VkResult result = xr->vk.table.vkMapMemory(xr->vk.device, command->staging_buffer.memory, 0, size, 0, &mapped_data);
	hard_assert_eq(result, VK_SUCCESS);

	memcpy(mapped_data, data, size);

	xr->vk.table.vkUnmapMemory(xr->vk.device, command->staging_buffer.memory);

	VkBufferCopy buffer_copy =
	{
		.srcOffset = 0,
		.dstOffset = 0,
		.size = size
	};

	xr->vk.table.vkCmdCopyBuffer(command->buffer, command->staging_buffer.buffer, buffer->buffer, 1, &buffer_copy);

	xr_run_command(xr, command);
}


private void
xr_read_from_buffer(
	xr_t xr,
	vk_buffer_t* buffer,
	void* data,
	VkDeviceSize size
	)
{
	assert_not_null(xr);
	assert_not_null(buffer);
	assert_ptr(data, size);

	if(!size)
	{
		return;
	}

	vk_command_t* command = xr_get_command(xr);

	xr_free_staging_buffer(xr, command);
	xr_init_staging_buffer(xr, command, size);

	VkBufferCopy buffer_copy =
	{
		.srcOffset = 0,
		.dstOffset = 0,
		.size = size
	};

	xr->vk.table.vkCmdCopyBuffer(command->buffer, buffer->buffer, command->staging_buffer.buffer, 1, &buffer_copy);

	xr_run_command(xr, command);
	xr_wait_command(xr, command);

	void* mapped_data;
	VkResult result = xr->vk.table.vkMapMemory(xr->vk.device, command->staging_buffer.memory, 0, size, 0, &mapped_data);
	hard_assert_eq(result, VK_SUCCESS);

	memcpy(data, mapped_data, size);

	xr->vk.table.vkUnmapMemory(xr->vk.device, command->staging_buffer.memory);
}


private void
xr_copy_texture_to_image(
	xr_t xr,
	vk_image_t* image
	)
{
	assert_not_null(xr);
	assert_not_null(image);

	vk_command_t* command = xr_get_command(xr);

	xr_free_staging_buffer(xr, command);
	xr_init_staging_buffer(xr, command, image->size);

	void* mapped_data;
	VkResult result = xr->vk.table.vkMapMemory(xr->vk.device,
		command->staging_buffer.memory, 0, image->size, 0, &mapped_data);
	hard_assert_eq(result, VK_SUCCESS);

	memcpy(mapped_data, image->data, image->size);

	xr->vk.table.vkUnmapMemory(xr->vk.device, command->staging_buffer.memory);

	uint32_t count = image->levels * image->layers;
	VkBufferImageCopy buffer_image_copies[count];
	VkBufferImageCopy* buffer_image_copy = buffer_image_copies;

	uint32_t width = image->width;
	uint32_t height = image->height;
	uint32_t offset = 0;

	for(uint32_t level = 0; level < image->levels; ++level)
	{
		uint32_t stride = width * height * 4;

		for(uint32_t layer = 0; layer < image->layers; ++layer)
		{
			*(buffer_image_copy++) =
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

	xr->vk.table.vkCmdCopyBufferToImage(command->buffer, command->staging_buffer.buffer,
		image->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, count, buffer_image_copies);

	xr_run_command(xr, command);
}


private void
xr_transition_image_layout(
	xr_t xr,
	vk_image_t* image,
	VkImageLayout from,
	VkImageLayout to
	)
{
	assert_not_null(xr);
	assert_not_null(image);

	vk_command_t* command = xr_get_command(xr);

	VkPipelineStageFlags src_stage;
	VkPipelineStageFlags dst_stage;

	VkImageMemoryBarrier barrier =
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
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

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
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}
	else
	{
		hard_assert_unreachable();
	}

	xr->vk.table.vkCmdPipelineBarrier(command->buffer, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &barrier);

	xr_run_command(xr, command);
}


private void
xr_copy_data_to_image(
	xr_t xr,
	vk_image_t* image
	)
{
	assert_not_null(xr);
	assert_not_null(image);

	xr_transition_image_layout(xr, image,
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

	xr_copy_texture_to_image(xr, image);

	xr_transition_image_layout(xr, image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}


private void
xr_init_image(
	xr_t xr,
	vk_image_t* image
	)
{
	assert_not_null(xr);
	assert_not_null(image);

	assert_neq(image->type & (VK_IMAGE_TYPE_DEPTH_BIT | VK_IMAGE_TYPE_TEXTURE_BIT),
		VK_IMAGE_TYPE_DEPTH_BIT | VK_IMAGE_TYPE_TEXTURE_BIT);

	VkImageCreateFlags create_flags = 0;
	VkImageViewType view_type = VK_IMAGE_VIEW_TYPE_2D;


	image->usage = 0;

	if(image->type & VK_IMAGE_TYPE_DEPTH_BIT)
	{
		image->format = VK_FORMAT_D32_SFLOAT;
		image->aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
		image->usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
	}
	else
	{
		image->aspect = VK_IMAGE_ASPECT_COLOR_BIT;

		if(image->type & VK_IMAGE_TYPE_ATTACHMENT_BIT)
		{
			image->usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		}

		if(!(image->type & VK_IMAGE_TYPE_CUSTOM_FORMAT_BIT))
		{
			if(image->type & VK_IMAGE_TYPE_TEXTURE_BIT)
			{
				image->format = VK_FORMAT_R8G8B8A8_SRGB;
			}
			else
			{
				image->format = xr->vk.format;
			}
		}
	}

	if(image->type & VK_IMAGE_TYPE_TEXTURE_BIT)
	{
		image->usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	}

	if(image->type & VK_IMAGE_TYPE_SAMPLED_BIT)
	{
		image->usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
	}

	if(image->type & VK_IMAGE_TYPE_TRANSIENT_BIT)
	{
		image->usage |= VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
	}

	if(image->type & VK_IMAGE_TYPE_MULTISAMPLED_BIT)
	{
		image->samples = xr->vk.samples;
	}
	else
	{
		image->samples = VK_SAMPLE_COUNT_1_BIT;
	}

	if(!(image->type & VK_IMAGE_TYPE_CUSTOM_SIZE_BIT))
	{
		image->width = xr->vk.screen_extent.width;
		image->height = xr->vk.screen_extent.height;
	}

	bool image_backed_texture =
		(image->type & VK_IMAGE_TYPE_TEXTURE_BIT) && image->path && !str_is_empty(image->path);

	if(image->type & VK_IMAGE_TYPE_TEXTURE_BIT)
	{
		if(image_backed_texture)
		{
			assert_eq(image->type & VK_IMAGE_TYPE_CUSTOM_SIZE_BIT, 0);

			bool cube = image->type & VK_IMAGE_TYPE_CUBE_BIT;
			simulation_texture_t* texture = simulation_get_texture(xr->simulation, image->path, cube);
			image->data = texture->data;
			image->size = texture->size;
			image->width = texture->width;
			image->height = texture->height;
			image->levels = 1 + MACRO_MIN(
				MACRO_LOG2(MACRO_MAX(image->width, image->height)),
				xr->options.mipmap_levels
				);
			image->layers = texture->layers;

			if(cube)
			{
				create_flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
				view_type = VK_IMAGE_VIEW_TYPE_CUBE;
			}
		}
		else
		{
			image->levels = 1;
			image->layers = 1;
		}
	}
	else
	{
		image->data = NULL;
		image->size = 0;
		image->levels = 1;
		if(image->type & VK_IMAGE_TYPE_MULTIVIEW_BIT)
		{
			image->layers = 2;
			view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
		}
		else
		{
			image->layers = 1;
		}
	}


	VkImageCreateInfo image_info =
	{
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.pNext = NULL,
		.flags = create_flags,
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

	VkResult result = xr->vk.table.vkCreateImage(xr->vk.device, &image_info, NULL, &image->image);
	hard_assert_eq(result, VK_SUCCESS);

	VkMemoryRequirements memory_requirements;
	xr->vk.table.vkGetImageMemoryRequirements(xr->vk.device, image->image, &memory_requirements);

	uint32_t memory_type_index = xr_get_memory(xr,
		memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	VkMemoryAllocateInfo memory_info =
	{
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext = NULL,
		.allocationSize = memory_requirements.size,
		.memoryTypeIndex = memory_type_index
	};

	result = xr->vk.table.vkAllocateMemory(
		xr->vk.device, &memory_info, NULL, &image->memory);
	hard_assert_eq(result, VK_SUCCESS);

	result = xr->vk.table.vkBindImageMemory(
		xr->vk.device, image->image, image->memory, 0);
	hard_assert_eq(result, VK_SUCCESS);

	VkImageViewCreateInfo image_view_info =
	{
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.image = image->image,
		.viewType = view_type,
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

	result = xr->vk.table.vkCreateImageView(xr->vk.device, &image_view_info, NULL, &image->view);
	hard_assert_eq(result, VK_SUCCESS);

	if(image_backed_texture)
	{
		xr_copy_data_to_image(xr, image);
	}
}


private void
xr_free_image(
	xr_t xr,
	vk_image_t* image
	)
{
	assert_not_null(xr);
	assert_not_null(image);

	xr->vk.table.vkDestroyImageView(xr->vk.device, image->view, NULL);
	xr->vk.table.vkDestroyImage(xr->vk.device, image->image, NULL);
	xr->vk.table.vkFreeMemory(xr->vk.device, image->memory, NULL);
}


private void
xr_write_images_to_set(
	xr_t xr,
	VkDescriptorSet set,
	vk_image_t* images,
	uint32_t count
	)
{
	assert_not_null(xr);
	assert_not_null(set);
	assert_ptr(images, count);

	if(!count)
	{
		return;
	}

	VkDescriptorImageInfo image_infos[count];
	VkDescriptorImageInfo* image_info = image_infos;
	VkDescriptorImageInfo* image_info_end = image_info + count;

	VkWriteDescriptorSet write_sets[count];
	VkWriteDescriptorSet* write_set = write_sets;

	while(image_info < image_info_end)
	{
		VkImageLayout layout;
		VkSampler sampler;
		if(images->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)
		{
			layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
			sampler = xr->vk.depth_sampler;
		}
		else
		{
			layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			sampler = xr->vk.image_sampler;
		}

		*image_info =
		(VkDescriptorImageInfo)
		{
			.sampler = sampler,
			.imageView = images->view,
			.imageLayout = layout
		};

		*(write_set++) =
		(VkWriteDescriptorSet)
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.pNext = NULL,
			.dstSet = set,
			.dstBinding = image_info - image_infos,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = image_info,
			.pBufferInfo = NULL,
			.pTexelBufferView = NULL
		};

		++image_info;
		++images;
	}

	xr->vk.table.vkUpdateDescriptorSets(xr->vk.device, count, write_sets, 0, NULL);
}


private void
xr_init_frame_image(
	xr_t xr,
	vk_frame_image_t* frame_image
	)
{
	assert_not_null(xr);
	assert_not_null(frame_image);

	xr_init_image(xr, &frame_image->image);

	if(!frame_image->set)
	{
		xr_get_descriptor_sets(xr, xr->vk.sampler_set_layout, &frame_image->set, 1);
	}

	xr_write_images_to_set(xr, frame_image->set, &frame_image->image, 1);
}


private void
xr_free_frame_image(
	xr_t xr,
	vk_frame_image_t* frame_image
	)
{
	assert_not_null(xr);
	assert_not_null(frame_image);

	xr_free_image(xr, &frame_image->image);
}


private void
xr_init_timing(
	xr_t xr,
	vk_timing_t* timing,
	VkCommandBuffer command_buffer,
	uint32_t count
	)
{
	assert_not_null(xr);
	assert_not_null(timing);

	if(!xr->vk.timing_enabled)
	{
		return;
	}

	timing->command_buffer = command_buffer;
	timing->count = count;

	VkQueryPoolCreateInfo query_pool_info =
	{
		.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
		.queryType = VK_QUERY_TYPE_TIMESTAMP,
		.queryCount = timing->count * 2
	};

	VkResult result = xr->vk.table.vkCreateQueryPool(xr->vk.device, &query_pool_info, NULL, &timing->pool);
	hard_assert_eq(result, VK_SUCCESS);

	VkSemaphoreCreateInfo semaphore_info =
	{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0
	};

	result = xr->vk.table.vkCreateSemaphore(xr->vk.device, &semaphore_info, NULL, &timing->semaphore);
	hard_assert_eq(result, VK_SUCCESS);

	VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	xr_init_buffer(xr, sizeof(uint64_t) * count * 2, usage, flags, &timing->buffer);

	timing->results = alloc_malloc(sizeof(*timing->results) * timing->count * 2);
	assert_not_null(timing->results);

	timing->index_map = alloc_malloc(sizeof(*timing->index_map) * timing->count);
	assert_not_null(timing->index_map);

	timing->current = 0;
	timing->first_reset = true;
}


private void
xr_free_timing(
	xr_t xr,
	vk_timing_t* timing
	)
{
	assert_not_null(xr);

	if(!xr->vk.timing_enabled)
	{
		return;
	}

	alloc_free(timing->index_map, sizeof(*timing->index_map) * timing->count);
	alloc_free(timing->results, sizeof(*timing->results) * timing->count * 2);

	xr_free_buffer(xr, &timing->buffer);

	xr->vk.table.vkDestroySemaphore(xr->vk.device, timing->semaphore, NULL);
	xr->vk.table.vkDestroyQueryPool(xr->vk.device, timing->pool, NULL);
}


private void
xr_timing_start(
	xr_t xr,
	vk_timing_t* timing,
	uint32_t index
	)
{
	assert_not_null(xr);

	if(!xr->vk.timing_enabled)
	{
		return;
	}

	assert_lt(index, timing->count);

	uint32_t query_index = timing->current;
	assert_lt(query_index, timing->count * 2 - 1);

	xr->vk.table.vkCmdWriteTimestamp(timing->command_buffer,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timing->pool, query_index);

	timing->index_map[index] = query_index;
}


private void
xr_timing_end(
	xr_t xr,
	vk_timing_t* timing,
	uint32_t index
	)
{
	assert_not_null(xr);

	if(!xr->vk.timing_enabled)
	{
		return;
	}

	assert_lt(index, timing->count);

	uint32_t end_query_index = timing->index_map[index] + 1;

	xr->vk.table.vkCmdWriteTimestamp(timing->command_buffer,
		VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing->pool, end_query_index);

	timing->current = end_query_index + 1;
}


private void
xr_timing_query(
	xr_t xr,
	vk_timing_t* timing
	)
{
	assert_not_null(xr);

	if(!xr->vk.timing_enabled)
	{
		return;
	}

	xr->vk.table.vkCmdCopyQueryPoolResults(timing->command_buffer,
		timing->pool, 0, timing->count * 2, timing->buffer.buffer, 0,
		sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
}


private void
xr_timing_load(
	xr_t xr,
	vk_timing_t* timing
	)
{
	assert_not_null(xr);

	if(!xr->vk.timing_enabled)
	{
		return;
	}

	xr_read_from_buffer(xr, &timing->buffer, timing->results, sizeof(*timing->results) * timing->count * 2);
}


private uint64_t
xr_timing_get(
	xr_t xr,
	vk_timing_t* timing,
	uint32_t index
	)
{
	assert_not_null(xr);

	if(!xr->vk.timing_enabled)
	{
		return 0;
	}

	assert_lt(index, timing->count);

	uint32_t start_query_index = timing->index_map[index];
	uint32_t end_query_index = start_query_index + 1;

	return (timing->results[end_query_index] - timing->results[start_query_index]) * xr->vk.timestamp_period;
}


private void
xr_timing_reset(
	xr_t xr,
	vk_timing_t* timing
	)
{
	assert_not_null(xr);

	if(!xr->vk.timing_enabled || (!timing->current && !timing->first_reset))
	{
		return;
	}

	xr->vk.table.vkCmdResetQueryPool(timing->command_buffer, timing->pool, 0, timing->count * 2);
	timing->current = 0;
	timing->first_reset = false;
}


private VkShaderModule
xr_create_shader(
	xr_t xr,
	const char* path
	)
{
	file_t file;
	bool status = file_read(path, &file);
	hard_assert_true(status);

	VkShaderModuleCreateInfo shader_info =
	{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.codeSize = file.len,
		.pCode = (void*) file.data
	};

	VkShaderModule shader_module;
	VkResult result = xr->vk.table.vkCreateShaderModule(xr->vk.device, &shader_info, NULL, &shader_module);
	hard_assert_eq(result, VK_SUCCESS);

	file_free(file);

	return shader_module;
}


private void
xr_destroy_shader(
	xr_t xr,
	VkShaderModule shader
	)
{
	assert_not_null(xr);
	assert_not_null(shader);

	xr->vk.table.vkDestroyShaderModule(xr->vk.device, shader, NULL);
}


private VkPipelineCache
xr_init_pipeline_cache(
	xr_t xr,
	const char* path
	)
{
	assert_not_null(xr);
	assert_not_null(path);

	VkPipelineCacheCreateInfo pipeline_cache_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.initialDataSize = 0,
		.pInitialData = NULL
	};

	file_t file = {0};
	VkPipelineCache pipeline_cache;

	if(file_exists(path))
	{
		bool status = file_read(path, &file);
		if(status)
		{
			pipeline_cache_info.initialDataSize = file.len;
			pipeline_cache_info.pInitialData = file.data;
		}
		else
		{
			hard_assert_log("file_read(\"%s\")", path);
		}
	}

	VkResult result = xr->vk.table.vkCreatePipelineCache(xr->vk.device, &pipeline_cache_info, NULL, &pipeline_cache);
	hard_assert_eq(result, VK_SUCCESS);

	file_free(file);

	return pipeline_cache;
}


private void
xr_free_pipeline_cache(
	xr_t xr,
	const char* path,
	VkPipelineCache pipeline_cache
	)
{
	assert_not_null(xr);

	file_t file;
	VkResult result = xr->vk.table.vkGetPipelineCacheData(xr->vk.device, pipeline_cache, &file.len, NULL);
	hard_assert_eq(result, VK_SUCCESS);

	file.data = alloc_malloc(file.len);
	assert_ptr(file.data, file.len);

	result = xr->vk.table.vkGetPipelineCacheData(xr->vk.device, pipeline_cache, &file.len, file.data);
	hard_assert_eq(result, VK_SUCCESS);

	bool status = file_write(path, file);
	if(!status)
	{
		hard_assert_log("file_write(\"%s\")", path);
	}

	file_free(file);

	xr->vk.table.vkDestroyPipelineCache(xr->vk.device, pipeline_cache, NULL);

}


private void
xr_free_sampler(
	xr_t xr,
	VkSampler sampler
	)
{
	assert_not_null(xr);

	xr->vk.table.vkDestroySampler(xr->vk.device, sampler, NULL);
}


private void
xr_init_depth_sampler(
	xr_t xr
	)
{
	assert_not_null(xr);

	VkSamplerCreateInfo sampler_info =
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

	VkResult result = xr->vk.table.vkCreateSampler(xr->vk.device, &sampler_info, NULL, &xr->vk.depth_sampler);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
xr_init_image_sampler(
	xr_t xr
	)
{
	assert_not_null(xr);

	VkSamplerCreateInfo sampler_info =
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
		.anisotropyEnable = !!xr->vk.anisotropy,
		.maxAnisotropy = xr->vk.anisotropy,
		.compareEnable = VK_FALSE,
		.compareOp = VK_COMPARE_OP_ALWAYS,
		.minLod = 0.0f,
		.maxLod = VK_LOD_CLAMP_NONE,
		.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
		.unnormalizedCoordinates = VK_FALSE
	};

	VkResult result = xr->vk.table.vkCreateSampler(
		xr->vk.device, &sampler_info, NULL, &xr->vk.image_sampler);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
xr_init_samplers(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_init_depth_sampler(xr);
	xr_init_image_sampler(xr);
}


private void
xr_free_samplers(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_sampler(xr, xr->vk.image_sampler);
	xr_free_sampler(xr, xr->vk.depth_sampler);
}


private void
xr_init_shadow_render_pass(
	xr_t xr
	)
{
	assert_not_null(xr);

	VkAttachmentDescription attachments[] =
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

	VkAttachmentReference depth_attachment =
	{
		.attachment = 0,
		.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
	};

	VkSubpassDescription subpasses[] =
	{
		{
			.flags = 0,
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount = 0,
			.pInputAttachments = NULL,
			.colorAttachmentCount = 0,
			.pColorAttachments = NULL,
			.pResolveAttachments = NULL,
			.pDepthStencilAttachment = &depth_attachment,
			.preserveAttachmentCount = 0,
			.pPreserveAttachments = NULL
		}
	};

	VkSubpassDependency subpass_dependencies[] =
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

	VkRenderPassMultiviewCreateInfo multiview_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO,
		.pNext = NULL,
		.subpassCount = 1,
		.pViewMasks = (uint32_t[]){ 0b11 },
		.dependencyCount = 0,
		.pViewOffsets = NULL,
		.correlationMaskCount = 1,
		.pCorrelationMasks = (uint32_t[]){ 0b11 }
	};

	VkRenderPassCreateInfo render_pass_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.pNext = &multiview_info,
		.flags = 0,
		.attachmentCount = MACRO_ARRAY_LEN(attachments),
		.pAttachments = attachments,
		.subpassCount = MACRO_ARRAY_LEN(subpasses),
		.pSubpasses = subpasses,
		.dependencyCount = MACRO_ARRAY_LEN(subpass_dependencies),
		.pDependencies = subpass_dependencies
	};

	VkResult result = xr->vk.table.vkCreateRenderPass(xr->vk.device,
		&render_pass_info, NULL, &xr->vk.shadow.render_pass);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
xr_free_shadow_render_pass(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr->vk.table.vkDestroyRenderPass(xr->vk.device, xr->vk.shadow.render_pass, NULL);
}


private void
xr_init_shadow_pass(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_init_shadow_render_pass(xr);
}


private void
xr_free_shadow_pass(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_shadow_render_pass(xr);
}


private void
xr_init_shadow_pipeline(
	xr_t xr
	)
{
	assert_not_null(xr);

	VkPipelineShaderStageCreateInfo shader_stages[] =
	{
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = xr_create_shader(xr, "shaders/xr/shadow.vert.spv"),
			.pName = "main",
			.pSpecializationInfo = NULL
		}
	};

	VkPipelineDynamicStateCreateInfo dynamic_state_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.dynamicStateCount = 0,
		.pDynamicStates = NULL
	};

	VkVertexInputBindingDescription vertex_bindings[] =
	{
		{
			.binding = 0,
			.stride = sizeof(vk_shadow_vertex_data_t),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
		},
		{
			.binding = 1,
			.stride = sizeof(vk_model_instance_data_t),
			.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE
		}
	};

	VkVertexInputAttributeDescription vertex_attributes[] =
	{
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(vk_shadow_vertex_data_t, position)
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

	VkPipelineVertexInputStateCreateInfo vertex_input_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.vertexBindingDescriptionCount = MACRO_ARRAY_LEN(vertex_bindings),
		.pVertexBindingDescriptions = vertex_bindings,
		.vertexAttributeDescriptionCount = MACRO_ARRAY_LEN(vertex_attributes),
		.pVertexAttributeDescriptions = vertex_attributes
	};

	VkPipelineInputAssemblyStateCreateInfo input_assembly_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE
	};

	VkPipelineViewportStateCreateInfo viewport_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.viewportCount = 1,
		.pViewports = &xr->vk.shadow_extent.viewport,
		.scissorCount = 1,
		.pScissors = &xr->vk.shadow_extent.scissor
	};

	VkPipelineRasterizationStateCreateInfo rasterization_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_FRONT_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.depthBiasConstantFactor = 0.0f,
		.depthBiasClamp = 0.0f,
		.depthBiasSlopeFactor = 0.0f,
		.lineWidth = 1.0f
	};

	VkPipelineMultisampleStateCreateInfo multisample_info =
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

	VkPipelineDepthStencilStateCreateInfo depth_stencil_info =
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

	VkPipelineColorBlendStateCreateInfo color_blend_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.logicOpEnable = VK_FALSE,
		.logicOp = VK_LOGIC_OP_CLEAR,
		.attachmentCount = 1,
		.pAttachments = &xr_vk_no_blending_attachment,
		.blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f }
	};

	VkPushConstantRange push_constants[] =
	{
		{
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
			.offset = 0,
			.size = sizeof(vk_shadow_vert_constant_data_t)
		}
	};

	VkPipelineLayoutCreateInfo pipeline_layout_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.setLayoutCount = 0,
		.pSetLayouts = NULL,
		.pushConstantRangeCount = MACRO_ARRAY_LEN(push_constants),
		.pPushConstantRanges = push_constants
	};

	VkResult result = xr->vk.table.vkCreatePipelineLayout(xr->vk.device,
		&pipeline_layout_info, NULL, &xr->vk.shadow.pipeline_layout);
	hard_assert_eq(result, VK_SUCCESS);


	VkGraphicsPipelineCreateInfo pipeline_info =
	{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.stageCount = MACRO_ARRAY_LEN(shader_stages),
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input_info,
		.pInputAssemblyState = &input_assembly_info,
		.pTessellationState = NULL,
		.pViewportState = &viewport_info,
		.pRasterizationState = &rasterization_info,
		.pMultisampleState = &multisample_info,
		.pDepthStencilState = &depth_stencil_info,
		.pColorBlendState = &color_blend_info,
		.pDynamicState = &dynamic_state_info,
		.layout = xr->vk.shadow.pipeline_layout,
		.renderPass = xr->vk.shadow.render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	static const char* pipeline_cache_path = "cache/xr/shadow_pipeline.bin";
	VkPipelineCache pipeline_cache = xr_init_pipeline_cache(xr, pipeline_cache_path);

	result = xr->vk.table.vkCreateGraphicsPipelines(xr->vk.device,
		pipeline_cache, 1, &pipeline_info, NULL, &xr->vk.shadow.pipeline);
	hard_assert_eq(result, VK_SUCCESS);

	xr_free_pipeline_cache(xr, pipeline_cache_path, pipeline_cache);

	for(uint32_t i = 0; i < MACRO_ARRAY_LEN(shader_stages); ++i)
	{
		xr_destroy_shader(xr, shader_stages[i].module);
	}
}


private void
xr_free_shadow_pipeline(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr->vk.table.vkDestroyPipeline(xr->vk.device, xr->vk.shadow.pipeline, NULL);
	xr->vk.table.vkDestroyPipelineLayout(xr->vk.device, xr->vk.shadow.pipeline_layout, NULL);
}


private void
xr_init_shadow_consts(
	xr_t xr
	)
{
	assert_not_null(xr);
}


private void
xr_free_shadow_consts(
	xr_t xr
	)
{
	assert_not_null(xr);
}


private void
xr_init_shadow(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_init_shadow_pass(xr);
	xr_init_shadow_pipeline(xr);
	xr_init_shadow_consts(xr);
}


private void
xr_free_shadow(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_shadow_consts(xr);
	xr_free_shadow_pipeline(xr);
	xr_free_shadow_pass(xr);
}


private void
xr_init_scene_render_pass(
	xr_t xr
	)
{
	assert_not_null(xr);

	VkAttachmentDescription attachments[] =
	{
		{
			.flags = 0,
			.format = VK_FORMAT_D32_SFLOAT,
			.samples = xr->vk.samples,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
		},
		{
			.flags = 0,
			.format = VK_FORMAT_R32G32B32A32_SFLOAT,
			.samples = xr->vk.samples,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		},
		{
			.flags = 0,
			.format = VK_FORMAT_R8G8B8A8_UNORM,
			.samples = xr->vk.samples,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		},
		{
			.flags = 0,
			.format = xr->vk.format,
			.samples = xr->vk.samples,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		},
		{
			.flags = 0,
			.format = VK_FORMAT_R32G32B32A32_SFLOAT,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		},
		{
			.flags = 0,
			.format = VK_FORMAT_R8G8B8A8_UNORM,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		},
		{
			.flags = 0,
			.format = xr->vk.format,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		}
	};

	VkAttachmentReference color_attachments[] =
	{
		{
			.attachment = 1,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		},
		{
			.attachment = 2,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		},
		{
			.attachment = 3,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		}
	};

	VkAttachmentReference depth_attachment =
	{
		.attachment = 0,
		.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
	};

	VkAttachmentReference resolve_attachments[] =
	{
		{
			.attachment = 4,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		},
		{
			.attachment = 5,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		},
		{
			.attachment = 6,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		}
	};

	VkSubpassDescription subpasses[] =
	{
		{
			.flags = 0,
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount = 0,
			.pInputAttachments = NULL,
			.colorAttachmentCount = MACRO_ARRAY_LEN(color_attachments),
			.pColorAttachments = color_attachments,
			.pResolveAttachments = resolve_attachments,
			.pDepthStencilAttachment = &depth_attachment,
			.preserveAttachmentCount = 0,
			.pPreserveAttachments = NULL
		}
	};

	VkSubpassDependency subpass_dependencies[] =
	{
		{
			.srcSubpass = VK_SUBPASS_EXTERNAL,
			.dstSubpass = 0,
			.srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
				VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			.srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
				VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT
		},
		{
			.srcSubpass = 0,
			.dstSubpass = VK_SUBPASS_EXTERNAL,
			.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
				VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
				VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
			.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT
		}
	};

	VkRenderPassMultiviewCreateInfo multiview_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO,
		.pNext = NULL,
		.subpassCount = 1,
		.pViewMasks = (uint32_t[]){ 0b11 },
		.dependencyCount = 0,
		.pViewOffsets = NULL,
		.correlationMaskCount = 1,
		.pCorrelationMasks = (uint32_t[]){ 0b11 }
	};

	VkRenderPassCreateInfo render_pass_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.pNext = &multiview_info,
		.flags = 0,
		.attachmentCount = MACRO_ARRAY_LEN(attachments),
		.pAttachments = attachments,
		.subpassCount = MACRO_ARRAY_LEN(subpasses),
		.pSubpasses = subpasses,
		.dependencyCount = MACRO_ARRAY_LEN(subpass_dependencies),
		.pDependencies = subpass_dependencies
	};

	VkResult result = xr->vk.table.vkCreateRenderPass(xr->vk.device,
		&render_pass_info, NULL, &xr->vk.scene.render_pass);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
xr_free_scene_render_pass(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr->vk.table.vkDestroyRenderPass(xr->vk.device, xr->vk.scene.render_pass, NULL);
}


private void
xr_init_scene_pass(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_init_scene_render_pass(xr);
}


private void
xr_free_scene_pass(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_scene_render_pass(xr);
}


private void
xr_init_scene_pipeline(
	xr_t xr
	)
{
	assert_not_null(xr);

	typedef struct vk_scene_frag_specialization
	{
		int32_t enable_depth_shadows;
		int32_t enable_backface_shadows;
		int32_t enable_specular;
		float shadow_value;
		float lambert_start_angle;
	}
	vk_scene_frag_specialization_t;

	vk_scene_frag_specialization_t frag_specialization_data =
	{
		.enable_depth_shadows = xr->options.enable_depth_shadows,
		.enable_backface_shadows = xr->options.enable_backface_shadows,
		.enable_specular = xr->options.enable_specular,
		.shadow_value = xr->options.shadow_value,
		.lambert_start_angle = xr->options.lambert_start_angle
	};

	VkSpecializationMapEntry frag_map_entries[] =
	{
		{
			.constantID = 0,
			.offset = offsetof(vk_scene_frag_specialization_t, enable_depth_shadows),
			.size = sizeof(frag_specialization_data.enable_depth_shadows)
		},
		{
			.constantID = 1,
			.offset = offsetof(vk_scene_frag_specialization_t, enable_backface_shadows),
			.size = sizeof(frag_specialization_data.enable_backface_shadows)
		},
		{
			.constantID = 2,
			.offset = offsetof(vk_scene_frag_specialization_t, enable_specular),
			.size = sizeof(frag_specialization_data.enable_specular)
		},
		{
			.constantID = 3,
			.offset = offsetof(vk_scene_frag_specialization_t, shadow_value),
			.size = sizeof(frag_specialization_data.shadow_value)
		},
		{
			.constantID = 4,
			.offset = offsetof(vk_scene_frag_specialization_t, lambert_start_angle),
			.size = sizeof(frag_specialization_data.lambert_start_angle)
		}
	};

	VkSpecializationInfo frag_specialization_info =
	{
		.mapEntryCount = MACRO_ARRAY_LEN(frag_map_entries),
		.pMapEntries = frag_map_entries,
		.dataSize = sizeof(frag_specialization_data),
		.pData = &frag_specialization_data
	};

	VkPipelineShaderStageCreateInfo shader_stages[] =
	{
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = xr_create_shader(xr, "shaders/xr/scene.vert.spv"),
			.pName = "main",
			.pSpecializationInfo = NULL
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = xr_create_shader(xr, "shaders/xr/scene.frag.spv"),
			.pName = "main",
			.pSpecializationInfo = &frag_specialization_info
		}
	};

	VkDynamicState dynamic_states[] =
	{
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};

	VkPipelineDynamicStateCreateInfo dynamic_state_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.dynamicStateCount = MACRO_ARRAY_LEN(dynamic_states),
		.pDynamicStates = dynamic_states
	};

	VkVertexInputBindingDescription vertex_bindings[] =
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

	VkVertexInputAttributeDescription vertex_attributes[] =
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

	VkPipelineVertexInputStateCreateInfo vertex_input_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.vertexBindingDescriptionCount = MACRO_ARRAY_LEN(vertex_bindings),
		.pVertexBindingDescriptions = vertex_bindings,
		.vertexAttributeDescriptionCount = MACRO_ARRAY_LEN(vertex_attributes),
		.pVertexAttributeDescriptions = vertex_attributes
	};

	VkPipelineInputAssemblyStateCreateInfo input_assembly_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE
	};

	VkPipelineViewportStateCreateInfo viewport_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.viewportCount = 1,
		.pViewports = NULL,
		.scissorCount = 1,
		.pScissors = NULL
	};

	VkPipelineRasterizationStateCreateInfo rasterization_info =
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

	VkPipelineMultisampleStateCreateInfo multisample_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.rasterizationSamples = xr->vk.samples,
		.sampleShadingEnable = xr->options.sample_shading,
		.minSampleShading = xr->options.min_sample_shading,
		.pSampleMask = NULL,
		.alphaToCoverageEnable = VK_FALSE,
		.alphaToOneEnable = VK_FALSE
	};

	VkPipelineDepthStencilStateCreateInfo depth_stencil_info =
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

	VkPipelineColorBlendAttachmentState color_blend_attachments[3];
	VkPipelineColorBlendAttachmentState* color_blend_attachment = color_blend_attachments;
	VkPipelineColorBlendAttachmentState* color_blend_attachment_end =
		color_blend_attachments + MACRO_ARRAY_LEN(color_blend_attachments);

	while(color_blend_attachment < color_blend_attachment_end)
	{
		*color_blend_attachment = xr_vk_no_blending_attachment;
		++color_blend_attachment;
	}

	VkPipelineColorBlendStateCreateInfo color_blend_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.logicOpEnable = VK_FALSE,
		.logicOp = VK_LOGIC_OP_CLEAR,
		.attachmentCount = MACRO_ARRAY_LEN(color_blend_attachments),
		.pAttachments = color_blend_attachments,
		.blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f }
	};

	VkDescriptorSetLayout set_layouts[] =
	{
		xr->vk.vert_ubo_set_layout.layout,
		xr->vk.sampler_set_layout.layout,
		xr->vk.sampler_set_layout.layout
	};

	VkPushConstantRange push_constants[] =
	{
		{
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
			.offset = 0,
			.size = sizeof(vk_scene_frag_constant_data_t)
		}
	};

	VkPipelineLayoutCreateInfo pipeline_layout_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.setLayoutCount = MACRO_ARRAY_LEN(set_layouts),
		.pSetLayouts = set_layouts,
		.pushConstantRangeCount = MACRO_ARRAY_LEN(push_constants),
		.pPushConstantRanges = push_constants
	};

	VkResult result = xr->vk.table.vkCreatePipelineLayout(xr->vk.device,
		&pipeline_layout_info, NULL, &xr->vk.scene.pipeline_layout);
	hard_assert_eq(result, VK_SUCCESS);


	VkGraphicsPipelineCreateInfo pipeline_info =
	{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.stageCount = MACRO_ARRAY_LEN(shader_stages),
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input_info,
		.pInputAssemblyState = &input_assembly_info,
		.pTessellationState = NULL,
		.pViewportState = &viewport_info,
		.pRasterizationState = &rasterization_info,
		.pMultisampleState = &multisample_info,
		.pDepthStencilState = &depth_stencil_info,
		.pColorBlendState = &color_blend_info,
		.pDynamicState = &dynamic_state_info,
		.layout = xr->vk.scene.pipeline_layout,
		.renderPass = xr->vk.scene.render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	static const char* pipeline_cache_path = "cache/xr/scene_pipeline.bin";
	VkPipelineCache pipeline_cache = xr_init_pipeline_cache(xr, pipeline_cache_path);

	result = xr->vk.table.vkCreateGraphicsPipelines(xr->vk.device,
		pipeline_cache, 1, &pipeline_info, NULL, &xr->vk.scene.pipeline);
	hard_assert_eq(result, VK_SUCCESS);

	xr_free_pipeline_cache(xr, pipeline_cache_path, pipeline_cache);

	for(uint32_t i = 0; i < MACRO_ARRAY_LEN(shader_stages); ++i)
	{
		xr_destroy_shader(xr, shader_stages[i].module);
	}
}


private void
xr_free_scene_pipeline(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr->vk.table.vkDestroyPipeline(xr->vk.device, xr->vk.scene.pipeline, NULL);
	xr->vk.table.vkDestroyPipelineLayout(xr->vk.device, xr->vk.scene.pipeline_layout, NULL);
}


private void
xr_init_scene_consts(
	xr_t xr
	)
{
	assert_not_null(xr);
}


private void
xr_free_scene_consts(
	xr_t xr
	)
{
	assert_not_null(xr);
}


private void
xr_init_scene(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_init_scene_pass(xr);
	xr_init_scene_pipeline(xr);
	xr_init_scene_consts(xr);
}


private void
xr_free_scene(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_scene_consts(xr);
	xr_free_scene_pipeline(xr);
	xr_free_scene_pass(xr);
}


private void
xr_init_ssao_render_pass(
	xr_t xr
	)
{
	assert_not_null(xr);

	VkAttachmentDescription attachments[] =
	{
		{
			.flags = 0,
			.format = VK_FORMAT_R8_UNORM,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		}
	};

	VkAttachmentReference color_attachments[] =
	{
		{
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		}
	};

	VkSubpassDescription subpasses[] =
	{
		{
			.flags = 0,
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount = 0,
			.pInputAttachments = NULL,
			.colorAttachmentCount = MACRO_ARRAY_LEN(color_attachments),
			.pColorAttachments = color_attachments,
			.pResolveAttachments = NULL,
			.pDepthStencilAttachment = NULL,
			.preserveAttachmentCount = 0,
			.pPreserveAttachments = NULL
		}
	};

	VkSubpassDependency subpass_dependencies[] =
	{
		{
			.srcSubpass = VK_SUBPASS_EXTERNAL,
			.dstSubpass = 0,
			.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT
		},
		{
			.srcSubpass = 0,
			.dstSubpass = VK_SUBPASS_EXTERNAL,
			.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
			.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT
		}
	};

	VkRenderPassMultiviewCreateInfo multiview_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO,
		.pNext = NULL,
		.subpassCount = 1,
		.pViewMasks = (uint32_t[]){ 0b11 },
		.dependencyCount = 0,
		.pViewOffsets = NULL,
		.correlationMaskCount = 1,
		.pCorrelationMasks = (uint32_t[]){ 0b11 }
	};

	VkRenderPassCreateInfo render_pass_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.pNext = &multiview_info,
		.flags = 0,
		.attachmentCount = MACRO_ARRAY_LEN(attachments),
		.pAttachments = attachments,
		.subpassCount = MACRO_ARRAY_LEN(subpasses),
		.pSubpasses = subpasses,
		.dependencyCount = MACRO_ARRAY_LEN(subpass_dependencies),
		.pDependencies = subpass_dependencies
	};

	VkResult result = xr->vk.table.vkCreateRenderPass(xr->vk.device,
		&render_pass_info, NULL, &xr->vk.ssao.render_pass);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
xr_free_ssao_render_pass(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr->vk.table.vkDestroyRenderPass(xr->vk.device, xr->vk.ssao.render_pass, NULL);
}


private void
xr_init_ssao_pass(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_init_ssao_render_pass(xr);
}


private void
xr_free_ssao_pass(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_ssao_render_pass(xr);
}


private void
xr_init_ssao_pipeline(
	xr_t xr
	)
{
	assert_not_null(xr);

	typedef struct xr_ssao_frag_specialization
	{
		int32_t enable_ssao;
		int32_t ssao_kernel_size;
		int32_t ssao_noise_size;
		float ssao_radius;
		float ssao_bias;
		float ssao_power;
		float ssao_range_check;
		float ssao_depth_k;
		float ssao_depth_gamma;
		int32_t ssao_debug;
	}
	xr_ssao_frag_specialization_t;

	xr_ssao_frag_specialization_t frag_specialization_data =
	{
		.enable_ssao = xr->options.enable_ssao,
		.ssao_kernel_size = xr->options.ssao_kernel_size,
		.ssao_noise_size = xr->options.ssao_noise_size,
		.ssao_radius = xr->options.ssao_radius,
		.ssao_bias = xr->options.ssao_bias,
		.ssao_power = xr->options.ssao_power,
		.ssao_range_check = xr->options.ssao_range_check,
		.ssao_depth_k = xr->options.ssao_depth_k,
		.ssao_depth_gamma = xr->options.ssao_depth_gamma,
		.ssao_debug = xr->options.ssao_debug
	};

	VkSpecializationMapEntry frag_map_entries[] =
	{
		{
			.constantID = 0,
			.offset = offsetof(xr_ssao_frag_specialization_t, enable_ssao),
			.size = sizeof(frag_specialization_data.enable_ssao)
		},
		{
			.constantID = 1,
			.offset = offsetof(xr_ssao_frag_specialization_t, ssao_kernel_size),
			.size = sizeof(frag_specialization_data.ssao_kernel_size)
		},
		{
			.constantID = 2,
			.offset = offsetof(xr_ssao_frag_specialization_t, ssao_noise_size),
			.size = sizeof(frag_specialization_data.ssao_noise_size)
		},
		{
			.constantID = 3,
			.offset = offsetof(xr_ssao_frag_specialization_t, ssao_radius),
			.size = sizeof(frag_specialization_data.ssao_radius)
		},
		{
			.constantID = 4,
			.offset = offsetof(xr_ssao_frag_specialization_t, ssao_bias),
			.size = sizeof(frag_specialization_data.ssao_bias)
		},
		{
			.constantID = 5,
			.offset = offsetof(xr_ssao_frag_specialization_t, ssao_power),
			.size = sizeof(frag_specialization_data.ssao_power)
		},
		{
			.constantID = 6,
			.offset = offsetof(xr_ssao_frag_specialization_t, ssao_range_check),
			.size = sizeof(frag_specialization_data.ssao_range_check)
		},
		{
			.constantID = 7,
			.offset = offsetof(xr_ssao_frag_specialization_t, ssao_depth_k),
			.size = sizeof(frag_specialization_data.ssao_depth_k)
		},
		{
			.constantID = 8,
			.offset = offsetof(xr_ssao_frag_specialization_t, ssao_depth_gamma),
			.size = sizeof(frag_specialization_data.ssao_depth_gamma)
		},
		{
			.constantID = 9,
			.offset = offsetof(xr_ssao_frag_specialization_t, ssao_debug),
			.size = sizeof(frag_specialization_data.ssao_debug)
		}
	};

	VkSpecializationInfo frag_specialization_info =
	{
		.mapEntryCount = MACRO_ARRAY_LEN(frag_map_entries),
		.pMapEntries = frag_map_entries,
		.dataSize = sizeof(frag_specialization_data),
		.pData = &frag_specialization_data
	};

	VkPipelineShaderStageCreateInfo shader_stages[] =
	{
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = xr_create_shader(xr, "shaders/xr/fullscreen.vert.spv"),
			.pName = "main",
			.pSpecializationInfo = NULL
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = xr_create_shader(xr, "shaders/xr/ssao.frag.spv"),
			.pName = "main",
			.pSpecializationInfo = &frag_specialization_info
		}
	};

	VkDynamicState dynamic_states[] =
	{
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};

	VkPipelineDynamicStateCreateInfo dynamic_state_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.dynamicStateCount = MACRO_ARRAY_LEN(dynamic_states),
		.pDynamicStates = dynamic_states
	};

	VkPipelineVertexInputStateCreateInfo vertex_input_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.vertexBindingDescriptionCount = 0,
		.pVertexBindingDescriptions = NULL,
		.vertexAttributeDescriptionCount = 0,
		.pVertexAttributeDescriptions = NULL
	};

	VkPipelineInputAssemblyStateCreateInfo input_assembly_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE
	};

	VkPipelineViewportStateCreateInfo viewport_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.viewportCount = 1,
		.pViewports = NULL,
		.scissorCount = 1,
		.pScissors = NULL
	};

	VkPipelineRasterizationStateCreateInfo rasterization_info =
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

	VkPipelineMultisampleStateCreateInfo multisample_info =
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

	VkPipelineDepthStencilStateCreateInfo depth_stencil_info =
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
		.maxDepthBounds = 0.0f
	};

	VkPipelineColorBlendStateCreateInfo color_blend_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.logicOpEnable = VK_FALSE,
		.logicOp = VK_LOGIC_OP_CLEAR,
		.attachmentCount = 1,
		.pAttachments = &xr_vk_no_blending_attachment,
		.blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f }
	};

	VkDescriptorSetLayout set_layouts[] =
	{
		xr->vk.scene_set_layout.layout,
		xr->vk.sampler_set_layout.layout,
		xr->vk.frag_ubo_set_layout.layout,
		xr->vk.frag_ubo_set_layout.layout
	};

	VkPipelineLayoutCreateInfo pipeline_layout_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.setLayoutCount = MACRO_ARRAY_LEN(set_layouts),
		.pSetLayouts = set_layouts,
		.pushConstantRangeCount = 0,
		.pPushConstantRanges = NULL
	};

	VkResult result = xr->vk.table.vkCreatePipelineLayout(xr->vk.device,
		&pipeline_layout_info, NULL, &xr->vk.ssao.pipeline_layout);
	hard_assert_eq(result, VK_SUCCESS);


	VkGraphicsPipelineCreateInfo pipeline_info =
	{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.stageCount = MACRO_ARRAY_LEN(shader_stages),
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input_info,
		.pInputAssemblyState = &input_assembly_info,
		.pTessellationState = NULL,
		.pViewportState = &viewport_info,
		.pRasterizationState = &rasterization_info,
		.pMultisampleState = &multisample_info,
		.pDepthStencilState = &depth_stencil_info,
		.pColorBlendState = &color_blend_info,
		.pDynamicState = &dynamic_state_info,
		.layout = xr->vk.ssao.pipeline_layout,
		.renderPass = xr->vk.ssao.render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	static const char* pipeline_cache_path = "cache/xr/ssao_pipeline.bin";
	VkPipelineCache pipeline_cache = xr_init_pipeline_cache(xr, pipeline_cache_path);

	result = xr->vk.table.vkCreateGraphicsPipelines(xr->vk.device,
		pipeline_cache, 1, &pipeline_info, NULL, &xr->vk.ssao.pipeline);
	hard_assert_eq(result, VK_SUCCESS);

	xr_free_pipeline_cache(xr, pipeline_cache_path, pipeline_cache);

	for(uint32_t i = 0; i < MACRO_ARRAY_LEN(shader_stages); ++i)
	{
		xr_destroy_shader(xr, shader_stages[i].module);
	}
}


private void
xr_free_ssao_pipeline(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr->vk.table.vkDestroyPipeline(xr->vk.device, xr->vk.ssao.pipeline, NULL);
	xr->vk.table.vkDestroyPipelineLayout(xr->vk.device, xr->vk.ssao.pipeline_layout, NULL);
}


private void
xr_init_ssao_noise_const(
	xr_t xr
	)
{
	assert_not_null(xr);

	vk_image_t* image = &xr->vk.ssao.noise.image;
	*image =
	(vk_image_t)
	{
		.path = NULL,
		.type = VK_IMAGE_TYPE_TEXTURE_2D_BITS | VK_IMAGE_TYPE_CUSTOM_FORMAT_BIT | VK_IMAGE_TYPE_CUSTOM_SIZE_BIT,
		.format = VK_FORMAT_R32G32B32A32_SFLOAT,
		.width = xr->options.ssao_noise_size,
		.height = xr->options.ssao_noise_size
	};
	xr_init_frame_image(xr, &xr->vk.ssao.noise);

	uint32_t size = xr->options.ssao_noise_size * xr->options.ssao_noise_size;
	vec4 data[size];
	vec4* noise = data;
	vec4* noise_end = noise + size;

	while(noise != noise_end)
	{
		(*noise)[0] = drand48();
		(*noise)[1] = drand48();
		(*noise)[2] = drand48();

		if(glm_vec3_norm(*noise) > 0.01f)
		{
			glm_vec3_normalize(*noise);
		}
		else
		{
			continue;
		}

		++noise;
	}

	image->data = data;
	image->size = sizeof(*data) * size;
	xr_copy_data_to_image(xr, image);
}


private void
xr_free_ssao_noise_const(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_frame_image(xr, &xr->vk.ssao.noise);
}


private void
xr_init_ssao_kernel_const(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_init_frag_ubo_buffer(xr, sizeof(vec4) * xr->options.ssao_kernel_size, &xr->vk.ssao.kernel_ubo);

	uint32_t size = xr->options.ssao_kernel_size;
	vec4 data[size];
	vec4* kernel = data;
	vec4* kernel_end = kernel + size;

	while(kernel != kernel_end)
	{
		vec3 sample;
		sample[0] = drand48() * 2.0f - 1.0f;
		sample[1] = drand48() * 2.0f - 1.0f;
		sample[2] = drand48();

		if(glm_vec3_norm(sample) > 0.01f)
		{
			glm_vec3_normalize(sample);
		}
		else
		{
			continue;
		}

		float scale = (float)(kernel - data) / size;
		scale = glm_lerp(0.1f, 1.0f, scale * scale);

		sample[0] *= scale;
		sample[1] *= scale;
		sample[2] *= scale;

		(*kernel)[0] = sample[0];
		(*kernel)[1] = sample[1];
		(*kernel)[2] = sample[2];
		(*kernel)[3] = 0.0f;

		++kernel;
	}

	xr_copy_to_buffer(xr, &xr->vk.ssao.kernel_ubo.buffer, data, sizeof(*data) * size);
}


private void
xr_free_ssao_kernel_const(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_ubo_buffer(xr, &xr->vk.ssao.kernel_ubo);
}


private void
xr_init_ssao_consts(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_init_ssao_noise_const(xr);
	xr_init_ssao_kernel_const(xr);
}


private void
xr_free_ssao_consts(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_ssao_kernel_const(xr);
	xr_free_ssao_noise_const(xr);
}


private void
xr_init_ssao(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_init_ssao_pass(xr);
	xr_init_ssao_pipeline(xr);
	xr_init_ssao_consts(xr);
}


private void
xr_free_ssao(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_ssao_consts(xr);
	xr_free_ssao_pipeline(xr);
	xr_free_ssao_pass(xr);
}


private void
xr_init_ssao_blur_render_pass(
	xr_t xr
	)
{
	assert_not_null(xr);

	assert_not_null(xr);

	VkAttachmentDescription attachments[] =
	{
		{
			.flags = 0,
			.format = VK_FORMAT_R8_UNORM,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
		}
	};

	VkAttachmentReference color_attachments[] =
	{
		{
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		}
	};

	VkSubpassDescription subpasses[] =
	{
		{
			.flags = 0,
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount = 0,
			.pInputAttachments = NULL,
			.colorAttachmentCount = MACRO_ARRAY_LEN(color_attachments),
			.pColorAttachments = color_attachments,
			.pResolveAttachments = NULL,
			.pDepthStencilAttachment = NULL,
			.preserveAttachmentCount = 0,
			.pPreserveAttachments = NULL
		}
	};

	VkSubpassDependency subpass_dependencies[] =
	{
		{
			.srcSubpass = VK_SUBPASS_EXTERNAL,
			.dstSubpass = 0,
			.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT
		},
		{
			.srcSubpass = 0,
			.dstSubpass = VK_SUBPASS_EXTERNAL,
			.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
			.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT
		}
	};

	VkRenderPassMultiviewCreateInfo multiview_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO,
		.pNext = NULL,
		.subpassCount = 1,
		.pViewMasks = (uint32_t[]){ 0b11 },
		.dependencyCount = 0,
		.pViewOffsets = NULL,
		.correlationMaskCount = 1,
		.pCorrelationMasks = (uint32_t[]){ 0b11 }
	};

	VkRenderPassCreateInfo render_pass_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.pNext = &multiview_info,
		.flags = 0,
		.attachmentCount = MACRO_ARRAY_LEN(attachments),
		.pAttachments = attachments,
		.subpassCount = MACRO_ARRAY_LEN(subpasses),
		.pSubpasses = subpasses,
		.dependencyCount = MACRO_ARRAY_LEN(subpass_dependencies),
		.pDependencies = subpass_dependencies
	};

	VkResult result = xr->vk.table.vkCreateRenderPass(xr->vk.device,
		&render_pass_info, NULL, &xr->vk.ssao_blur.render_pass);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
xr_free_ssao_blur_render_pass(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr->vk.table.vkDestroyRenderPass(xr->vk.device, xr->vk.ssao_blur.render_pass, NULL);
}


private void
xr_init_ssao_blur_pass(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_init_ssao_blur_render_pass(xr);
}


private void
xr_free_ssao_blur_pass(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_ssao_blur_render_pass(xr);
}


private void
xr_init_ssao_blur_pipeline(
	xr_t xr
	)
{
	assert_not_null(xr);

	typedef struct xr_ssao_blur_frag_specialization
	{
		int32_t enable_ssao;
		float ssao_blur_radius;
		float ssao_blur_falloff;
		float ssao_blur_depth_tolerance;
	}
	xr_ssao_blur_frag_specialization_t;

	xr_ssao_blur_frag_specialization_t frag_specialization_data =
	{
		.enable_ssao = xr->options.enable_ssao,
		.ssao_blur_radius = xr->options.ssao_blur_radius,
		.ssao_blur_falloff = xr->options.ssao_blur_falloff,
		.ssao_blur_depth_tolerance = xr->options.ssao_blur_depth_tolerance
	};

	VkSpecializationMapEntry frag_map_entries[] =
	{
		{
			.constantID = 0,
			.offset = offsetof(xr_ssao_blur_frag_specialization_t, enable_ssao),
			.size = sizeof(frag_specialization_data.enable_ssao)
		},
		{
			.constantID = 1,
			.offset = offsetof(xr_ssao_blur_frag_specialization_t, ssao_blur_radius),
			.size = sizeof(frag_specialization_data.ssao_blur_radius)
		},
		{
			.constantID = 2,
			.offset = offsetof(xr_ssao_blur_frag_specialization_t, ssao_blur_falloff),
			.size = sizeof(frag_specialization_data.ssao_blur_falloff)
		},
		{
			.constantID = 3,
			.offset = offsetof(xr_ssao_blur_frag_specialization_t, ssao_blur_depth_tolerance),
			.size = sizeof(frag_specialization_data.ssao_blur_depth_tolerance)
		}
	};

	VkSpecializationInfo frag_specialization_info =
	{
		.mapEntryCount = MACRO_ARRAY_LEN(frag_map_entries),
		.pMapEntries = frag_map_entries,
		.dataSize = sizeof(frag_specialization_data),
		.pData = &frag_specialization_data
	};

	VkPipelineShaderStageCreateInfo shader_stages[] =
	{
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = xr_create_shader(xr, "shaders/xr/fullscreen.vert.spv"),
			.pName = "main",
			.pSpecializationInfo = NULL
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = xr_create_shader(xr, "shaders/xr/ssao_blur.frag.spv"),
			.pName = "main",
			.pSpecializationInfo = &frag_specialization_info
		}
	};

	VkDynamicState dynamic_states[] =
	{
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};

	VkPipelineDynamicStateCreateInfo dynamic_state_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.dynamicStateCount = MACRO_ARRAY_LEN(dynamic_states),
		.pDynamicStates = dynamic_states
	};

	VkPipelineVertexInputStateCreateInfo vertex_input_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.vertexBindingDescriptionCount = 0,
		.pVertexBindingDescriptions = NULL,
		.vertexAttributeDescriptionCount = 0,
		.pVertexAttributeDescriptions = NULL
	};

	VkPipelineInputAssemblyStateCreateInfo input_assembly_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE
	};

	VkPipelineViewportStateCreateInfo viewport_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.viewportCount = 1,
		.pViewports = NULL,
		.scissorCount = 1,
		.pScissors = NULL
	};

	VkPipelineRasterizationStateCreateInfo rasterization_info =
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

	VkPipelineMultisampleStateCreateInfo multisample_info =
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

	VkPipelineDepthStencilStateCreateInfo depth_stencil_info =
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
		.maxDepthBounds = 0.0f
	};

	VkPipelineColorBlendStateCreateInfo color_blend_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.logicOpEnable = VK_FALSE,
		.logicOp = VK_LOGIC_OP_CLEAR,
		.attachmentCount = 1,
		.pAttachments = &xr_vk_no_blending_attachment,
		.blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f }
	};

	VkDescriptorSetLayout set_layouts[] =
	{
		xr->vk.scene_set_layout.layout,
		xr->vk.sampler_set_layout.layout
	};

	VkPipelineLayoutCreateInfo pipeline_layout_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.setLayoutCount = MACRO_ARRAY_LEN(set_layouts),
		.pSetLayouts = set_layouts,
		.pushConstantRangeCount = 0,
		.pPushConstantRanges = NULL
	};

	VkResult result = xr->vk.table.vkCreatePipelineLayout(xr->vk.device,
		&pipeline_layout_info, NULL, &xr->vk.ssao_blur.pipeline_layout);
	hard_assert_eq(result, VK_SUCCESS);


	VkGraphicsPipelineCreateInfo pipeline_info =
	{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.stageCount = MACRO_ARRAY_LEN(shader_stages),
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input_info,
		.pInputAssemblyState = &input_assembly_info,
		.pTessellationState = NULL,
		.pViewportState = &viewport_info,
		.pRasterizationState = &rasterization_info,
		.pMultisampleState = &multisample_info,
		.pDepthStencilState = &depth_stencil_info,
		.pColorBlendState = &color_blend_info,
		.pDynamicState = &dynamic_state_info,
		.layout = xr->vk.ssao_blur.pipeline_layout,
		.renderPass = xr->vk.ssao_blur.render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	const char* pipeline_cache_path = "cache/xr/ssao_blur_pipeline.bin";
	VkPipelineCache pipeline_cache = xr_init_pipeline_cache(xr, pipeline_cache_path);

	result = xr->vk.table.vkCreateGraphicsPipelines(
		xr->vk.device, pipeline_cache, 1, &pipeline_info, NULL, &xr->vk.ssao_blur.pipeline);
	hard_assert_eq(result, VK_SUCCESS);

	xr_free_pipeline_cache(xr, pipeline_cache_path, pipeline_cache);

	for(uint32_t i = 0; i < MACRO_ARRAY_LEN(shader_stages); ++i)
	{
		xr_destroy_shader(xr, shader_stages[i].module);
	}
}


private void
xr_free_ssao_blur_pipeline(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr->vk.table.vkDestroyPipeline(xr->vk.device, xr->vk.ssao_blur.pipeline, NULL);
	xr->vk.table.vkDestroyPipelineLayout(xr->vk.device, xr->vk.ssao_blur.pipeline_layout, NULL);
}


private void
xr_init_ssao_blur_consts(
	xr_t xr
	)
{
	assert_not_null(xr);
}


private void
xr_free_ssao_blur_consts(
	xr_t xr
	)
{
	assert_not_null(xr);
}


private void
xr_init_ssao_blur(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_init_ssao_blur_pass(xr);
	xr_init_ssao_blur_pipeline(xr);
	xr_init_ssao_blur_consts(xr);
}


private void
xr_free_ssao_blur(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_ssao_blur_consts(xr);
	xr_free_ssao_blur_pipeline(xr);
	xr_free_ssao_blur_pass(xr);
}


private void
xr_init_output_render_pass(
	xr_t xr
	)
{
	assert_not_null(xr);

	VkAttachmentDescription attachments[] =
	{
		{
			.flags = 0,
			.format = xr->vk.format,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
		}
	};

	VkAttachmentReference color_attachments[] =
	{
		{
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		}
	};

	VkSubpassDescription subpasses[] =
	{
		{
			.flags = 0,
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount = 0,
			.pInputAttachments = NULL,
			.colorAttachmentCount = MACRO_ARRAY_LEN(color_attachments),
			.pColorAttachments = color_attachments,
			.pResolveAttachments = NULL,
			.pDepthStencilAttachment = NULL,
			.preserveAttachmentCount = 0,
			.pPreserveAttachments = NULL
		}
	};

	VkSubpassDependency subpass_dependencies[] =
	{
		{
			.srcSubpass = VK_SUBPASS_EXTERNAL,
			.dstSubpass = 0,
			.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT
		},
		{
			.srcSubpass = 0,
			.dstSubpass = VK_SUBPASS_EXTERNAL,
			.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
			.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT
		}
	};

	VkRenderPassMultiviewCreateInfo multiview_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO,
		.pNext = NULL,
		.subpassCount = 1,
		.pViewMasks = (uint32_t[]){ 0b11 },
		.dependencyCount = 0,
		.pViewOffsets = NULL,
		.correlationMaskCount = 1,
		.pCorrelationMasks = (uint32_t[]){ 0b11 }
	};

	VkRenderPassCreateInfo render_pass_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.pNext = &multiview_info,
		.flags = 0,
		.attachmentCount = MACRO_ARRAY_LEN(attachments),
		.pAttachments = attachments,
		.subpassCount = MACRO_ARRAY_LEN(subpasses),
		.pSubpasses = subpasses,
		.dependencyCount = MACRO_ARRAY_LEN(subpass_dependencies),
		.pDependencies = subpass_dependencies
	};

	VkResult result = xr->vk.table.vkCreateRenderPass(xr->vk.device,
		&render_pass_info, NULL, &xr->vk.output.render_pass);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
xr_free_output_render_pass(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr->vk.table.vkDestroyRenderPass(xr->vk.device, xr->vk.output.render_pass, NULL);
}


private void
xr_init_output_pass(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_init_output_render_pass(xr);
}


private void
xr_free_output_pass(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_output_render_pass(xr);
}


private void
xr_init_skybox_pipeline(
	xr_t xr
	)
{
	assert_not_null(xr);

	VkPipelineShaderStageCreateInfo shader_stages[] =
	{
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = xr_create_shader(xr, "shaders/xr/skybox.vert.spv"),
			.pName = "main",
			.pSpecializationInfo = NULL
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = xr_create_shader(xr, "shaders/xr/skybox.frag.spv"),
			.pName = "main",
			.pSpecializationInfo = NULL
		}
	};

	VkDynamicState dynamic_states[] =
	{
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};

	VkPipelineDynamicStateCreateInfo dynamic_state_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.dynamicStateCount = MACRO_ARRAY_LEN(dynamic_states),
		.pDynamicStates = dynamic_states
	};

	VkVertexInputBindingDescription vertex_bindings[] =
	{
		{
			.binding = 0,
			.stride = sizeof(vk_skybox_vertex_data_t),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
		}
	};

	VkVertexInputAttributeDescription vertex_attributes[] =
	{
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(vk_skybox_vertex_data_t, position)
		}
	};

	VkPipelineVertexInputStateCreateInfo vertex_input_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.vertexBindingDescriptionCount = MACRO_ARRAY_LEN(vertex_bindings),
		.pVertexBindingDescriptions = vertex_bindings,
		.vertexAttributeDescriptionCount = MACRO_ARRAY_LEN(vertex_attributes),
		.pVertexAttributeDescriptions = vertex_attributes
	};

	VkPipelineInputAssemblyStateCreateInfo input_assembly_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE
	};

	VkPipelineViewportStateCreateInfo viewport_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.viewportCount = 1,
		.pViewports = NULL,
		.scissorCount = 1,
		.pScissors = NULL
	};

	VkPipelineRasterizationStateCreateInfo rasterization_info =
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

	VkPipelineMultisampleStateCreateInfo multisample_info =
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

	VkPipelineDepthStencilStateCreateInfo depth_stencil_info =
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

	VkPipelineColorBlendStateCreateInfo color_blend_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.logicOpEnable = VK_FALSE,
		.logicOp = VK_LOGIC_OP_CLEAR,
		.attachmentCount = 1,
		.pAttachments = &xr_vk_no_blending_attachment,
		.blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f }
	};

	VkPushConstantRange push_constants[] =
	{
		{
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
			.offset = 0,
			.size = sizeof(vk_skybox_constant_data_t)
		}
	};

	VkDescriptorSetLayout set_layouts[] =
	{
		xr->vk.sampler_set_layout.layout
	};

	VkPipelineLayoutCreateInfo pipeline_layout_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.setLayoutCount = MACRO_ARRAY_LEN(set_layouts),
		.pSetLayouts = set_layouts,
		.pushConstantRangeCount = MACRO_ARRAY_LEN(push_constants),
		.pPushConstantRanges = push_constants
	};

	VkResult result = xr->vk.table.vkCreatePipelineLayout(xr->vk.device,
		&pipeline_layout_info, NULL, &xr->vk.output.skybox.pipeline_layout);
	hard_assert_eq(result, VK_SUCCESS);


	VkGraphicsPipelineCreateInfo pipeline_info =
	{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.stageCount = MACRO_ARRAY_LEN(shader_stages),
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input_info,
		.pInputAssemblyState = &input_assembly_info,
		.pTessellationState = NULL,
		.pViewportState = &viewport_info,
		.pRasterizationState = &rasterization_info,
		.pMultisampleState = &multisample_info,
		.pDepthStencilState = &depth_stencil_info,
		.pColorBlendState = &color_blend_info,
		.pDynamicState = &dynamic_state_info,
		.layout = xr->vk.output.skybox.pipeline_layout,
		.renderPass = xr->vk.output.render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	static const char* pipeline_cache_path = "cache/xr/skybox_pipeline.bin";
	VkPipelineCache pipeline_cache = xr_init_pipeline_cache(xr, pipeline_cache_path);

	result = xr->vk.table.vkCreateGraphicsPipelines(xr->vk.device,
		pipeline_cache, 1, &pipeline_info, NULL, &xr->vk.output.skybox.pipeline);
	hard_assert_eq(result, VK_SUCCESS);

	xr_free_pipeline_cache(xr, pipeline_cache_path, pipeline_cache);

	for(uint32_t i = 0; i < MACRO_ARRAY_LEN(shader_stages); ++i)
	{
		xr_destroy_shader(xr, shader_stages[i].module);
	}
}


private void
xr_free_skybox_pipeline(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr->vk.table.vkDestroyPipeline(xr->vk.device, xr->vk.output.skybox.pipeline, NULL);
	xr->vk.table.vkDestroyPipelineLayout(xr->vk.device, xr->vk.output.skybox.pipeline_layout, NULL);
}


private void
xr_init_skybox_consts(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr->vk.output.skybox.sky.image =
	(vk_image_t)
	{
		.path = simulation_get_skybox_path(xr->simulation),
		.type = VK_IMAGE_TYPE_TEXTURE_CUBE_BITS
	};
	xr_init_frame_image(xr, &xr->vk.output.skybox.sky);

	xr_init_vertex_buffer(xr, sizeof(xr_vk_skybox_vertex_data), &xr->vk.output.skybox.vertex_buffer);

	xr_copy_to_buffer(xr, &xr->vk.output.skybox.vertex_buffer,
		xr_vk_skybox_vertex_data, sizeof(xr_vk_skybox_vertex_data));

	xr_init_index_buffer(xr, sizeof(xr_vk_skybox_index_data), &xr->vk.output.skybox.index_buffer);

	xr_copy_to_buffer(xr, &xr->vk.output.skybox.index_buffer,
		xr_vk_skybox_index_data, sizeof(xr_vk_skybox_index_data));
}


private void
xr_free_skybox_consts(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_buffer(xr, &xr->vk.output.skybox.index_buffer);
	xr_free_buffer(xr, &xr->vk.output.skybox.vertex_buffer);

	xr_free_frame_image(xr, &xr->vk.output.skybox.sky);
}


private void
xr_init_compose_pipeline(
	xr_t xr
	)
{
	assert_not_null(xr);

	typedef struct xr_compose_frag_specialization
	{
		int32_t enable_ssao;
	}
	xr_compose_frag_specialization_t;

	xr_compose_frag_specialization_t frag_specialization_data =
	{
		.enable_ssao = xr->options.enable_ssao
	};

	VkSpecializationMapEntry frag_map_entries[] =
	{
		{
			.constantID = 0,
			.offset = offsetof(xr_compose_frag_specialization_t, enable_ssao),
			.size = sizeof(frag_specialization_data.enable_ssao)
		}
	};

	VkSpecializationInfo frag_specialization_info =
	{
		.mapEntryCount = MACRO_ARRAY_LEN(frag_map_entries),
		.pMapEntries = frag_map_entries,
		.dataSize = sizeof(frag_specialization_data),
		.pData = &frag_specialization_data
	};

	VkPipelineShaderStageCreateInfo shader_stages[] =
	{
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = xr_create_shader(xr, "shaders/xr/fullscreen.vert.spv"),
			.pName = "main",
			.pSpecializationInfo = NULL
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = xr_create_shader(xr, "shaders/xr/compose.frag.spv"),
			.pName = "main",
			.pSpecializationInfo = &frag_specialization_info
		}
	};

	VkDynamicState dynamic_states[] =
	{
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};

	VkPipelineDynamicStateCreateInfo dynamic_state_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.dynamicStateCount = MACRO_ARRAY_LEN(dynamic_states),
		.pDynamicStates = dynamic_states
	};

	VkPipelineVertexInputStateCreateInfo vertex_input_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.vertexBindingDescriptionCount = 0,
		.pVertexBindingDescriptions = NULL,
		.vertexAttributeDescriptionCount = 0,
		.pVertexAttributeDescriptions = NULL
	};

	VkPipelineInputAssemblyStateCreateInfo input_assembly_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE
	};

	VkPipelineViewportStateCreateInfo viewport_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.viewportCount = 1,
		.pViewports = NULL,
		.scissorCount = 1,
		.pScissors = NULL
	};

	VkPipelineRasterizationStateCreateInfo rasterization_info =
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

	VkPipelineMultisampleStateCreateInfo multisample_info =
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

	VkPipelineDepthStencilStateCreateInfo depth_stencil_info =
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

	VkPipelineColorBlendStateCreateInfo color_blend_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.logicOpEnable = VK_FALSE,
		.logicOp = VK_LOGIC_OP_CLEAR,
		.attachmentCount = 1,
		.pAttachments = &xr_vk_blending_attachment,
		.blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f }
	};

	VkDescriptorSetLayout set_layouts[] =
	{
		xr->vk.sampler_set_layout.layout,
		xr->vk.sampler_set_layout.layout
	};

	VkPipelineLayoutCreateInfo pipeline_layout_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.setLayoutCount = MACRO_ARRAY_LEN(set_layouts),
		.pSetLayouts = set_layouts,
		.pushConstantRangeCount = 0,
		.pPushConstantRanges = NULL
	};

	VkResult result = xr->vk.table.vkCreatePipelineLayout(xr->vk.device,
		&pipeline_layout_info, NULL, &xr->vk.output.compose.pipeline_layout);
	hard_assert_eq(result, VK_SUCCESS);


	VkGraphicsPipelineCreateInfo pipeline_info =
	{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.stageCount = MACRO_ARRAY_LEN(shader_stages),
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input_info,
		.pInputAssemblyState = &input_assembly_info,
		.pTessellationState = NULL,
		.pViewportState = &viewport_info,
		.pRasterizationState = &rasterization_info,
		.pMultisampleState = &multisample_info,
		.pDepthStencilState = &depth_stencil_info,
		.pColorBlendState = &color_blend_info,
		.pDynamicState = &dynamic_state_info,
		.layout = xr->vk.output.compose.pipeline_layout,
		.renderPass = xr->vk.output.render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	static const char* pipeline_cache_path = "cache/xr/compose_pipeline.bin";
	VkPipelineCache pipeline_cache = xr_init_pipeline_cache(xr, pipeline_cache_path);

	result = xr->vk.table.vkCreateGraphicsPipelines(xr->vk.device,
		pipeline_cache, 1, &pipeline_info, NULL, &xr->vk.output.compose.pipeline);
	hard_assert_eq(result, VK_SUCCESS);

	xr_free_pipeline_cache(xr, pipeline_cache_path, pipeline_cache);

	for(uint32_t i = 0; i < MACRO_ARRAY_LEN(shader_stages); ++i)
	{
		xr_destroy_shader(xr, shader_stages[i].module);
	}
}


private void
xr_free_compose_pipeline(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr->vk.table.vkDestroyPipeline(xr->vk.device, xr->vk.output.compose.pipeline, NULL);
	xr->vk.table.vkDestroyPipelineLayout(xr->vk.device, xr->vk.output.compose.pipeline_layout, NULL);
}


private void
xr_init_compose_consts(
	xr_t xr
	)
{
	assert_not_null(xr);
}


private void
xr_free_compose_consts(
	xr_t xr
	)
{
	assert_not_null(xr);
}


private void
xr_init_preview_pipeline(
	xr_t xr,
	const char* frag_shader_path,
	const char* pipeline_cache_path,
	VkPipelineLayout* pipeline_layout,
	VkPipeline* pipeline
	)
{
	assert_not_null(xr);
	assert_not_null(frag_shader_path);
	assert_not_null(pipeline_cache_path);
	assert_not_null(pipeline_layout);
	assert_not_null(pipeline);

	VkPipelineShaderStageCreateInfo shader_stages[] =
	{
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = xr_create_shader(xr, "shaders/xr/fullscreen.vert.spv"),
			.pName = "main",
			.pSpecializationInfo = NULL
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = xr_create_shader(xr, frag_shader_path),
			.pName = "main",
			.pSpecializationInfo = NULL
		}
	};

	VkDynamicState dynamic_states[] =
	{
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};

	VkPipelineDynamicStateCreateInfo dynamic_state_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.dynamicStateCount = MACRO_ARRAY_LEN(dynamic_states),
		.pDynamicStates = dynamic_states
	};

	VkPipelineVertexInputStateCreateInfo vertex_input_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.vertexBindingDescriptionCount = 0,
		.pVertexBindingDescriptions = NULL,
		.vertexAttributeDescriptionCount = 0,
		.pVertexAttributeDescriptions = NULL
	};

	VkPipelineInputAssemblyStateCreateInfo input_assembly_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE
	};

	VkPipelineViewportStateCreateInfo viewport_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.viewportCount = 1,
		.pViewports = NULL,
		.scissorCount = 1,
		.pScissors = NULL
	};

	VkPipelineRasterizationStateCreateInfo rasterization_info =
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

	VkPipelineMultisampleStateCreateInfo multisample_info =
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

	VkPipelineDepthStencilStateCreateInfo depth_stencil_info =
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

	VkPipelineColorBlendStateCreateInfo color_blend_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.logicOpEnable = VK_FALSE,
		.logicOp = VK_LOGIC_OP_CLEAR,
		.attachmentCount = 1,
		.pAttachments = &xr_vk_no_blending_attachment,
		.blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f }
	};

	VkDescriptorSetLayout set_layouts[] =
	{
		xr->vk.sampler_set_layout.layout
	};

	VkPipelineLayoutCreateInfo pipeline_layout_info =
	{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.setLayoutCount = MACRO_ARRAY_LEN(set_layouts),
		.pSetLayouts = set_layouts,
		.pushConstantRangeCount = 0,
		.pPushConstantRanges = NULL
	};

	VkResult result = xr->vk.table.vkCreatePipelineLayout(
		xr->vk.device, &pipeline_layout_info, NULL, pipeline_layout);
	hard_assert_eq(result, VK_SUCCESS);


	VkGraphicsPipelineCreateInfo pipeline_info =
	{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.stageCount = MACRO_ARRAY_LEN(shader_stages),
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input_info,
		.pInputAssemblyState = &input_assembly_info,
		.pTessellationState = NULL,
		.pViewportState = &viewport_info,
		.pRasterizationState = &rasterization_info,
		.pMultisampleState = &multisample_info,
		.pDepthStencilState = &depth_stencil_info,
		.pColorBlendState = &color_blend_info,
		.pDynamicState = &dynamic_state_info,
		.layout = *pipeline_layout,
		.renderPass = xr->vk.output.render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	VkPipelineCache pipeline_cache = xr_init_pipeline_cache(xr, pipeline_cache_path);

	result = xr->vk.table.vkCreateGraphicsPipelines(
		xr->vk.device, pipeline_cache, 1, &pipeline_info, NULL, pipeline);
	hard_assert_eq(result, VK_SUCCESS);

	xr_free_pipeline_cache(xr, pipeline_cache_path, pipeline_cache);

	for(uint32_t i = 0; i < MACRO_ARRAY_LEN(shader_stages); ++i)
	{
		xr_destroy_shader(xr, shader_stages[i].module);
	}
}


private void
xr_free_preview_pipeline(
	xr_t xr,
	VkPipelineLayout pipeline_layout,
	VkPipeline pipeline
	)
{
	assert_not_null(xr);
	assert_not_null(pipeline_layout);
	assert_not_null(pipeline);

	xr->vk.table.vkDestroyPipeline(xr->vk.device, pipeline, NULL);
	xr->vk.table.vkDestroyPipelineLayout(xr->vk.device, pipeline_layout, NULL);
}


private void
xr_init_preview_pipelines(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_init_preview_pipeline(xr, "shaders/xr/preview_depth.frag.spv", "cache/xr/preview_depth_pipeline.bin",
		&xr->vk.output.preview.depth.pipeline_layout, &xr->vk.output.preview.depth.pipeline);

	xr_init_preview_pipeline(xr, "shaders/xr/preview_image.frag.spv", "cache/xr/preview_image_pipeline.bin",
		&xr->vk.output.preview.image.pipeline_layout, &xr->vk.output.preview.image.pipeline);
}


private void
xr_free_preview_pipelines(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_preview_pipeline(xr, xr->vk.output.preview.image.pipeline_layout, xr->vk.output.preview.image.pipeline);
	xr_free_preview_pipeline(xr, xr->vk.output.preview.depth.pipeline_layout, xr->vk.output.preview.depth.pipeline);
}


private void
xr_init_preview_consts(
	xr_t xr
	)
{
	assert_not_null(xr);
}


private void
xr_free_preview_consts(
	xr_t xr
	)
{
	assert_not_null(xr);
}


private void
xr_init_output(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_init_output_pass(xr);
	xr_init_skybox_pipeline(xr);
	xr_init_skybox_consts(xr);
	xr_init_compose_pipeline(xr);
	xr_init_compose_consts(xr);
	xr_init_preview_pipelines(xr);
	xr_init_preview_consts(xr);
}


private void
xr_free_output(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_preview_consts(xr);
	xr_free_preview_pipelines(xr);
	xr_free_compose_consts(xr);
	xr_free_compose_pipeline(xr);
	xr_free_skybox_consts(xr);
	xr_free_skybox_pipeline(xr);
	xr_free_output_pass(xr);
}


private void
xr_init_pipelines(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_init_samplers(xr);
	xr_init_shadow(xr);
	xr_init_scene(xr);
	xr_init_ssao(xr);
	xr_init_ssao_blur(xr);
	xr_init_output(xr);
}


private void
xr_free_pipelines(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_output(xr);
	xr_free_ssao_blur(xr);
	xr_free_ssao(xr);
	xr_free_scene(xr);
	xr_free_shadow(xr);
	xr_free_samplers(xr);
}


private void
xr_init_models(
	xr_t xr
	)
{
	assert_not_null(xr);

	simulation_model_info_t info = simulation_get_model_info(xr->simulation);
	xr->vk.model_count = info.model_count;
	xr->vk.material_count = info.material_count;

	VkDescriptorImageInfo descriptor_images[xr->vk.material_count];
	VkDescriptorImageInfo* descriptor_image = descriptor_images;

	VkWriteDescriptorSet descriptor_writes[xr->vk.material_count];
	VkWriteDescriptorSet* descriptor_write = descriptor_writes;

	VkDescriptorSet sets[xr->vk.material_count];
	VkDescriptorSet* set = sets;

	xr_get_descriptor_sets(xr, xr->vk.sampler_set_layout, sets, xr->vk.material_count);



	xr->vk.materials = alloc_malloc(sizeof(*xr->vk.materials) * xr->vk.material_count);
	assert_ptr(xr->vk.materials, sizeof(*xr->vk.materials) * xr->vk.material_count);

	xr->vk.models = alloc_malloc(sizeof(*xr->vk.models) * xr->vk.model_count);
	assert_ptr(xr->vk.models, sizeof(*xr->vk.models) * xr->vk.model_count);

	vk_material_t* material = xr->vk.materials;
	vk_model_t* model = xr->vk.models;

	for(uint32_t i = 0; i < xr->vk.model_count; ++i)
	{
		model_t* sim_model = info.models[i];

		model->meshes = alloc_malloc(sizeof(*model->meshes) * sim_model->mesh_count);
		assert_ptr(model->meshes, sizeof(*model->meshes) * sim_model->mesh_count);
		model->mesh_count = sim_model->mesh_count;

		vk_mesh_t* mesh = model->meshes;

		mesh_t* sim_mesh = sim_model->meshes;
		mesh_t* sim_mesh_end = sim_mesh + sim_model->mesh_count;

		while(sim_mesh < sim_mesh_end)
		{
			mesh->material_idx = material - xr->vk.materials + sim_mesh->material_idx;
			mesh->vertex_count = sim_mesh->vertex_count;
			mesh->index_count = sim_mesh->index_count;

			xr_init_vertex_buffer(xr, sizeof(vk_shadow_vertex_data_t) *
				sim_mesh->vertex_count, &mesh->shadow_vertex_buffer);
			xr_copy_to_buffer(xr, &mesh->shadow_vertex_buffer,
				sim_mesh->vertices, sizeof(*sim_mesh->vertices) * sim_mesh->vertex_count);

			vk_mesh_vertex_data_t* vertex_data =
				alloc_malloc(sizeof(*vertex_data) * sim_mesh->vertex_count);
			assert_ptr(vertex_data, sizeof(*vertex_data) * sim_mesh->vertex_count);

			vk_mesh_vertex_data_t* data = vertex_data;
			vk_mesh_vertex_data_t* data_end = data + sim_mesh->vertex_count;

			vec3* vertex = sim_mesh->vertices;
			vec3* normal = sim_mesh->normals;
			vec2* coord = sim_mesh->coords;

			while(data < data_end)
			{
				glm_vec3_copy(*vertex, data->position);
				glm_vec3_copy(*normal, data->normal);
				glm_vec2_copy(*coord, data->coords);

				++data;
				++vertex;
				++normal;
				++coord;
			}

			xr_init_vertex_buffer(xr, sizeof(*vertex_data) *
				sim_mesh->vertex_count, &mesh->scene_vertex_buffer);
			xr_copy_to_buffer(xr, &mesh->scene_vertex_buffer,
				vertex_data, sizeof(*vertex_data) * sim_mesh->vertex_count);

			alloc_free(vertex_data, sizeof(*vertex_data) * sim_mesh->vertex_count);

			xr_init_index_buffer(xr, sizeof(*sim_mesh->indexes) *
				sim_mesh->index_count, &mesh->index_buffer);
			xr_copy_to_buffer(xr, &mesh->index_buffer,
				sim_mesh->indexes, sizeof(*sim_mesh->indexes) * sim_mesh->index_count);

			++mesh;
			++sim_mesh;
		}


		material_t* sim_material = sim_model->materials;
		material_t* sim_material_end = sim_material + sim_model->material_count;

		while(sim_material < sim_material_end)
		{
			material->texture.type = VK_IMAGE_TYPE_TEXTURE_2D_BITS;

			if(!str_is_empty(sim_material->texture))
			{
				material->texture.path = str_init_copy(sim_material->texture);
			}
			else
			{
				material->texture.path = str_init_copy_cstr("assets/blank.png");
			}

			xr_init_image(xr, &material->texture);

			glm_vec3_copy(sim_material->ambient, material->constant_data.ambient);
			glm_vec3_copy(sim_material->diffuse, material->constant_data.diffuse);
			material->constant_data.shininess = sim_material->shininess;
			material->constant_data.shininess_strength = sim_material->shininess_strength;

			*descriptor_image =
			(VkDescriptorImageInfo)
			{
				.sampler = xr->vk.image_sampler,
				.imageView = material->texture.view,
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
			};

			*descriptor_write =
			(VkWriteDescriptorSet)
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.pNext = NULL,
				.dstSet = *set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = descriptor_image,
				.pBufferInfo = NULL,
				.pTexelBufferView = NULL
			};

			material->set = *set;

			++material;
			++sim_material;
			++descriptor_image;
			++descriptor_write;
			++set;
		}

		xr_init_vertex_buffer(xr, sizeof(vk_model_instance_data_t) * VK_MAX_INSTANCES, &model->instance_buffer);

		++model;
	}

	xr->vk.table.vkUpdateDescriptorSets(xr->vk.device, xr->vk.material_count, descriptor_writes, 0, NULL);
}


private void
xr_free_models(
	xr_t xr
	)
{
	assert_not_null(xr);

	vk_model_t* model = xr->vk.models;
	vk_model_t* model_end = model + xr->vk.model_count;

	while(model < model_end)
	{
		xr_free_buffer(xr, &model->instance_buffer);

		vk_mesh_t* mesh = model->meshes;
		vk_mesh_t* mesh_end = mesh + model->mesh_count;

		while(mesh < mesh_end)
		{
			xr_free_buffer(xr, &mesh->index_buffer);
			xr_free_buffer(xr, &mesh->scene_vertex_buffer);
			xr_free_buffer(xr, &mesh->shadow_vertex_buffer);

			++mesh;
		}

		alloc_free(model->meshes, sizeof(*model->meshes) * model->mesh_count);

		++model;
	}

	alloc_free(xr->vk.models, sizeof(*xr->vk.models) * xr->vk.model_count);

	vk_material_t* material = xr->vk.materials;
	vk_material_t* material_end = material + xr->vk.material_count;

	while(material < material_end)
	{
		xr_free_image(xr, &material->texture);
		str_free(material->texture.path);
		++material;
	}

	alloc_free(xr->vk.materials, sizeof(*xr->vk.materials) * xr->vk.material_count);
}


private void
xr_init_frames(
	xr_t xr
	)
{
	assert_not_null(xr);

	VkCommandBuffer command_buffers[MACRO_ARRAY_LEN(xr->vk.barriers)];
	VkCommandBuffer* command_buffer = command_buffers;

	VkCommandBufferAllocateInfo command_buffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = NULL,
		.commandPool = xr->vk.command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = MACRO_ARRAY_LEN(command_buffers)
	};

	VkResult result = xr->vk.table.vkAllocateCommandBuffers(
		xr->vk.device, &command_buffer_info, command_buffers);
	hard_assert_eq(result, VK_SUCCESS);


	VkSemaphoreCreateInfo semaphore_info =
	{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0
	};

	VkFenceCreateInfo fence_info =
	{
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.pNext = NULL,
		.flags = VK_FENCE_CREATE_SIGNALED_BIT
	};

	vk_barrier_t* barrier = xr->vk.barriers;
	vk_barrier_t* barrier_end = barrier + MACRO_ARRAY_LEN(xr->vk.barriers);

	while(barrier < barrier_end)
	{
		result = xr->vk.table.vkCreateSemaphore(xr->vk.device, &semaphore_info, NULL, &barrier->semaphore);
		hard_assert_eq(result, VK_SUCCESS);

		result = xr->vk.table.vkCreateFence(xr->vk.device, &fence_info, NULL, &barrier->fence);
		hard_assert_eq(result, VK_SUCCESS);

		barrier->command_buffer = *command_buffer;

		xr_init_timing(xr, &barrier->timing, barrier->command_buffer, VK_BARRIER_TIMING_IDX__COUNT);

		++barrier;
		++command_buffer;
	}

	xr->vk.barrier = xr->vk.barriers;
}


private void
xr_free_frames(
	xr_t xr
	)
{
	assert_not_null(xr);

	VkCommandBuffer command_buffers[MACRO_ARRAY_LEN(xr->vk.barriers)];
	VkCommandBuffer* command_buffer = command_buffers;

	vk_barrier_t* barrier = xr->vk.barriers;
	vk_barrier_t* barrier_end = barrier + MACRO_ARRAY_LEN(xr->vk.barriers);

	while(barrier < barrier_end)
	{
		xr_free_timing(xr, &barrier->timing);

		*command_buffer = barrier->command_buffer;

		xr->vk.table.vkDestroyFence(xr->vk.device, barrier->fence, NULL);
		xr->vk.table.vkDestroySemaphore(xr->vk.device, barrier->semaphore, NULL);

		++barrier;
		++command_buffer;
	}

	xr->vk.table.vkFreeCommandBuffers(xr->vk.device,
		xr->vk.command_pool, MACRO_ARRAY_LEN(xr->vk.barriers), command_buffers);
}


private void
xr_init_shadow_framebuffer(
	xr_t xr,
	vk_frame_t* frame
	)
{
	assert_not_null(xr);
	assert_not_null(frame);

	vk_image_t* image = &frame->shadow.map.image;
	image->type = VK_IMAGE_TYPE_ATTACHMENT_BIT | VK_IMAGE_TYPE_DEPTH_BIT |
		VK_IMAGE_TYPE_SAMPLED_BIT | VK_IMAGE_TYPE_CUSTOM_SIZE_BIT | VK_IMAGE_TYPE_MULTIVIEW_BIT;
	image->width = xr->options.shadow_map_size;
	image->height = xr->options.shadow_map_size;
	xr_init_frame_image(xr, &frame->shadow.map);

	VkImageView attachments[] =
	{
		image->view
	};

	VkFramebufferCreateInfo framebuffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.renderPass = xr->vk.shadow.render_pass,
		.attachmentCount = MACRO_ARRAY_LEN(attachments),
		.pAttachments = attachments,
		.width = image->width,
		.height = image->height,
		.layers = 1
	};

	VkResult result = xr->vk.table.vkCreateFramebuffer(xr->vk.device,
		&framebuffer_info, NULL, &frame->shadow.framebuffer);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
xr_free_shadow_framebuffer(
	xr_t xr,
	vk_frame_t* frame
	)
{
	assert_not_null(xr);
	assert_not_null(frame);

	xr->vk.table.vkDestroyFramebuffer(xr->vk.device, frame->shadow.framebuffer, NULL);

	xr_free_frame_image(xr, &frame->shadow.map);
}


private void
xr_init_scene_framebuffer(
	xr_t xr,
	vk_frame_t* frame
	)
{
	assert_not_null(xr);
	assert_not_null(frame);

	xr_init_vert_ubo_buffer(xr, sizeof(vk_scene_vert_ubo_data_t), &frame->scene.vert_ubo);

	frame->scene.position_ms.image.type = VK_IMAGE_TYPE_ATTACHMENT_BIT |
		VK_IMAGE_TYPE_SAMPLED_BIT | VK_IMAGE_TYPE_MULTISAMPLED_BIT | VK_IMAGE_TYPE_CUSTOM_FORMAT_BIT | VK_IMAGE_TYPE_MULTIVIEW_BIT;
	frame->scene.position_ms.image.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	xr_init_frame_image(xr, &frame->scene.position_ms);

	frame->scene.normal_ms.image.type = VK_IMAGE_TYPE_ATTACHMENT_BIT |
		VK_IMAGE_TYPE_SAMPLED_BIT | VK_IMAGE_TYPE_MULTISAMPLED_BIT | VK_IMAGE_TYPE_CUSTOM_FORMAT_BIT | VK_IMAGE_TYPE_MULTIVIEW_BIT;
	frame->scene.normal_ms.image.format = VK_FORMAT_R8G8B8A8_UNORM;
	xr_init_frame_image(xr, &frame->scene.normal_ms);

	frame->scene.map_ms.image.type = VK_IMAGE_TYPE_ATTACHMENT_BIT |
		VK_IMAGE_TYPE_SAMPLED_BIT | VK_IMAGE_TYPE_MULTISAMPLED_BIT | VK_IMAGE_TYPE_MULTIVIEW_BIT;
	xr_init_frame_image(xr, &frame->scene.map_ms);

	frame->scene.position.image.type = VK_IMAGE_TYPE_ATTACHMENT_BIT |
		VK_IMAGE_TYPE_SAMPLED_BIT | VK_IMAGE_TYPE_CUSTOM_FORMAT_BIT | VK_IMAGE_TYPE_MULTIVIEW_BIT;
	frame->scene.position.image.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	xr_init_frame_image(xr, &frame->scene.position);

	frame->scene.normal.image.type = VK_IMAGE_TYPE_ATTACHMENT_BIT |
		VK_IMAGE_TYPE_SAMPLED_BIT | VK_IMAGE_TYPE_CUSTOM_FORMAT_BIT | VK_IMAGE_TYPE_MULTIVIEW_BIT;
	frame->scene.normal.image.format = VK_FORMAT_R8G8B8A8_UNORM;
	xr_init_frame_image(xr, &frame->scene.normal);

	frame->scene.map.image.type = VK_IMAGE_TYPE_ATTACHMENT_BIT | VK_IMAGE_TYPE_SAMPLED_BIT | VK_IMAGE_TYPE_MULTIVIEW_BIT;
	xr_init_frame_image(xr, &frame->scene.map);

	vk_image_t images[] =
	{
		frame->scene.position.image,
		frame->scene.normal.image
	};

	xr_get_descriptor_sets(xr, xr->vk.scene_set_layout, &frame->scene.set, 1);

	xr_write_images_to_set(xr, frame->scene.set, images, MACRO_ARRAY_LEN(images));

	frame->scene.depth.type = VK_IMAGE_TYPE_ATTACHMENT_BIT |
		VK_IMAGE_TYPE_DEPTH_BIT | VK_IMAGE_TYPE_MULTISAMPLED_BIT | VK_IMAGE_TYPE_TRANSIENT_BIT | VK_IMAGE_TYPE_MULTIVIEW_BIT;
	xr_init_image(xr, &frame->scene.depth);

	VkImageView attachments[] =
	{
		frame->scene.depth.view,
		frame->scene.position_ms.image.view,
		frame->scene.normal_ms.image.view,
		frame->scene.map_ms.image.view,
		frame->scene.position.image.view,
		frame->scene.normal.image.view,
		frame->scene.map.image.view
	};

	VkFramebufferCreateInfo framebuffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.renderPass = xr->vk.scene.render_pass,
		.attachmentCount = MACRO_ARRAY_LEN(attachments),
		.pAttachments = attachments,
		.width = xr->vk.screen_extent.width,
		.height = xr->vk.screen_extent.height,
		.layers = 1
	};

	VkResult result = xr->vk.table.vkCreateFramebuffer(xr->vk.device,
		&framebuffer_info, NULL, &frame->scene.framebuffer);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
xr_free_scene_framebuffer(
	xr_t xr,
	vk_frame_t* frame
	)
{
	assert_not_null(xr);
	assert_not_null(frame);

	xr->vk.table.vkDestroyFramebuffer(xr->vk.device, frame->scene.framebuffer, NULL);

	xr_free_image(xr, &frame->scene.depth);
	xr_free_frame_image(xr, &frame->scene.map);
	xr_free_frame_image(xr, &frame->scene.normal);
	xr_free_frame_image(xr, &frame->scene.position);
	xr_free_frame_image(xr, &frame->scene.map_ms);
	xr_free_frame_image(xr, &frame->scene.normal_ms);
	xr_free_frame_image(xr, &frame->scene.position_ms);
	xr_free_ubo_buffer(xr, &frame->scene.vert_ubo);
}


private void
xr_init_ssao_framebuffer(
	xr_t xr,
	vk_frame_t* frame
	)
{
	assert_not_null(xr);
	assert_not_null(frame);

	xr_init_frag_ubo_buffer(xr, sizeof(vk_ssao_frag_ubo_data_t), &frame->ssao.frag_ubo);

	frame->ssao.map.image =
	(vk_image_t)
	{
		.type = VK_IMAGE_TYPE_ATTACHMENT_BIT | VK_IMAGE_TYPE_SAMPLED_BIT |
			VK_IMAGE_TYPE_CUSTOM_FORMAT_BIT | VK_IMAGE_TYPE_CUSTOM_SIZE_BIT | VK_IMAGE_TYPE_MULTIVIEW_BIT,
		.format = VK_FORMAT_R8_UNORM,
		.width = xr->vk.ssao_extent.width,
		.height = xr->vk.ssao_extent.height
	};
	xr_init_frame_image(xr, &frame->ssao.map);

	VkImageView attachments[] =
	{
		frame->ssao.map.image.view
	};

	VkFramebufferCreateInfo framebuffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.renderPass = xr->vk.ssao.render_pass,
		.attachmentCount = MACRO_ARRAY_LEN(attachments),
		.pAttachments = attachments,
		.width = xr->vk.ssao_extent.width,
		.height = xr->vk.ssao_extent.height,
		.layers = 1
	};

	VkResult result = xr->vk.table.vkCreateFramebuffer(xr->vk.device,
		&framebuffer_info, NULL, &frame->ssao.framebuffer);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
xr_free_ssao_framebuffer(
	xr_t xr,
	vk_frame_t* frame
	)
{
	assert_not_null(xr);
	assert_not_null(frame);

	xr->vk.table.vkDestroyFramebuffer(xr->vk.device, frame->ssao.framebuffer, NULL);

	xr_free_frame_image(xr, &frame->ssao.map);
	xr_free_ubo_buffer(xr, &frame->ssao.frag_ubo);
}


private void
xr_init_ssao_blur_framebuffer(
	xr_t xr,
	vk_frame_t* frame
	)
{
	assert_not_null(xr);
	assert_not_null(frame);

	frame->ssao_blur.map.image =
	(vk_image_t)
	{
		.type = VK_IMAGE_TYPE_ATTACHMENT_BIT | VK_IMAGE_TYPE_SAMPLED_BIT |
			VK_IMAGE_TYPE_CUSTOM_FORMAT_BIT | VK_IMAGE_TYPE_CUSTOM_SIZE_BIT | VK_IMAGE_TYPE_MULTIVIEW_BIT,
		.format = VK_FORMAT_R8_UNORM,
		.width = xr->vk.ssao_extent.width,
		.height = xr->vk.ssao_extent.height
	};
	xr_init_frame_image(xr, &frame->ssao_blur.map);

	VkImageView attachments[] =
	{
		frame->ssao_blur.map.image.view
	};

	VkFramebufferCreateInfo framebuffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.renderPass = xr->vk.ssao_blur.render_pass,
		.attachmentCount = MACRO_ARRAY_LEN(attachments),
		.pAttachments = attachments,
		.width = xr->vk.ssao_extent.width,
		.height = xr->vk.ssao_extent.height,
		.layers = 1
	};

	VkResult result = xr->vk.table.vkCreateFramebuffer(xr->vk.device,
		&framebuffer_info, NULL, &frame->ssao_blur.framebuffer);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
xr_free_ssao_blur_framebuffer(
	xr_t xr,
	vk_frame_t* frame
	)
{
	assert_not_null(xr);
	assert_not_null(frame);

	xr->vk.table.vkDestroyFramebuffer(xr->vk.device, frame->ssao_blur.framebuffer, NULL);

	xr_free_frame_image(xr, &frame->ssao_blur.map);
}


private void
xr_init_output_framebuffer(
	xr_t xr,
	vk_frame_t* frame,
	VkImage* swapchain_image
	)
{
	assert_not_null(xr);
	assert_not_null(frame);

	frame->output.image = *swapchain_image;

	VkImageViewCreateInfo image_view_info =
	{
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.image = frame->output.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
		.format = xr->vk.format,
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
			.layerCount = xr->view_count
		}
	};

	VkResult result = xr->vk.table.vkCreateImageView(xr->vk.device,
		&image_view_info, NULL, &frame->output.image_view);
	hard_assert_eq(result, VK_SUCCESS);

	VkImageView attachments[] =
	{
		frame->output.image_view
	};

	VkFramebufferCreateInfo framebuffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.renderPass = xr->vk.output.render_pass,
		.attachmentCount = MACRO_ARRAY_LEN(attachments),
		.pAttachments = attachments,
		.width = xr->vk.screen_extent.width,
		.height = xr->vk.screen_extent.height,
		.layers = 1
	};

	result = xr->vk.table.vkCreateFramebuffer(xr->vk.device,
		&framebuffer_info, NULL, &frame->output.framebuffer);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
xr_free_output_framebuffer(
	xr_t xr,
	vk_frame_t* frame
	)
{
	assert_not_null(xr);
	assert_not_null(frame);

	xr->vk.table.vkDestroyFramebuffer(xr->vk.device, frame->output.framebuffer, NULL);
	xr->vk.table.vkDestroyImageView(xr->vk.device, frame->output.image_view, NULL);
}


private void
xr_init_framebuffers(
	xr_t xr
	)
{
	assert_not_null(xr);

	VkSemaphoreCreateInfo semaphore_info =
	{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0
	};

	vk_frame_t* frame = xr->vk.frames;
	vk_frame_t* frame_end = frame + xr->vk.image_count;

	uint32_t image_idx = 0;

	while(frame < frame_end)
	{
		xr_init_shadow_framebuffer(xr, frame);
		xr_init_scene_framebuffer(xr, frame);
		xr_init_ssao_framebuffer(xr, frame);
		xr_init_ssao_blur_framebuffer(xr, frame);
		xr_init_output_framebuffer(xr, frame, &xr->vk.images[image_idx].image);


		VkResult result = xr->vk.table.vkCreateSemaphore(xr->vk.device, &semaphore_info, NULL, &frame->semaphore);
		hard_assert_eq(result, VK_SUCCESS);

		frame->barrier = NULL;


		++frame;
		++image_idx;
	}
}


private void
xr_free_framebuffers(
	xr_t xr
	)
{
	assert_not_null(xr);

	vk_frame_t* frame = xr->vk.frames;
	vk_frame_t* frame_end = frame + xr->vk.image_count;

	while(frame < frame_end)
	{
		xr->vk.table.vkDestroySemaphore(xr->vk.device, frame->semaphore, NULL);

		xr_free_output_framebuffer(xr, frame);
		xr_free_ssao_blur_framebuffer(xr, frame);
		xr_free_ssao_framebuffer(xr, frame);
		xr_free_scene_framebuffer(xr, frame);
		xr_free_shadow_framebuffer(xr, frame);

		++frame;
	}
}


private void
xr_device_wait_idle(
	xr_t xr
	)
{
	assert_not_null(xr);

	VkResult result = xr->vk.table.vkDeviceWaitIdle(xr->vk.device);
	hard_assert_eq(result, VK_SUCCESS);
}


private triplet_t
xr_quaternion_to_euler(
	XrQuaternionf q
	)
{
	triplet_t euler;

	float sinr_cosp = 2.0f * (q.w * q.x + q.y * q.z);
	float cosr_cosp = 1.0f - 2.0f * (q.x * q.x + q.y * q.y);
	euler.x = atan2f(sinr_cosp, cosr_cosp);

	float sinp = 2.0f * (q.w * q.y - q.z * q.x);
	if(fabsf(sinp) >= 1.0f)
	{
		euler.y = copysignf(M_PI / 2.0f, sinp);
	}
	else
	{
		euler.y = asinf(sinp);
	}

	float siny_cosp = 2.0f * (q.w * q.z + q.x * q.y);
	float cosy_cosp = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
	euler.z = atan2f(siny_cosp, cosy_cosp);

	return euler;
}


private void
xr_update_hand_tracking(
	xr_t xr
	)
{
	assert_not_null(xr);

	if(xr->options.xr_monado)
	{
		return;
	}

	{
		XrHandJointLocationsEXT locations =
		{
			.type = XR_TYPE_HAND_JOINT_LOCATIONS_EXT,
			.next = NULL,
			.jointCount = XR_HAND_JOINT_COUNT_EXT,
			.jointLocations = xr->hand_joints_left
		};

		XrHandJointsLocateInfoEXT locate_info =
		{
			.type = XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT,
			.next = NULL,
			.baseSpace = xr->space,
			.time = xr->predicted_display_time
		};

		XrResult result = xr->xrLocateHandJointsEXT(xr->hand_tracker_left, &locate_info, &locations);
		xr->hand_joints_left_active = (result == XR_SUCCESS && locations.isActive);
	}

	{
		XrHandJointLocationsEXT locations =
		{
			.type = XR_TYPE_HAND_JOINT_LOCATIONS_EXT,
			.next = NULL,
			.jointCount = XR_HAND_JOINT_COUNT_EXT,
			.jointLocations = xr->hand_joints_right
		};

		XrHandJointsLocateInfoEXT locate_info =
		{
			.type = XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT,
			.next = NULL,
			.baseSpace = xr->space,
			.time = xr->predicted_display_time
		};

		XrResult result = xr->xrLocateHandJointsEXT(xr->hand_tracker_right, &locate_info, &locations);
		xr->hand_joints_right_active = (result == XR_SUCCESS && locations.isActive);
	}
}


private void
xr_get_head_pose(
	xr_t xr,
	xr_pose_t* pose
	)
{
	assert_not_null(xr);
	assert_not_null(pose);

	XrSpaceLocation location = {XR_TYPE_SPACE_LOCATION};
	XrResult result = xrLocateSpace(xr->space, xr->space, xr->predicted_display_time, &location);
	hard_assert_eq(result, XR_SUCCESS);

	hard_assert_neq(location.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT, 0);
	hard_assert_neq(location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT, 0);

	pose->position.x = location.pose.position.x;
	pose->position.y = location.pose.position.y;
	pose->position.z = location.pose.position.z;

	pose->rotation = xr_quaternion_to_euler(location.pose.orientation);
}


private void
xr_get_hand_pose(
	xr_t xr,
	XrHandTrackerEXT hand_tracker,
	xr_pose_t* pose
	)
{
	assert_not_null(xr);
	assert_not_null(pose);

	if(xr->options.xr_monado)
	{
		*pose = (xr_pose_t){0};
		return;
	}

	XrHandJointLocationEXT* joints;
	bool active;

	if(hand_tracker == xr->hand_tracker_left)
	{
		joints = xr->hand_joints_left;
		active = xr->hand_joints_left_active;
	}
	else if(hand_tracker == xr->hand_tracker_right)
	{
		joints = xr->hand_joints_right;
		active = xr->hand_joints_right_active;
	}
	else
	{
		hard_assert_unreachable();
	}

	hard_assert_true(active);

	XrHandJointLocationEXT palm = joints[XR_HAND_JOINT_PALM_EXT];
	hard_assert_neq(palm.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT, 0);
	hard_assert_neq(palm.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT, 0);

	pose->position.x = palm.pose.position.x;
	pose->position.y = palm.pose.position.y;
	pose->position.z = palm.pose.position.z;

	pose->rotation = xr_quaternion_to_euler(palm.pose.orientation);
}


private void
xr_get_eye_poses(
	xr_t xr,
	xr_pose_t* left_pose,
	xr_pose_t* right_pose,
	XrView views[2]
	)
{
	assert_not_null(xr);
	assert_not_null(left_pose);
	assert_not_null(right_pose);

	XrViewState view_state = {XR_TYPE_VIEW_STATE};
	XrView local_views[2] =
	{
		{XR_TYPE_VIEW},
		{XR_TYPE_VIEW}
	};

	XrViewLocateInfo locate_info =
	{
		.type = XR_TYPE_VIEW_LOCATE_INFO,
		.next = NULL,
		.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
		.displayTime = xr->predicted_display_time,
		.space = xr->space
	};

	uint32_t view_count = 0;
	XrResult result = xrLocateViews(xr->session, &locate_info, &view_state, 2, &view_count, local_views);
	hard_assert_eq(result, XR_SUCCESS);
	hard_assert_eq(view_count, 2);

	hard_assert_neq(view_state.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT, 0);
	hard_assert_neq(view_state.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT, 0);

	xr_pose_t head_pose;
	xr_get_head_pose(xr, &head_pose);

	left_pose->position.x = (local_views[0].pose.position.x - head_pose.position.x) * XR_COORDINATE_SCALE;
	left_pose->position.y = (local_views[0].pose.position.y - head_pose.position.y) * XR_COORDINATE_SCALE;
	left_pose->position.z = (local_views[0].pose.position.z - head_pose.position.z) * XR_COORDINATE_SCALE;
	left_pose->rotation = head_pose.rotation;

	right_pose->position.x = (local_views[1].pose.position.x - head_pose.position.x) * XR_COORDINATE_SCALE;
	right_pose->position.y = (local_views[1].pose.position.y - head_pose.position.y) * XR_COORDINATE_SCALE;
	right_pose->position.z = (local_views[1].pose.position.z - head_pose.position.z) * XR_COORDINATE_SCALE;
	right_pose->rotation = head_pose.rotation;

	if(views)
	{
		views[0] = local_views[0];
		views[1] = local_views[1];
	}
}


#define XR_FOR_EACH_MODEL(entities_per_model, ...)									\
do																					\
{																					\
	vk_entities_per_model_t* entities_per_model = entity_data;						\
	vk_entities_per_model_t* entities_per_model##_end =								\
		entities_per_model + xr->vk.model_count;									\
																					\
	__VA_OPT__(vk_model_t* __VA_ARGS__ = xr->vk.models;)							\
																					\
	while(entities_per_model < entities_per_model##_end)							\
	{																				\
		if(entities_per_model->entities_used != 0)									\
		{																			\
			hard_assert_le(entities_per_model->entities_used, VK_MAX_INSTANCES);

#define XR_FOR_EACH_MODEL_END(entities_per_model, ...)	\
		}												\
														\
		++entities_per_model;							\
		__VA_OPT__(++__VA_ARGS__;)						\
	}													\
}														\
while(0)


private void
xr_draw_shadow(
	xr_t xr,
	vk_frame_t* frame,
	simulation_camera_t* camera,
	simulation_vr_transform_t* vr_transform,
	vk_entities_per_model_t* entity_data
	)
{
	assert_not_null(xr);
	assert_not_null(frame);

	VkClearValue clear_values[] =
	{
		{
			.depthStencil = { 1.0f, 0 }
		}
	};

	VkRenderPassBeginInfo render_pass_begin_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.pNext = NULL,
		.renderPass = xr->vk.shadow.render_pass,
		.framebuffer = frame->shadow.framebuffer,
		.renderArea = xr->vk.shadow_extent.scissor,
		.clearValueCount = MACRO_ARRAY_LEN(clear_values),
		.pClearValues = clear_values
	};

	xr->vk.table.vkCmdBeginRenderPass(xr->vk.barrier->command_buffer,
		&render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

	xr->vk.table.vkCmdBindPipeline(xr->vk.barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.shadow.pipeline);

	vk_shadow_vert_constant_data_t shadow_vert_constant_data;
	glm_mat4_copy(vr_transform->light_transform, shadow_vert_constant_data.transform);

	xr->vk.table.vkCmdPushConstants(xr->vk.barrier->command_buffer,
		xr->vk.shadow.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
		0, sizeof(shadow_vert_constant_data), &shadow_vert_constant_data);

	XR_FOR_EACH_MODEL(entities_per_model, model)
	{
		xr->vk.table.vkCmdBindVertexBuffers(xr->vk.barrier->command_buffer,
			1, 1, &model->instance_buffer.buffer, (VkDeviceSize[]){0});

		vk_mesh_t* mesh = model->meshes;
		vk_mesh_t* mesh_end = mesh + model->mesh_count;

		while(mesh < mesh_end)
		{
			xr->vk.table.vkCmdBindVertexBuffers(xr->vk.barrier->command_buffer,
				0, 1, &mesh->shadow_vertex_buffer.buffer, (VkDeviceSize[]){0});

			xr->vk.table.vkCmdBindIndexBuffer(xr->vk.barrier->command_buffer,
				mesh->index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

			xr->vk.table.vkCmdDrawIndexed(xr->vk.barrier->command_buffer,
				mesh->index_count, entities_per_model->entities_used, 0, 0, 0);

			++mesh;
		}
	}
	XR_FOR_EACH_MODEL_END(entities_per_model, model);

	xr->vk.table.vkCmdEndRenderPass(xr->vk.barrier->command_buffer);
}


private void
xr_draw_scene(
	xr_t xr,
	vk_frame_t* frame,
	simulation_camera_t* camera,
	simulation_vr_transform_t* vr_transform,
	vk_entities_per_model_t* entity_data
	)
{
	assert_not_null(xr);
	assert_not_null(frame);

	VkClearValue clear_values[] =
	{
		{
			.depthStencil = { 0.0f, 0 }
		},
		{
			.color = { { 0.0f, 0.0f, 0.0f, 0.0f } }
		},
		{
			.color = { { 0.0f, 0.0f, 0.0f, 0.0f } }
		},
		{
			.color = { { 0.0f, 0.0f, 0.0f, 0.0f } }
		},
		{
			.color = { { 0.0f, 0.0f, 0.0f, 0.0f } }
		},
		{
			.color = { { 0.0f, 0.0f, 0.0f, 0.0f } }
		},
		{
			.color = { { 0.0f, 0.0f, 0.0f, 0.0f } }
		}
	};

	VkRenderPassBeginInfo render_pass_begin_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.pNext = NULL,
		.renderPass = xr->vk.scene.render_pass,
		.framebuffer = frame->scene.framebuffer,
		.renderArea = xr->vk.screen_extent.scissor,
		.clearValueCount = MACRO_ARRAY_LEN(clear_values),
		.pClearValues = clear_values
	};

	xr->vk.table.vkCmdBeginRenderPass(xr->vk.barrier->command_buffer,
		&render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

	xr->vk.table.vkCmdSetViewport(xr->vk.barrier->command_buffer, 0, 1, &xr->vk.screen_extent.viewport);
	xr->vk.table.vkCmdSetScissor(xr->vk.barrier->command_buffer, 0, 1, &xr->vk.screen_extent.scissor);

	xr_pose_t left_eye_pose;
	xr_pose_t right_eye_pose;
	xr_get_eye_poses(xr, &left_eye_pose, &right_eye_pose, NULL);

	vk_scene_vert_ubo_data_t scene_vert_ubo_data;

	glm_mat4_copy(vr_transform->projection[0], scene_vert_ubo_data.projection[0]);
	glm_mat4_copy(vr_transform->projection[1], scene_vert_ubo_data.projection[1]);

	glm_mat4_copy(vr_transform->view[0], scene_vert_ubo_data.view[0]);
	glm_mat4_copy(vr_transform->view[1], scene_vert_ubo_data.view[1]);

	glm_mat4_copy(vr_transform->light_transform, scene_vert_ubo_data.light_transform);
	glm_vec4_copy(vr_transform->light_direction, scene_vert_ubo_data.light_direction);

	glm_vec3_copy((vec3){left_eye_pose.position.x, left_eye_pose.position.y, left_eye_pose.position.z},
		scene_vert_ubo_data.camera_position[0]);
	scene_vert_ubo_data.camera_position[0][3] = 1.0f;

	glm_vec3_copy((vec3){right_eye_pose.position.x, right_eye_pose.position.y, right_eye_pose.position.z},
		scene_vert_ubo_data.camera_position[1]);
	scene_vert_ubo_data.camera_position[1][3] = 1.0f;

	xr_copy_to_buffer(xr, &frame->scene.vert_ubo.buffer,
		&scene_vert_ubo_data, sizeof(scene_vert_ubo_data));

	vk_scene_frag_constant_data_t scene_frag_constant_data =
	{
		.near = camera->near
	};

	xr->vk.table.vkCmdBindPipeline(xr->vk.barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.scene.pipeline);

	xr->vk.table.vkCmdBindDescriptorSets(xr->vk.barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.scene.pipeline_layout,
		0, 1, &frame->scene.vert_ubo.set, 0, NULL);

	xr->vk.table.vkCmdBindDescriptorSets(xr->vk.barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.scene.pipeline_layout,
		2, 1, &frame->shadow.map.set, 0, NULL);

	XR_FOR_EACH_MODEL(entities_per_model, model)
	{
		xr->vk.table.vkCmdBindVertexBuffers(xr->vk.barrier->command_buffer,
			1, 1, &model->instance_buffer.buffer, (VkDeviceSize[]){0});

		vk_mesh_t* mesh = model->meshes;
		vk_mesh_t* mesh_end = mesh + model->mesh_count;

		while(mesh < mesh_end)
		{
			vk_material_t* material = xr->vk.materials + mesh->material_idx;

			glm_vec4_copy(material->constant_data.diffuse, scene_frag_constant_data.diffuse);
			glm_vec4_copy(material->constant_data.ambient, scene_frag_constant_data.ambient);
			scene_frag_constant_data.shininess = material->constant_data.shininess;
			scene_frag_constant_data.shininess_strength = material->constant_data.shininess_strength;

			xr->vk.table.vkCmdPushConstants(xr->vk.barrier->command_buffer,
				xr->vk.scene.pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
				0, sizeof(scene_frag_constant_data), &scene_frag_constant_data);

			xr->vk.table.vkCmdBindVertexBuffers(xr->vk.barrier->command_buffer,
				0, 1, &mesh->scene_vertex_buffer.buffer, (VkDeviceSize[]){0});

			xr->vk.table.vkCmdBindIndexBuffer(xr->vk.barrier->command_buffer,
				mesh->index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

			xr->vk.table.vkCmdBindDescriptorSets(xr->vk.barrier->command_buffer,
				VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.scene.pipeline_layout,
				1, 1, &material->set, 0, NULL);

			xr->vk.table.vkCmdDrawIndexed(xr->vk.barrier->command_buffer,
				mesh->index_count, entities_per_model->entities_used, 0, 0, 0);

			++mesh;
		}
	}
	XR_FOR_EACH_MODEL_END(entities_per_model, model);

	xr->vk.table.vkCmdEndRenderPass(xr->vk.barrier->command_buffer);
}


private void
xr_draw_ssao(
	xr_t xr,
	vk_frame_t* frame,
	simulation_camera_t* camera,
	simulation_vr_transform_t* vr_transform,
	vk_entities_per_model_t* entity_data
	)
{
	assert_not_null(xr);
	assert_not_null(frame);

	VkRenderPassBeginInfo render_pass_begin_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.pNext = NULL,
		.renderPass = xr->vk.ssao.render_pass,
		.framebuffer = frame->ssao.framebuffer,
		.renderArea = xr->vk.ssao_extent.scissor,
		.clearValueCount = 0,
		.pClearValues = NULL
	};

	xr->vk.table.vkCmdBeginRenderPass(xr->vk.barrier->command_buffer,
		&render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

	xr->vk.table.vkCmdSetViewport(xr->vk.barrier->command_buffer, 0, 1, &xr->vk.ssao_extent.viewport);
	xr->vk.table.vkCmdSetScissor(xr->vk.barrier->command_buffer, 0, 1, &xr->vk.ssao_extent.scissor);

	vk_ssao_frag_ubo_data_t ssao_frag_ubo_data;
	glm_mat4_copy(vr_transform->projection[0], ssao_frag_ubo_data.projection);

	xr_copy_to_buffer(xr, &frame->ssao.frag_ubo.buffer,
		&ssao_frag_ubo_data, sizeof(ssao_frag_ubo_data));

	xr->vk.table.vkCmdBindPipeline(xr->vk.barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.ssao.pipeline);

	VkDescriptorSet sets[] =
	{
		frame->scene.set,
		xr->vk.ssao.noise.set,
		frame->ssao.frag_ubo.set,
		xr->vk.ssao.kernel_ubo.set
	};

	xr->vk.table.vkCmdBindDescriptorSets(xr->vk.barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.ssao.pipeline_layout,
		0, MACRO_ARRAY_LEN(sets), sets, 0, NULL);

	xr->vk.table.vkCmdDraw(xr->vk.barrier->command_buffer, 6, 1, 0, 0);

	xr->vk.table.vkCmdEndRenderPass(xr->vk.barrier->command_buffer);
}


private void
xr_draw_ssao_blur(
	xr_t xr,
	vk_frame_t* frame,
	simulation_camera_t* camera,
	simulation_vr_transform_t* vr_transform,
	vk_entities_per_model_t* entity_data
	)
{
	assert_not_null(xr);
	assert_not_null(frame);

	VkRenderPassBeginInfo render_pass_begin_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.pNext = NULL,
		.renderPass = xr->vk.ssao_blur.render_pass,
		.framebuffer = frame->ssao_blur.framebuffer,
		.renderArea = xr->vk.ssao_extent.scissor,
		.clearValueCount = 0,
		.pClearValues = NULL
	};

	xr->vk.table.vkCmdBeginRenderPass(xr->vk.barrier->command_buffer,
		&render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

	xr->vk.table.vkCmdSetViewport(xr->vk.barrier->command_buffer, 0, 1, &xr->vk.ssao_extent.viewport);
	xr->vk.table.vkCmdSetScissor(xr->vk.barrier->command_buffer, 0, 1, &xr->vk.ssao_extent.scissor);

	xr->vk.table.vkCmdBindPipeline(xr->vk.barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.ssao_blur.pipeline);

	VkDescriptorSet sets[] =
	{
		frame->scene.set,
		frame->ssao.map.set
	};

	xr->vk.table.vkCmdBindDescriptorSets(xr->vk.barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.ssao_blur.pipeline_layout,
		0, MACRO_ARRAY_LEN(sets), sets, 0, NULL);

	xr->vk.table.vkCmdDraw(xr->vk.barrier->command_buffer, 6, 1, 0, 0);

	xr->vk.table.vkCmdEndRenderPass(xr->vk.barrier->command_buffer);
}


private void
xr_draw_output(
	xr_t xr,
	vk_frame_t* frame,
	simulation_camera_t* camera,
	simulation_vr_transform_t* vr_transform,
	vk_entities_per_model_t* entity_data
	)
{
	assert_not_null(xr);
	assert_not_null(frame);

	VkRenderPassBeginInfo render_pass_begin_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.pNext = NULL,
		.renderPass = xr->vk.output.render_pass,
		.framebuffer = frame->output.framebuffer,
		.renderArea = xr->vk.screen_extent.scissor,
		.clearValueCount = 0,
		.pClearValues = NULL
	};

	xr->vk.table.vkCmdBeginRenderPass(xr->vk.barrier->command_buffer,
		&render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

	xr->vk.table.vkCmdSetViewport(xr->vk.barrier->command_buffer, 0, 1, &xr->vk.screen_extent.viewport);
	xr->vk.table.vkCmdSetScissor(xr->vk.barrier->command_buffer, 0, 1, &xr->vk.screen_extent.scissor);


	switch(xr->options.preview)
	{

	case VK_PREVIEW_NONE:
	{
		vk_skybox_constant_data_t skybox_constant_data;

		glm_mat4_copy(vr_transform->projection[0], skybox_constant_data.transform[0]);
		mat4 view_left;
		glm_mat4_copy(vr_transform->view[0], view_left);
		view_left[3][0] = 0.0f;
		view_left[3][1] = 0.0f;
		view_left[3][2] = 0.0f;
		glm_mat4_mul(skybox_constant_data.transform[0], view_left, skybox_constant_data.transform[0]);

		glm_mat4_copy(vr_transform->projection[1], skybox_constant_data.transform[1]);
		mat4 view_right;
		glm_mat4_copy(vr_transform->view[1], view_right);
		view_right[3][0] = 0.0f;
		view_right[3][1] = 0.0f;
		view_right[3][2] = 0.0f;
		glm_mat4_mul(skybox_constant_data.transform[1], view_right, skybox_constant_data.transform[1]);

		xr->vk.table.vkCmdBindPipeline(xr->vk.barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.output.skybox.pipeline);

		xr->vk.table.vkCmdPushConstants(xr->vk.barrier->command_buffer,
			xr->vk.output.skybox.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
			0, sizeof(skybox_constant_data), &skybox_constant_data);

		xr->vk.table.vkCmdBindVertexBuffers(xr->vk.barrier->command_buffer,
			0, 1, &xr->vk.output.skybox.vertex_buffer.buffer, (VkDeviceSize[]){0});

		xr->vk.table.vkCmdBindIndexBuffer(xr->vk.barrier->command_buffer,
			xr->vk.output.skybox.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT16);

		xr->vk.table.vkCmdBindDescriptorSets(xr->vk.barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.output.skybox.pipeline_layout,
			0, 1, &xr->vk.output.skybox.sky.set, 0, NULL);

		xr->vk.table.vkCmdDrawIndexed(xr->vk.barrier->command_buffer,
			MACRO_ARRAY_LEN(xr_vk_skybox_index_data), 1, 0, 0, 0);


		xr->vk.table.vkCmdBindPipeline(xr->vk.barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.output.compose.pipeline);

		VkDescriptorSet sets[] =
		{
			frame->scene.map.set,
			frame->ssao_blur.map.set
		};

		xr->vk.table.vkCmdBindDescriptorSets(xr->vk.barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.output.compose.pipeline_layout,
			0, MACRO_ARRAY_LEN(sets), sets, 0, NULL);

		xr->vk.table.vkCmdDraw(xr->vk.barrier->command_buffer, 6, 1, 0, 0);

		break;
	}

	case VK_PREVIEW_SHADOW_MAP:
	{
		xr->vk.table.vkCmdBindPipeline(xr->vk.barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.output.preview.depth.pipeline);

		xr->vk.table.vkCmdBindDescriptorSets(xr->vk.barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.output.preview.depth.pipeline_layout,
			0, 1, &frame->shadow.map.set, 0, NULL);

		xr->vk.table.vkCmdDraw(xr->vk.barrier->command_buffer, 6, 1, 0, 0);

		break;
	}

	case VK_PREVIEW_SCENE_POSITION_MAP:
	{
		xr->vk.table.vkCmdBindPipeline(xr->vk.barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.output.preview.image.pipeline);

		xr->vk.table.vkCmdBindDescriptorSets(xr->vk.barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.output.preview.image.pipeline_layout,
			0, 1, &frame->scene.position.set, 0, NULL);

		xr->vk.table.vkCmdDraw(xr->vk.barrier->command_buffer, 6, 1, 0, 0);

		break;
	}

	case VK_PREVIEW_SCENE_NORMAL_MAP:
	{
		xr->vk.table.vkCmdBindPipeline(xr->vk.barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.output.preview.image.pipeline);

		xr->vk.table.vkCmdBindDescriptorSets(xr->vk.barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.output.preview.image.pipeline_layout,
			0, 1, &frame->scene.normal.set, 0, NULL);

		xr->vk.table.vkCmdDraw(xr->vk.barrier->command_buffer, 6, 1, 0, 0);

		break;
	}

	case VK_PREVIEW_SCENE_MAP:
	{
		xr->vk.table.vkCmdBindPipeline(xr->vk.barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.output.preview.image.pipeline);

		xr->vk.table.vkCmdBindDescriptorSets(xr->vk.barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.output.preview.image.pipeline_layout,
			0, 1, &frame->scene.map.set, 0, NULL);

		xr->vk.table.vkCmdDraw(xr->vk.barrier->command_buffer, 6, 1, 0, 0);

		break;
	}

	case VK_PREVIEW_SSAO_MAP:
	{
		xr->vk.table.vkCmdBindPipeline(xr->vk.barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.output.preview.image.pipeline);

		xr->vk.table.vkCmdBindDescriptorSets(xr->vk.barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.output.preview.image.pipeline_layout,
			0, 1, &frame->ssao.map.set, 0, NULL);

		xr->vk.table.vkCmdDraw(xr->vk.barrier->command_buffer, 6, 1, 0, 0);

		break;
	}

	case VK_PREVIEW_SSAO_BLUR_MAP:
	{
		xr->vk.table.vkCmdBindPipeline(xr->vk.barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.output.preview.image.pipeline);

		xr->vk.table.vkCmdBindDescriptorSets(xr->vk.barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, xr->vk.output.preview.image.pipeline_layout,
			0, 1, &frame->ssao_blur.map.set, 0, NULL);

		xr->vk.table.vkCmdDraw(xr->vk.barrier->command_buffer, 6, 1, 0, 0);

		break;
	}

	default: assert_unreachable();

	}


	xr->vk.table.vkCmdEndRenderPass(xr->vk.barrier->command_buffer);
}


private void
xr_draw(
	xr_t xr,
	XrView views[2]
	)
{
	assert_not_null(xr);

	VkResult result = xr->vk.table.vkWaitForFences(xr->vk.device, 1, &xr->vk.barrier->fence, VK_TRUE, UINT64_MAX);
	hard_assert_eq(result, VK_SUCCESS);

	uint64_t frame_time = 0;
	if(xr->vk.barrier->timing.current)
	{
		xr_timing_load(xr, &xr->vk.barrier->timing);

		uint64_t shadow_time = xr_timing_get(xr, &xr->vk.barrier->timing, VK_BARRIER_TIMING_IDX_SHADOW);
		stats_log(xr->stats, "xr_barrier_timing_shadow", shadow_time);
		frame_time += shadow_time;

		uint64_t scene_time = xr_timing_get(xr, &xr->vk.barrier->timing, VK_BARRIER_TIMING_IDX_SCENE);
		stats_log(xr->stats, "xr_barrier_timing_scene", scene_time);
		frame_time += scene_time;

		uint64_t ssao_time = xr_timing_get(xr, &xr->vk.barrier->timing, VK_BARRIER_TIMING_IDX_SSAO);
		stats_log(xr->stats, "xr_barrier_timing_ssao", ssao_time);
		frame_time += ssao_time;

		uint64_t ssao_blur_time = xr_timing_get(xr, &xr->vk.barrier->timing, VK_BARRIER_TIMING_IDX_SSAO_BLUR);
		stats_log(xr->stats, "xr_barrier_timing_ssao_blur", ssao_blur_time);
		frame_time += ssao_blur_time;

		uint64_t output_time = xr_timing_get(xr, &xr->vk.barrier->timing, VK_BARRIER_TIMING_IDX_OUTPUT);
		stats_log(xr->stats, "xr_barrier_timing_output", output_time);
		frame_time += output_time;
	}

	uint32_t image_idx;
	XrSwapchainImageAcquireInfo acquire_info = {XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
	XrResult xr_result = xrAcquireSwapchainImage(xr->vk.swapchain, &acquire_info, &image_idx);
	hard_assert_eq(xr_result, XR_SUCCESS);

	XrSwapchainImageWaitInfo wait_info =
	{
		.type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO,
		.timeout = XR_INFINITE_DURATION
	};
	xr_result = xrWaitSwapchainImage(xr->vk.swapchain, &wait_info);
	hard_assert_eq(xr_result, XR_SUCCESS);

	vk_frame_t* frame = xr->vk.frames + image_idx;

	if(frame->barrier)
	{
		result = xr->vk.table.vkWaitForFences(xr->vk.device, 1, &frame->barrier->fence, VK_TRUE, UINT64_MAX);
		hard_assert_eq(result, VK_SUCCESS);
	}
	frame->barrier = xr->vk.barrier;

	result = xr->vk.table.vkResetFences(xr->vk.device, 1, &xr->vk.barrier->fence);
	hard_assert_eq(result, VK_SUCCESS);



	simulation_update(xr->simulation);
	uint64_t start_time = time_get();

	result = xr->vk.table.vkResetCommandBuffer(xr->vk.barrier->command_buffer, 0);
	hard_assert_eq(result, VK_SUCCESS);

	VkCommandBufferBeginInfo command_buffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = NULL,
		.flags = 0,
		.pInheritanceInfo = NULL
	};

	result = xr->vk.table.vkBeginCommandBuffer(xr->vk.barrier->command_buffer, &command_buffer_info);
	hard_assert_eq(result, VK_SUCCESS);

	simulation_camera_t camera = simulation_get_camera(xr->simulation);

	xr_pose_t left_eye, right_eye;
	xr_get_eye_poses(xr, &left_eye, &right_eye, views);

	simulation_vr_transform_t vr_transform = simulation_get_vr_transform(xr->simulation,
		xr->vk.screen_extent.pair, *(simulation_eye_pose_t*) &left_eye, *(simulation_eye_pose_t*) &right_eye);

	uint32_t sim_entity_count;
	simulation_entity_data_t* sim_entity_data = simulation_get_entity_data(xr->simulation, &sim_entity_count);

	simulation_entity_data_t* sim_entity = sim_entity_data;
	simulation_entity_data_t* sim_entity_end = sim_entity + sim_entity_count;

	vk_entities_per_model_t* entity_data = alloc_calloc(sizeof(*entity_data) * xr->vk.model_count);
	assert_ptr(entity_data, sizeof(*entity_data) * xr->vk.model_count);

	while(sim_entity < sim_entity_end)
	{
		vk_entities_per_model_t* entities = entity_data + sim_entity->model_idx;

		if(entities->entities_used >= entities->entities_size)
		{
			uint32_t new_size = (entities->entities_size << 1) | 1;

			entities->entities = alloc_recalloc(
				entities->entities,
				sizeof(*entities->entities) * entities->entities_size,
				sizeof(*entities->entities) * new_size
				);
			assert_ptr(entities->entities, sizeof(*entities->entities) * new_size);

			entities->entities_size = new_size;
		}

		entities->entities[entities->entities_used++] = sim_entity;

		++sim_entity;
	}

	XR_FOR_EACH_MODEL(entities_per_model, model)
	{
		simulation_entity_data_t** entity = entities_per_model->entities;
		simulation_entity_data_t** entity_end = entity + entities_per_model->entities_used;

		uint64_t instance_data_size = sizeof(vk_model_instance_data_t) * entities_per_model->entities_used;
		vk_model_instance_data_t* instance_data = alloc_malloc(instance_data_size);
		assert_ptr(instance_data, instance_data_size);

		vk_model_instance_data_t* instance = instance_data;

		while(entity < entity_end)
		{
			glm_mat4_copy((*entity)->transform, instance->transform);

			++entity;
			++instance;
		}

		xr_copy_to_buffer(xr, &model->instance_buffer, instance_data, instance_data_size);
		alloc_free(instance_data, instance_data_size);
	}
	XR_FOR_EACH_MODEL_END(entities_per_model, model);

	xr_timing_reset(xr, &xr->vk.barrier->timing);

	xr_timing_start(xr, &xr->vk.barrier->timing, VK_BARRIER_TIMING_IDX_SHADOW);
	xr_draw_shadow(xr, frame, &camera, &vr_transform, entity_data);
	xr_timing_end(xr, &xr->vk.barrier->timing, VK_BARRIER_TIMING_IDX_SHADOW);

	xr_timing_start(xr, &xr->vk.barrier->timing, VK_BARRIER_TIMING_IDX_SCENE);
	xr_draw_scene(xr, frame, &camera, &vr_transform, entity_data);
	xr_timing_end(xr, &xr->vk.barrier->timing, VK_BARRIER_TIMING_IDX_SCENE);

	xr_timing_start(xr, &xr->vk.barrier->timing, VK_BARRIER_TIMING_IDX_SSAO);
	xr_draw_ssao(xr, frame, &camera, &vr_transform, entity_data);
	xr_timing_end(xr, &xr->vk.barrier->timing, VK_BARRIER_TIMING_IDX_SSAO);

	xr_timing_start(xr, &xr->vk.barrier->timing, VK_BARRIER_TIMING_IDX_SSAO_BLUR);
	xr_draw_ssao_blur(xr, frame, &camera, &vr_transform, entity_data);
	xr_timing_end(xr, &xr->vk.barrier->timing, VK_BARRIER_TIMING_IDX_SSAO_BLUR);

	xr_timing_start(xr, &xr->vk.barrier->timing, VK_BARRIER_TIMING_IDX_OUTPUT);
	xr_draw_output(xr, frame, &camera, &vr_transform, entity_data);
	xr_timing_end(xr, &xr->vk.barrier->timing, VK_BARRIER_TIMING_IDX_OUTPUT);

	xr_timing_query(xr, &xr->vk.barrier->timing);

	XR_FOR_EACH_MODEL(entities_per_model)
	{
		alloc_free(entities_per_model->entities,
			sizeof(*entities_per_model->entities) * entities_per_model->entities_size);
	}
	XR_FOR_EACH_MODEL_END(entities_per_model);

	alloc_free(entity_data, sizeof(*entity_data) * xr->vk.model_count);

	simulation_free_entity_data(sim_entity_data, sim_entity_count);

	result = xr->vk.table.vkEndCommandBuffer(xr->vk.barrier->command_buffer);
	hard_assert_eq(result, VK_SUCCESS);


	VkSubmitInfo submit_info =
	{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = NULL,
		.waitSemaphoreCount = 0,
		.pWaitSemaphores = NULL,
		.pWaitDstStageMask = NULL,
		.commandBufferCount = 1,
		.pCommandBuffers = &xr->vk.barrier->command_buffer,
		.signalSemaphoreCount = 0,
		.pSignalSemaphores = NULL
	};

	result = xr->vk.table.vkQueueSubmit(xr->vk.queue, 1, &submit_info, xr->vk.barrier->fence);
	hard_assert_eq(result, VK_SUCCESS);

	XrSwapchainImageReleaseInfo release_info = {XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
	xr_result = xrReleaseSwapchainImage(xr->vk.swapchain, &release_info);
	hard_assert_eq(xr_result, XR_SUCCESS);

	uint64_t end_time = time_get();
	stats_log(xr->stats, "xr_command_record_time", end_time - start_time);

	frame_time += end_time - start_time;
	stats_log(xr->stats, "xr_frame_time", frame_time);

	++xr->vk.barrier;
	if(xr->vk.barrier >= xr->vk.barriers + VK_MAX_FRAMES)
	{
		xr->vk.barrier = xr->vk.barriers;
	}
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

	XrResult xr_result = xrCreateSession(xr->instance, &xr_session_info, &xr->session);
	assert_eq(xr_result, XR_SUCCESS);

	XrReferenceSpaceCreateInfo xr_space_info =
	{
		.type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO,
		.next = NULL,
		.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE,
		.poseInReferenceSpace =
		{
			.orientation = {0.0f, 0.0f, 0.0f, 1.0f},
			.position = {0.0f, 0.0f, 0.0f}
		}
	};

	xr_result = xrCreateReferenceSpace(xr->session, &xr_space_info, &xr->space);
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
xr_init_xr_views(
	xr_t xr
	)
{
	assert_not_null(xr);

	XrViewConfigurationType view_type = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;

	uint32_t view_count = 0;
	XrResult result = xrEnumerateViewConfigurationViews(xr->instance, xr->system, view_type, 0, &view_count, NULL);
	hard_assert_eq(result, XR_SUCCESS);
	hard_assert_gt(view_count, 0);

	printf("\nXR view count: %u\n", view_count);

	xr->view_count = view_count;
	xr->view_configs = alloc_calloc(sizeof(*xr->view_configs) * view_count);
	assert_ptr(xr->view_configs, sizeof(*xr->view_configs) * view_count);

	for(uint32_t i = 0; i < view_count; ++i)
	{
		xr->view_configs[i] = (XrViewConfigurationView){XR_TYPE_VIEW_CONFIGURATION_VIEW};
	}

	result = xrEnumerateViewConfigurationViews(xr->instance, xr->system, view_type, view_count, &view_count, xr->view_configs);
	hard_assert_eq(result, XR_SUCCESS);

	for(uint32_t i = 0; i < view_count; ++i)
	{
		printf(
			"XR view %u:\n"
			"\trecommendedImageRectWidth: %u\n"
			"\tmaxImageRectWidth: %u\n"
			"\trecommendedImageRectHeight: %u\n"
			"\tmaxImageRectHeight: %u\n"
			"\trecommendedSwapchainSampleCount: %u\n"
			"\tmaxSwapchainSampleCount: %u\n",
			i,
			xr->view_configs[i].recommendedImageRectWidth,
			xr->view_configs[i].maxImageRectWidth,
			xr->view_configs[i].recommendedImageRectHeight,
			xr->view_configs[i].maxImageRectHeight,
			xr->view_configs[i].recommendedSwapchainSampleCount,
			xr->view_configs[i].maxSwapchainSampleCount
			);
	}
}


private void
xr_free_xr_views(
	xr_t xr
	)
{
	assert_not_null(xr);

	alloc_free(xr->view_configs, sizeof(*xr->view_configs) * xr->view_count);
}


private void
xr_init_xr_hand_tracking(
	xr_t xr
	)
{
	assert_not_null(xr);

	if(xr->options.xr_monado)
	{
		xr->hand_tracker_left = XR_NULL_HANDLE;
		xr->hand_tracker_right = XR_NULL_HANDLE;
		xr->xrLocateHandJointsEXT = NULL;
		xr->hand_joints_left_active = false;
		xr->hand_joints_right_active = false;
		return;
	}

	xr->xrLocateHandJointsEXT = xr_xr_load_func(xr, "xrLocateHandJointsEXT");
	PFN_xrCreateHandTrackerEXT xrCreateHandTrackerEXT = xr_xr_load_func(xr, "xrCreateHandTrackerEXT");

	XrHandTrackerCreateInfoEXT hand_tracker_info =
	{
		.type = XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT,
		.next = NULL,
		.handJointSet = XR_HAND_JOINT_SET_DEFAULT_EXT
	};

	hand_tracker_info.hand = XR_HAND_LEFT_EXT;
	XrResult result = xrCreateHandTrackerEXT(xr->session, &hand_tracker_info, &xr->hand_tracker_left);
	hard_assert_eq(result, XR_SUCCESS);

	hand_tracker_info.hand = XR_HAND_RIGHT_EXT;
	result = xrCreateHandTrackerEXT(xr->session, &hand_tracker_info, &xr->hand_tracker_right);
	hard_assert_eq(result, XR_SUCCESS);

	xr->hand_joints_left_active = false;
	xr->hand_joints_right_active = false;
}


private void
xr_free_xr_hand_tracking(
	xr_t xr
	)
{
	assert_not_null(xr);

	if(xr->options.xr_monado)
	{
		return;
	}

	PFN_xrDestroyHandTrackerEXT xrDestroyHandTrackerEXT = xr_xr_load_func(xr, "xrDestroyHandTrackerEXT");

	XrResult result = xrDestroyHandTrackerEXT(xr->hand_tracker_left);
	hard_assert_eq(result, XR_SUCCESS);

	result = xrDestroyHandTrackerEXT(xr->hand_tracker_right);
	hard_assert_eq(result, XR_SUCCESS);
}


private void
xr_poll_events(
	xr_t xr
	)
{
	assert_not_null(xr);

	while(true)
	{
		XrEventDataBuffer event_buffer = {XR_TYPE_EVENT_DATA_BUFFER};
		XrResult result = xrPollEvent(xr->instance, &event_buffer);
		if(result == XR_EVENT_UNAVAILABLE)
		{
			return;
		}

		switch(result)
		{
			case XR_SUCCESS: break;

			case XR_ERROR_INSTANCE_LOST:
			case XR_ERROR_SESSION_LOST:
			case XR_ERROR_RUNTIME_FAILURE:
			case XR_ERROR_VALIDATION_FAILURE:
			{
				printf("XR error %d, shutting down\n", result);
				atomic_store_rel(&is_running, false);
				return;
			}

			default: hard_assert_unreachable();
		}

		switch(event_buffer.type)
		{
			case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED:
			{
				XrEventDataSessionStateChanged* state_changed =
					(XrEventDataSessionStateChanged*) &event_buffer;

				xr->state = state_changed->state;

				switch(xr->state)
				{
					case XR_SESSION_STATE_READY:
					{
						puts("XR session ready");

						XrSessionBeginInfo begin_info =
						{
							.type = XR_TYPE_SESSION_BEGIN_INFO,
							.next = NULL,
							.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO
						};

						result = xrBeginSession(xr->session, &begin_info);
						hard_assert_eq(result, XR_SUCCESS);

						break;
					}

					case XR_SESSION_STATE_SYNCHRONIZED:
					{
						puts("XR session synchronized");
						break;
					}

					case XR_SESSION_STATE_STOPPING:
					{
						puts("XR session stopping");

						result = xrEndSession(xr->session);
						hard_assert_eq(result, XR_SUCCESS);

						break;
					}

					case XR_SESSION_STATE_EXITING:
					case XR_SESSION_STATE_LOSS_PENDING:
					{
						puts("XR session exiting or loss pending");
						atomic_store_rel(&is_running, false);
						break;
					}

					default: break;
				}

				break;
			}

			case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
			{
				puts("XR instance loss pending");
				atomic_store_rel(&is_running, false);
				break;
			}

			default:
			{
				printf("XR unknown event type %d\n", event_buffer.type);
				break;
			}
		}
	}
}


private void
xr_process_frame(
	xr_t xr
	)
{
	assert_not_null(xr);

	static XrTime previous_display_time = 0;

	XrFrameWaitInfo wait_info = {XR_TYPE_FRAME_WAIT_INFO};
	XrFrameState frame_state = {XR_TYPE_FRAME_STATE};
	XrResult result = xrWaitFrame(xr->session, &wait_info, &frame_state);
	hard_assert_eq(result, XR_SUCCESS);

	XrFrameBeginInfo begin_info = {XR_TYPE_FRAME_BEGIN_INFO};
	result = xrBeginFrame(xr->session, &begin_info);
	hard_assert_eq(result, XR_SUCCESS);

	xr->predicted_display_time = frame_state.predictedDisplayTime;

	if(previous_display_time != 0)
	{
		stats_log(xr->stats, "xr_frame_delta_time", xr->predicted_display_time - previous_display_time);
	}
	previous_display_time = xr->predicted_display_time;

	xr_update_hand_tracking(xr);

	XrCompositionLayerProjection projection_layer =
	{
		.type = XR_TYPE_COMPOSITION_LAYER_PROJECTION,
		.next = NULL,
		.layerFlags = 0,
		.space = xr->space,
		.viewCount = 0,
		.views = NULL
	};

	const XrCompositionLayerBaseHeader* const_layers[] =
	{
		(void*) &projection_layer
	};

	XrView views[2];

	if(frame_state.shouldRender)
	{
		xr_pose_t left_eye, right_eye;
		xr_get_eye_poses(xr, &left_eye, &right_eye, views);

		xr_draw(xr, views);

		XrCompositionLayerProjectionView projection_views[2] =
		{
			{
				.type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW,
				.next = NULL,
				.pose = views[0].pose,
				.fov = views[0].fov,
				.subImage =
				{
					.swapchain = xr->vk.swapchain,
					.imageRect =
					{
						.offset = {0, 0},
						.extent = xr->vk.screen_extent.xr_extent
					},
					.imageArrayIndex = 0
				}
			},
			{
				.type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW,
				.next = NULL,
				.pose = views[1].pose,
				.fov = views[1].fov,
				.subImage =
				{
					.swapchain = xr->vk.swapchain,
					.imageRect =
					{
						.offset = {0, 0},
						.extent = xr->vk.screen_extent.xr_extent
					},
					.imageArrayIndex = 1
				}
			}
		};

		projection_layer.viewCount = 2;
		projection_layer.views = projection_views;
	}

	XrFrameEndInfo end_info =
	{
		.type = XR_TYPE_FRAME_END_INFO,
		.displayTime = frame_state.predictedDisplayTime,
		.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE,
		.layerCount = frame_state.shouldRender ? 1 : 0,
		.layers = frame_state.shouldRender ? const_layers : NULL
	};

	result = xrEndFrame(xr->session, &end_info);
	hard_assert_eq(result, XR_SUCCESS);
}


private void
xr_session_thread_fn(
	xr_t xr
	)
{
	assert_not_null(xr);

	sigset_t set;
	sigemptyset(&set);
	sigaddset(&set, SIGINT);
	pthread_sigmask(SIG_UNBLOCK, &set, NULL);

	signal(SIGINT, xr_signal_handler);

	while(atomic_load_acq(&is_running))
	{
		xr_poll_events(xr);
		xr_process_frame(xr);
	}

	simulation_stop(xr->simulation);
}


private void
xr_init_xr_thread(
	xr_t xr
	)
{
	assert_not_null(xr);

	sigset_t set;
	sigemptyset(&set);
	sigaddset(&set, SIGINT);
	pthread_sigmask(SIG_BLOCK, &set, NULL);

	atomic_store_rel(&is_running, true);

	thread_data_t data =
	{
		.fn = (void*) xr_session_thread_fn,
		.data = xr
	};
	thread_init(&xr->thread, data);
}


private void
xr_free_xr_thread(
	xr_t xr
	)
{
	assert_not_null(xr);

	atomic_store_rel(&is_running, false);
	thread_join(xr->thread);

	thread_free(&xr->thread);
}


private void
xr_init_xr(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_init_stats(xr);
	xr_init_xr_instance(xr);
	xr_init_vk_instance(xr);
	xr_init_vk_physical_device(xr);
	xr_init_vk_device_properties(xr);
	xr_init_vk_logical_device(xr);
	xr_init_xr_views(xr);
	xr_init_xr_session(xr);
	xr_init_vk_capabilities(xr);
	xr_init_xr_swapchain(xr);
	xr_init_xr_hand_tracking(xr);
	xr_init_commands(xr);
	xr_init_sets(xr);
	xr_init_pipelines(xr);
	xr_init_models(xr);
	xr_init_frames(xr);
	xr_init_framebuffers(xr);

	xr_init_xr_thread(xr);

	puts("\nXR initialized");
}


private void
xr_free_xr(
	xr_t xr
	)
{
	assert_not_null(xr);

	xr_free_xr_thread(xr);

	xr_device_wait_idle(xr);

	xr_free_all_staging_buffers(xr);

	xr_free_framebuffers(xr);
	xr_free_frames(xr);
	xr_free_models(xr);
	xr_free_pipelines(xr);
	xr_free_sets(xr);
	xr_free_commands(xr);
	xr_free_xr_hand_tracking(xr);
	xr_free_xr_swapchain(xr);
	xr_free_xr_session(xr);
	xr_free_xr_views(xr);
	xr_free_vk_logical_device(xr);
	xr_free_vk_physical_device(xr);
	xr_free_vk_instance(xr);
	xr_free_xr_instance(xr);
	xr_free_stats(xr);
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

	if(!xr->options.xr_enable)
	{
		alloc_free(xr, sizeof(*xr));
		return NULL;
	}

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


