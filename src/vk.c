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
#include <thesis/time.h>
#include <thesis/debug.h>
#include <thesis/atomic.h>
#include <thesis/shared.h>
#include <thesis/window.h>
#include <thesis/options.h>
#include <thesis/threads.h>
#include <thesis/alloc_ext.h>

#include <volk.h>

#include <signal.h>
#include <string.h>

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
	VkExtent2D extent;
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

struct vk
{
	simulation_t simulation;
	stats_t stats;

	struct
	{
		bool window_enable;
		bool window_fullscreen;
		float window_width;
		float window_height;

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

	struct
	{
		event_listener_t* close_once_listener;
		event_listener_t* resize_listener;
		event_listener_t* mouse_down_listener;
		event_listener_t* mouse_up_listener;
		event_listener_t* mouse_move_listener;
		event_listener_t* key_down_listener;
		event_listener_t* key_up_listener;

		window_manager_t manager;
		window_t handle;
		thread_t thread;

		struct
		{
			_Atomic bool boolean;
			sync_mtx_t mtx;
			sync_cond_t cond;
		}
		resize;

		bool mouse_holding;
	}
	window;

	thread_t thread;
	_Atomic bool is_running;

	PFN_vkGetInstanceProcAddr proc_addr_fn;

#ifndef NDEBUG
	VkDebugUtilsMessengerEXT debug_messenger;
#endif

	VkInstance instance;

	VkSurfaceKHR surface;
	VkSurfaceCapabilitiesKHR surface_capabilities;

	uint32_t queue_id;
	VkFormat format;
	VkSampleCountFlagBits samples;
	float anisotropy;
	VkPhysicalDeviceLimits limits;
	float timestamp_period;
	bool timing_enabled;

	VkPhysicalDevice physical_device;
	VkDevice device;

	struct VolkDeviceTable table;

	VkQueue queue;
	VkPhysicalDeviceMemoryProperties memory_properties;

	vk_extent_t screen_extent;
	vk_extent_t shadow_extent;
	vk_extent_t ssao_extent;

	uint32_t image_count;
	VkSurfaceTransformFlagBitsKHR transform;
	VkPresentModeKHR present_mode;

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

	VkSwapchainKHR swapchain;
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
	"VK_LAYER_KHRONOS_validation",
#endif
};

private const char* vk_device_extensions[] =
{
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

private const char* vk_device_layers[] =
{
#ifndef NDEBUG
	"VK_LAYER_KHRONOS_validation",
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


private VkPipelineColorBlendAttachmentState vk_no_blending_attachment =
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


private VkPipelineColorBlendAttachmentState vk_blending_attachment =
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
vk_init_options(
	vk_t vk
	)
{
	assert_not_null(vk);

	puts("\nVK options:");

	vk->options.window_enable =
		options_get_boolean(global_options, "window_enable", true);
	printf("- window_enable: %d\n", vk->options.window_enable);

	vk->options.window_fullscreen =
		options_get_boolean(global_options, "window_fullscreen", true);
	printf("- window_fullscreen: %d\n", vk->options.window_fullscreen);

	vk->options.window_width =
		options_get_i64(global_options, "window_width", 1, 16384, VK_WINDOW_WIDTH);
	printf("- window_width: %.0f\n", vk->options.window_width);

	vk->options.window_height =
		options_get_i64(global_options, "window_height", 1, 16384, VK_WINDOW_HEIGHT);
	printf("- window_height: %.0f\n", vk->options.window_height);

	vk->options.max_msaa_samples =
		options_get_i64(global_options, "vk_max_msaa_samples", 1, 64, 8);
	printf("- max_msaa_samples: %u\n", vk->options.max_msaa_samples);

	vk->options.sample_shading =
		options_get_boolean(global_options, "vk_sample_shading", false);
	printf("- sample_shading: %d\n", vk->options.sample_shading);

	vk->options.min_sample_shading =
		options_get_f32(global_options, "vk_min_sample_shading", 0.0f, 1.0f, 0.2f);
	printf("- min_sample_shading: %.2f\n", vk->options.min_sample_shading);

	vk->options.mipmap_levels =
		options_get_i64(global_options, "vk_mipmap_levels", 0, 16, 3);
	printf("- mipmap_levels: %u\n", vk->options.mipmap_levels);

	vk->options.max_anisotropy =
		options_get_f32(global_options, "vk_max_anisotropy", 0.0f, 100.0f, 100.0f);
	printf("- max_anisotropy: %.1f\n", vk->options.max_anisotropy);

	vk->options.preview =
		options_get_i64(global_options, "vk_preview", 0, VK_PREVIEW__COUNT - 1, VK_PREVIEW_NONE);
	printf("- preview: %d\n", vk->options.preview);

	vk->options.shadow_map_size =
		options_get_i64(global_options, "vk_shadow_map_size", 1, 16384, 4096);
	printf("- shadow_map_size: %u\n", vk->options.shadow_map_size);

	vk->options.enable_depth_shadows =
		options_get_boolean(global_options, "vk_enable_depth_shadows", true);
	printf("- enable_depth_shadows: %d\n", vk->options.enable_depth_shadows);

	vk->options.enable_backface_shadows =
		options_get_boolean(global_options, "vk_enable_backface_shadows", true);
	printf("- enable_backface_shadows: %d\n", vk->options.enable_backface_shadows);

	vk->options.enable_specular =
		options_get_boolean(global_options, "vk_enable_specular", true);
	printf("- enable_specular: %d\n", vk->options.enable_specular);

	vk->options.shadow_value =
		options_get_f32(global_options, "vk_shadow_value", 0.0f, 1.0f, 0.2f);
	printf("- shadow_value: %.2f\n", vk->options.shadow_value);

	vk->options.lambert_start_angle =
		options_get_f32(global_options, "vk_lambert_start_angle", 0.0f, 90.0f, 80.0f);
	printf("- lambert_start_angle: %.1f\n", vk->options.lambert_start_angle);

	vk->options.enable_ssao =
		options_get_boolean(global_options, "vk_enable_ssao", true);
	printf("- enable_ssao: %d\n", vk->options.enable_ssao);

	vk->options.ssao_kernel_size =
		options_get_i64(global_options, "vk_ssao_kernel_size", 1, 256, 12);
	printf("- ssao_kernel_size: %u\n", vk->options.ssao_kernel_size);

	vk->options.ssao_noise_size =
		options_get_i64(global_options, "vk_ssao_noise_size", 1, 64, 4);
	printf("- ssao_noise_size: %u\n", vk->options.ssao_noise_size);

	vk->options.ssao_radius =
		options_get_f32(global_options, "vk_ssao_radius", 0.0f, 1024.0f, 96.0f);
	printf("- ssao_radius: %.2f\n", vk->options.ssao_radius);

	vk->options.ssao_bias =
		options_get_f32(global_options, "vk_ssao_bias", 0.0f, 1.0f, 0.1f);
	printf("- ssao_bias: %.3f\n", vk->options.ssao_bias);

	vk->options.ssao_power =
		options_get_f32(global_options, "vk_ssao_power", 0.0f, 16.0f, 6.0f);
	printf("- ssao_power: %.2f\n", vk->options.ssao_power);

	vk->options.ssao_range_check =
		options_get_f32(global_options, "vk_ssao_range_check", 0.0f, 16.0f, 4.0f);
	printf("- ssao_range_check: %.2f\n", vk->options.ssao_range_check);

	vk->options.ssao_depth_k =
		options_get_f32(global_options, "vk_ssao_depth_k", 0.0f, 1.0f, 0.06f);
	printf("- ssao_depth_k: %.4f\n", vk->options.ssao_depth_k);

	vk->options.ssao_depth_gamma =
		options_get_f32(global_options, "vk_ssao_depth_gamma", 0.0f, 64.0f, 16.0f);
	printf("- ssao_depth_gamma: %.2f\n", vk->options.ssao_depth_gamma);

	vk->options.ssao_debug =
		options_get_boolean(global_options, "vk_ssao_debug", false);
	printf("- ssao_debug: %d\n", vk->options.ssao_debug);

	vk->options.ssao_scale =
		options_get_f32(global_options, "vk_ssao_scale", 0.0f, 1.0f, 0.5f);
	printf("- ssao_scale: %.2f\n", vk->options.ssao_scale);

	vk->options.ssao_blur_radius =
		options_get_f32(global_options, "vk_ssao_blur_radius", 0.0f, 16.0f, 5.0f);
	printf("- ssao_blur_radius: %.2f\n", vk->options.ssao_blur_radius);

	vk->options.ssao_blur_falloff =
		options_get_f32(global_options, "vk_ssao_blur_falloff", 0.0f, 4.0f, 1.9f);
	printf("- ssao_blur_falloff: %.2f\n", vk->options.ssao_blur_falloff);

	vk->options.ssao_blur_depth_tolerance =
		options_get_f32(global_options, "vk_ssao_blur_depth_tolerance", 0.0f, 1024.0f, 256.0f);
	printf("- ssao_blur_depth_tolerance: %.2f\n", vk->options.ssao_blur_depth_tolerance);
}


private void
vk_init_stats(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->stats = simulation_get_stats(vk->simulation);

	stats_add(vk->stats, "vk_barrier_timing_shadow", VK_STATS_SIZE);
	stats_add(vk->stats, "vk_barrier_timing_scene", VK_STATS_SIZE);
	stats_add(vk->stats, "vk_barrier_timing_ssao", VK_STATS_SIZE);
	stats_add(vk->stats, "vk_barrier_timing_ssao_blur", VK_STATS_SIZE);
	stats_add(vk->stats, "vk_barrier_timing_output", VK_STATS_SIZE);
	stats_add(vk->stats, "vk_command_record_time", VK_STATS_SIZE);
	stats_add(vk->stats, "vk_frame_time", VK_STATS_SIZE);
}


private void
vk_free_stats(
	vk_t vk
	)
{
	assert_not_null(vk);

	stats_del(vk->stats, "vk_frame_time");
	stats_del(vk->stats, "vk_command_record_time");
	stats_del(vk->stats, "vk_barrier_timing_output");
	stats_del(vk->stats, "vk_barrier_timing_ssao_blur");
	stats_del(vk->stats, "vk_barrier_timing_ssao");
	stats_del(vk->stats, "vk_barrier_timing_scene");
	stats_del(vk->stats, "vk_barrier_timing_shadow");
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

	void* func = vk->proc_addr_fn(vk->instance, name);
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

	const char* const* instance_extension = vk_instance_extensions;
	const char* const* instance_extension_end = instance_extension + MACRO_ARRAY_LEN(vk_instance_extensions);

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

	uint32_t sdl_instance_extension_count = 0;
	const char* const* sdl_instance_extensions = window_get_vulkan_extensions(&sdl_instance_extension_count);
	hard_assert_ptr(sdl_instance_extensions, sdl_instance_extension_count);

	const char* const* sdl_instance_extension = sdl_instance_extensions;
	const char* const* sdl_instance_extension_end = sdl_instance_extension + sdl_instance_extension_count;

	while(sdl_instance_extension < sdl_instance_extension_end)
	{
		bool found = false;
		const char* extension_name = *(sdl_instance_extension++);

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

	const char* const* instance_layer = vk_instance_layers;
	const char* const* instance_layer_end = instance_layer + MACRO_ARRAY_LEN(vk_instance_layers);

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
vk_init_instance(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->proc_addr_fn = window_get_vulkan_proc_addr_fn();
	volkInitializeCustom(vk->proc_addr_fn);

	const char* instance_extensions[MAX_EXTENSIONS];
	const char** instance_extension = vk_get_instance_extensions(vk, instance_extensions);
	assert_lt(instance_extension, instance_extensions + MACRO_ARRAY_LEN(instance_extensions));

	const char* instance_layers[MAX_EXTENSIONS];
	const char** instance_layer = vk_get_instance_layers(vk, instance_layers);
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
		.pfnUserCallback = vk_debug_callback,
		.pUserData = NULL
	};

	instance_info.pNext = &debug_info;
#endif

	VkResult result = vkCreateInstance(&instance_info, NULL, &vk->instance);
	hard_assert_eq(result, VK_SUCCESS);


	shared_free_str_array(instance_extensions, instance_extension);
	shared_free_str_array(instance_layers, instance_layer);

#ifndef NDEBUG
	PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT =
		vk_load_func(vk, "vkCreateDebugUtilsMessengerEXT");

	result = vkCreateDebugUtilsMessengerEXT(
		vk->instance, &debug_info, NULL, &vk->debug_messenger);
	hard_assert_eq(result, VK_SUCCESS);
#endif

	volkLoadInstanceOnly(vk->instance);
}


private void
vk_free_instance(
	vk_t vk
	)
{
	assert_not_null(vk);

#ifndef NDEBUG
	vkDestroyDebugUtilsMessengerEXT(vk->instance, vk->debug_messenger, NULL);
#endif

	vkDestroyInstance(vk->instance, NULL);

	volkFinalize();
}


private void
vk_init_surface(
	vk_t vk
	)
{
	assert_not_null(vk);

	window_init_vulkan_surface(vk->window.handle, vk->instance, &vk->surface);
}


private void
vk_free_surface(
	vk_t vk
	)
{
	assert_not_null(vk);

	window_free_vulkan_surface(vk->instance, vk->surface);
}


typedef struct vk_device_score
{
	const char* name;
	uint32_t score;
	uint32_t queue_id;
	VkFormat format;
	VkSampleCountFlagBits samples;
	float anisotropy;
	VkPhysicalDeviceLimits limits;
	float timestamp_period;
	bool timing_enabled;
}
vk_device_score_t;


private bool
vk_get_device_features(
	vk_t vk,
	VkPhysicalDevice device,
	vk_device_score_t* device_score
	)
{
	VkPhysicalDeviceFeatures features;
	vkGetPhysicalDeviceFeatures(device, &features);

	if(vk->options.max_anisotropy && !features.samplerAnisotropy)
	{
		hard_assert_log();
		return false;
	}

	if(vk->options.sample_shading && !features.sampleRateShading)
	{
		hard_assert_log();
		return false;
	}

	VkFormatProperties format_properties;
	vkGetPhysicalDeviceFormatProperties(device, VK_FORMAT_R8G8B8A8_SRGB, &format_properties);

	if(
		vk->options.mipmap_levels &&
		!(
			format_properties.optimalTilingFeatures &
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
		VkResult result = vkGetPhysicalDeviceSurfaceSupportKHR(device, i, vk->surface, &present);
		if(result != VK_SUCCESS)
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

	uint32_t available_device_extension_count = 0;
	vkEnumerateDeviceExtensionProperties(device, NULL, &available_device_extension_count, NULL);
	if(available_device_extension_count == 0)
	{
		hard_assert_log();
		return false;
	}

	VkExtensionProperties available_device_extensions[available_device_extension_count];
	VkExtensionProperties* available_device_extension = available_device_extensions;
	VkExtensionProperties* available_device_extension_end =
		available_device_extension + available_device_extension_count;

	while(available_device_extension < available_device_extension_end)
	{
		*(available_device_extension++) = (VkExtensionProperties){0};
	}

	vkEnumerateDeviceExtensionProperties(device, NULL,
		&available_device_extension_count, available_device_extensions);

	puts("\nVK available device extensions:");

	for(
		available_device_extension = available_device_extensions;
		available_device_extension < available_device_extension_end;
		available_device_extension++
		)
	{
		printf("- %s\n", available_device_extension->extensionName);
	}

	puts("");

	const char* const* device_extension = vk_device_extensions;
	const char* const* device_extension_end = device_extension + MACRO_ARRAY_LEN(vk_device_extensions);

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

	uint32_t available_device_layer_count = 0;
	VkResult result = vkEnumerateDeviceLayerProperties(device, &available_device_layer_count, NULL);
	if(result != VK_SUCCESS || available_device_layer_count == 0)
	{
		hard_assert_log();
		return false;
	}

	VkLayerProperties available_device_layers[available_device_layer_count];
	result = vkEnumerateDeviceLayerProperties(device, &available_device_layer_count, available_device_layers);
	if(result != VK_SUCCESS)
	{
		hard_assert_log();
		return false;
	}

	puts("\nVK available device layers:");

	for(uint32_t i = 0; i < available_device_layer_count; ++i)
	{
		printf("- %s\n", available_device_layers[i].layerName);
	}

	puts("");

	const char* const* device_layer = vk_device_layers;
	const char* const* device_layer_end = device_layer + MACRO_ARRAY_LEN(vk_device_layers);

	while(device_layer < device_layer_end)
	{
		bool found = false;
		const char* layer_name = *(device_layer++);

		VkLayerProperties* layer = available_device_layers;
		VkLayerProperties* layer_end = layer + available_device_layer_count;

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
	VkResult result = vkGetPhysicalDeviceSurfaceFormatsKHR(device, vk->surface, &format_count, NULL);
	if(result != VK_SUCCESS || format_count == 0)
	{
		hard_assert_log();
		return false;
	}

	VkSurfaceFormatKHR formats[format_count];
	result = vkGetPhysicalDeviceSurfaceFormatsKHR(device, vk->surface, &format_count, formats);
	if(result != VK_SUCCESS)
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

	device_score->name = cstr_init(properties.deviceName);

	if(properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
	{
		device_score->score += 1000;
	}

	VkSampleCountFlags sample_count =
		properties.limits.framebufferColorSampleCounts &
		properties.limits.framebufferDepthSampleCounts;

	if(sample_count >= VK_SAMPLE_COUNT_64_BIT)
	{
		device_score->samples = VK_SAMPLE_COUNT_64_BIT;
	}
	else if(sample_count >= VK_SAMPLE_COUNT_32_BIT)
	{
		device_score->samples = VK_SAMPLE_COUNT_32_BIT;
	}
	else if(sample_count >= VK_SAMPLE_COUNT_16_BIT)
	{
		device_score->samples = VK_SAMPLE_COUNT_16_BIT;
	}
	if(sample_count >= VK_SAMPLE_COUNT_8_BIT)
	{
		device_score->samples = VK_SAMPLE_COUNT_8_BIT;
	}
	else if(sample_count >= VK_SAMPLE_COUNT_4_BIT)
	{
		device_score->samples = VK_SAMPLE_COUNT_4_BIT;
	}
	else if(sample_count >= VK_SAMPLE_COUNT_2_BIT)
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
			MACRO_MAX(sizeof(vk_skybox_constant_data_t), sizeof(vk_scene_frag_constant_data_t))
		)
	{
		hard_assert_log("%u\n", properties.limits.maxPushConstantsSize);
		return false;
	}

	if(properties.limits.maxBoundDescriptorSets < 8)
	{
		hard_assert_log("%u\n", properties.limits.maxBoundDescriptorSets);
		return false;
	}

	if(properties.limits.maxColorAttachments < 8)
	{
		hard_assert_log("%u\n", properties.limits.maxColorAttachments);
		return false;
	}

	device_score->score += properties.limits.maxImageDimension2D;
	device_score->limits = properties.limits;

	device_score->timestamp_period = properties.limits.timestampPeriod;
	device_score->timing_enabled = properties.limits.timestampComputeAndGraphics == VK_TRUE;

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


private void
vk_init_extent(
	vk_t vk,
	vk_extent_t* extent,
	uint32_t width,
	uint32_t height
	)
{
	assert_not_null(vk);
	assert_not_null(extent);

	width = MACRO_CLAMP(
		width,
		vk->surface_capabilities.minImageExtent.width,
		vk->surface_capabilities.maxImageExtent.width
		);

	height = MACRO_CLAMP(
		height,
		vk->surface_capabilities.minImageExtent.height,
		vk->surface_capabilities.maxImageExtent.height
		);

	extent->width = width;
	extent->height = height;

	extent->extent =
	(VkExtent2D)
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
vk_get_extent(
	vk_t vk
	)
{
	assert_not_null(vk);

	uint32_t width = 0;
	uint32_t height = 0;

	while(1)
	{
		VkResult result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
			vk->physical_device, vk->surface, &vk->surface_capabilities);
		hard_assert_eq(result, VK_SUCCESS);

		width = vk->surface_capabilities.currentExtent.width;
		height = vk->surface_capabilities.currentExtent.height;

		if(width != 0 && height != 0)
		{
			break;
		}

		sync_mtx_lock(&vk->window.resize.mtx);
			while(!atomic_load_acq(&vk->window.resize.boolean))
			{
				sync_cond_wait(&vk->window.resize.cond, &vk->window.resize.mtx);
			}

			atomic_store_rel(&vk->window.resize.boolean, false);
		sync_mtx_unlock(&vk->window.resize.mtx);
	}

	if(width == UINT32_MAX || height == UINT32_MAX)
	{
		window_info_t window_info;
		window_get_info(vk->window.handle, &window_info);

		width = window_info.extent.w;
		height = window_info.extent.h;
	}

	vk_init_extent(vk, &vk->screen_extent, width, height);
	vk_init_extent(vk, &vk->shadow_extent, vk->options.shadow_map_size, vk->options.shadow_map_size);
	vk_init_extent(vk, &vk->ssao_extent, width * vk->options.ssao_scale, height * vk->options.ssao_scale);
}


private void
vk_init_device(
	vk_t vk
	)
{
	assert_not_null(vk);

	uint32_t physical_device_count = 0;
	VkResult result = vkEnumeratePhysicalDevices(vk->instance, &physical_device_count, NULL);
	hard_assert_eq(result, VK_SUCCESS);
	hard_assert_neq(physical_device_count, 0);

	VkPhysicalDevice physical_devices[physical_device_count];
	VkPhysicalDevice* physical_device = physical_devices;
	VkPhysicalDevice* physical_device_end = physical_device + physical_device_count;

	result = vkEnumeratePhysicalDevices(vk->instance, &physical_device_count, physical_devices);
	hard_assert_eq(result, VK_SUCCESS);

	VkPhysicalDevice best_device = NULL;
	vk_device_score_t best_device_score = {0};

	printf("\nVK available physical devices (%u):\n", physical_device_count);

	do
	{
		vk_device_score_t this_device_score = vk_get_device_score(vk, *physical_device);

		printf(
			"\n= %s:\n"
			"\tscore: %u\n"
			"\tqueue_id: %u\n"
			"\tformat: %u\n"
			"\tsamples: %u\n"
			"\tanisotropy: %.1f\n"
			"\ttimestamp_period: %.3f\n"
			"\ttiming_enabled: %d\n",
			this_device_score.name,
			this_device_score.score,
			this_device_score.queue_id,
			this_device_score.format,
			this_device_score.samples,
			this_device_score.anisotropy,
			this_device_score.timestamp_period,
			this_device_score.timing_enabled
			);
		cstr_free(this_device_score.name);

		if(this_device_score.score > best_device_score.score)
		{
			best_device = *physical_device;
			best_device_score = this_device_score;
		}
	}
	while(++physical_device != physical_device_end);

	hard_assert_not_null(best_device);

	vk->queue_id = best_device_score.queue_id;
	vk->format = best_device_score.format;
	vk->samples = MACRO_MIN(vk->options.max_msaa_samples, best_device_score.samples);
	vk->anisotropy = MACRO_MIN(vk->options.max_anisotropy, best_device_score.anisotropy);
	vk->limits = best_device_score.limits;
	vk->timestamp_period = best_device_score.timestamp_period;
	vk->timing_enabled = best_device_score.timing_enabled;

	vk->physical_device = best_device;


	float priority = 1.0f;

	VkDeviceQueueCreateInfo device_queue_info =
	{
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.queueFamilyIndex = vk->queue_id,
		.queueCount = 1,
		.pQueuePriorities = &priority
	};

	VkPhysicalDeviceFeatures device_features =
	{
		.samplerAnisotropy = !!vk->anisotropy,
		.sampleRateShading = vk->options.sample_shading
	};

	VkDeviceCreateInfo device_info =
	{
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &device_queue_info,
		.enabledLayerCount = MACRO_ARRAY_LEN(vk_device_layers),
		.ppEnabledLayerNames = vk_device_layers,
		.enabledExtensionCount = MACRO_ARRAY_LEN(vk_device_extensions),
		.ppEnabledExtensionNames = vk_device_extensions,
		.pEnabledFeatures = &device_features
	};

	result = vkCreateDevice(best_device, &device_info, NULL, &vk->device);
	hard_assert_eq(result, VK_SUCCESS);


	volkLoadDeviceTable(&vk->table, vk->device);

	vk->table.vkGetDeviceQueue(vk->device, vk->queue_id, 0, &vk->queue);

	vkGetPhysicalDeviceMemoryProperties(vk->physical_device, &vk->memory_properties);


	vk_get_extent(vk);
	if(!vk->surface_capabilities.maxImageCount)
	{
		vk->surface_capabilities.maxImageCount = UINT32_MAX;
	}
	vk->image_count = MACRO_CLAMP(
		3,
		vk->surface_capabilities.minImageCount,
		vk->surface_capabilities.maxImageCount
		);
	vk->transform = vk->surface_capabilities.currentTransform;


	uint32_t present_mode_count;
	result = vkGetPhysicalDeviceSurfacePresentModesKHR(
		vk->physical_device, vk->surface, &present_mode_count, NULL);
	hard_assert_eq(result, VK_SUCCESS);
	hard_assert_neq(present_mode_count, 0);

	VkPresentModeKHR present_modes[present_mode_count];
	VkPresentModeKHR* present_mode = present_modes;
	VkPresentModeKHR* present_mode_end =
		present_mode + present_mode_count;

	result = vkGetPhysicalDeviceSurfacePresentModesKHR(
		vk->physical_device, vk->surface, &present_mode_count, present_modes);
	hard_assert_eq(result, VK_SUCCESS);

	while(1)
	{
		if(*present_mode == VK_PRESENT_MODE_FIFO_RELAXED_KHR)
		{
			vk->present_mode = *present_mode;
			break;
		}

		if(++present_mode == present_mode_end)
		{
			vk->present_mode = VK_PRESENT_MODE_FIFO_KHR;
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

	vk->table.vkDestroyDevice(vk->device, NULL);
}


private void
vk_init_commands(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkCommandPoolCreateInfo command_pool_info =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.pNext = NULL,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = vk->queue_id
	};

	VkResult result = vk->table.vkCreateCommandPool(vk->device, &command_pool_info, NULL, &vk->command_pool);
	hard_assert_eq(result, VK_SUCCESS);


	VkCommandBuffer command_buffers[MACRO_ARRAY_LEN(vk->commands)];
	VkCommandBuffer* command_buffer = command_buffers;

	VkCommandBufferAllocateInfo command_buffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = NULL,
		.commandPool = vk->command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = MACRO_ARRAY_LEN(command_buffers)
	};

	result = vk->table.vkAllocateCommandBuffers(vk->device, &command_buffer_info, command_buffers);
	hard_assert_eq(result, VK_SUCCESS);

	VkFenceCreateInfo fence_info =
	{
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.pNext = NULL,
		.flags = VK_FENCE_CREATE_SIGNALED_BIT
	};

	vk_command_t* command = vk->commands;
	vk_command_t* command_end = command + MACRO_ARRAY_LEN(vk->commands);

	while(command != command_end)
	{
		command->buffer = *command_buffer;

		result = vk->table.vkCreateFence(vk->device, &fence_info, NULL, &command->fence);
		hard_assert_eq(result, VK_SUCCESS);

		command->waited = false;

		++command_buffer;
		++command;
	}

	vk->command = vk->commands;
}


private void
vk_free_commands(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkCommandBuffer command_buffers[MACRO_ARRAY_LEN(vk->commands)];
	VkCommandBuffer* command_buffer = command_buffers;

	vk_command_t* command = vk->commands;
	vk_command_t* command_end = command + MACRO_ARRAY_LEN(vk->commands);

	while(command != command_end)
	{
		vk->table.vkDestroyFence(vk->device, command->fence, NULL);

		*command_buffer = command->buffer;

		++command_buffer;
		++command;
	}

	vk->table.vkFreeCommandBuffers(vk->device, vk->command_pool, MACRO_ARRAY_LEN(vk->commands), command_buffers);

	vk->table.vkDestroyCommandPool(vk->device, vk->command_pool, NULL);
}


private void
vk_wait_command(
	vk_t vk,
	vk_command_t* command
	)
{
	assert_not_null(vk);
	assert_not_null(command);

	if(command->waited)
	{
		return;
	}

	VkResult result = vk->table.vkWaitForFences(vk->device, 1, &command->fence, VK_TRUE, UINT64_MAX);
	hard_assert_eq(result, VK_SUCCESS);

	result = vk->table.vkResetFences(vk->device, 1, &command->fence);
	hard_assert_eq(result, VK_SUCCESS);

	command->waited = true;
}


private vk_command_t*
vk_get_command(
	vk_t vk
	)
{
	assert_not_null(vk);

	if(vk->command >= vk->commands + MACRO_ARRAY_LEN(vk->commands))
	{
		vk->command = vk->commands;
	}

	vk_command_t* command = vk->command;
	++vk->command;

	if(!command->waited)
	{
		vk_wait_command(vk, command);
	}

	VkResult result = vk->table.vkResetCommandBuffer(command->buffer, 0);
	hard_assert_eq(result, VK_SUCCESS);

	VkCommandBufferBeginInfo command_buffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = NULL,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		.pInheritanceInfo = NULL
	};

	result = vk->table.vkBeginCommandBuffer(command->buffer, &command_buffer_info);
	hard_assert_eq(result, VK_SUCCESS);

	return command;
}


private void
vk_run_command(
	vk_t vk,
	vk_command_t* command
	)
{
	assert_not_null(vk);
	assert_not_null(command);

	VkResult result = vk->table.vkEndCommandBuffer(command->buffer);
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

	result = vk->table.vkQueueSubmit(vk->queue, 1, &submit_info, command->fence);
	hard_assert_eq(result, VK_SUCCESS);

	command->waited = false;
}


private void
vk_descriptor_set_pool_add(
	vk_t vk,
	vk_descriptor_set_pool_t* set_pool
	)
{
	assert_not_null(vk);
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

	VkResult result = vk->table.vkCreateDescriptorPool(vk->device, &pool_info, NULL, &pool->pool);
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
vk_get_descriptor_sets(
	vk_t vk,
	vk_descriptor_set_layout_t dst_set_layout,
	VkDescriptorSet* sets,
	uint32_t count
	)
{
	assert_not_null(vk);
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
			vk_descriptor_set_pool_add(vk, set_pool);
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

		VkResult result = vk->table.vkAllocateDescriptorSets(vk->device, &alloc_info, sets);
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
vk_init_set_layout(
	vk_t vk,
	VkDescriptorPoolSize* sizes,
	uint32_t size_count,
	VkShaderStageFlags stage_flags,
	vk_descriptor_set_layout_t* set_layout
	)
{
	assert_not_null(vk);
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

	VkResult result = vk->table.vkCreateDescriptorSetLayout(
		vk->device, &set_layout_info, NULL, &set_layout->layout);
	hard_assert_eq(result, VK_SUCCESS);

	sizes = new_sizes;
	size_count = unique_types;


	vk_descriptor_set_pool_t** set_pool_ptr = vk->set_pools;
	vk_descriptor_set_pool_t** set_pool_ptr_end = set_pool_ptr + vk->set_pool_count;

	while(set_pool_ptr < set_pool_ptr_end)
	{
		vk_descriptor_set_pool_t* set_pool = *set_pool_ptr;

		if(set_pool->size_count != size_count)
		{
			++set_pool;
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


	vk->set_pools = alloc_remalloc(
		vk->set_pools,
		sizeof(*vk->set_pools) * vk->set_pool_count,
		sizeof(*vk->set_pools) * (vk->set_pool_count + 1)
		);
	assert_not_null(vk->set_pools);

	set_pool_ptr = vk->set_pools + vk->set_pool_count++;
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
vk_free_set_layout(
	vk_t vk,
	vk_descriptor_set_layout_t* set_layout
	)
{
	assert_not_null(vk);
	assert_not_null(set_layout);

	vk_descriptor_set_pool_t* set_pool = set_layout->set_pool;

	if(!--set_pool->refs)
	{
		vk_descriptor_pool_t* pool = set_pool->head;
		while(pool)
		{
			vk->table.vkDestroyDescriptorPool(vk->device, pool->pool, NULL);

			vk_descriptor_pool_t* next = pool->next;
			alloc_free(pool, sizeof(*pool));
			pool = next;
		}

		alloc_free(set_pool->sizes, sizeof(*set_pool->sizes) * set_pool->size_count);
	}

	vk->table.vkDestroyDescriptorSetLayout(vk->device, set_layout->layout, NULL);
}


private void
vk_init_vert_set_layout(
	vk_t vk,
	VkDescriptorPoolSize* sizes,
	uint32_t size_count,
	vk_descriptor_set_layout_t* set_layout
	)
{
	vk_init_set_layout(vk, sizes, size_count, VK_SHADER_STAGE_VERTEX_BIT, set_layout);
}


private void
vk_init_frag_set_layout(
	vk_t vk,
	VkDescriptorPoolSize* sizes,
	uint32_t size_count,
	vk_descriptor_set_layout_t* set_layout
	)
{
	vk_init_set_layout(vk, sizes, size_count, VK_SHADER_STAGE_FRAGMENT_BIT, set_layout);
}


private void
vk_init_set_layouts(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkDescriptorPoolSize sampler_sizes[] =
	{
		{
			.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1
		}
	};

	vk_init_frag_set_layout(vk, sampler_sizes, MACRO_ARRAY_LEN(sampler_sizes), &vk->sampler_set_layout);


	VkDescriptorPoolSize vert_ubo_sizes[] =
	{
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1
		}
	};

	vk_init_vert_set_layout(vk, vert_ubo_sizes, MACRO_ARRAY_LEN(vert_ubo_sizes), &vk->vert_ubo_set_layout);


	VkDescriptorPoolSize frag_ubo_sizes[] =
	{
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1
		}
	};

	vk_init_frag_set_layout(vk, frag_ubo_sizes, MACRO_ARRAY_LEN(frag_ubo_sizes), &vk->frag_ubo_set_layout);


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

	vk_init_frag_set_layout(vk, scene_sizes, MACRO_ARRAY_LEN(scene_sizes), &vk->scene_set_layout);
}


private void
vk_free_set_layouts(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_set_layout(vk, &vk->scene_set_layout);
	vk_free_set_layout(vk, &vk->frag_ubo_set_layout);
	vk_free_set_layout(vk, &vk->vert_ubo_set_layout);
	vk_free_set_layout(vk, &vk->sampler_set_layout);
}


private void
vk_init_sets(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_set_layouts(vk);
}


private void
vk_free_sets(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_set_layouts(vk);

	vk_descriptor_set_pool_t** set_pool_ptr = vk->set_pools;
	vk_descriptor_set_pool_t** set_pool_ptr_end = set_pool_ptr + vk->set_pool_count;

	while(set_pool_ptr < set_pool_ptr_end)
	{
		vk_descriptor_set_pool_t* set_pool = *set_pool_ptr;
		alloc_free(set_pool, sizeof(*set_pool));

		++set_pool_ptr;
	}

	alloc_free(vk->set_pools, sizeof(*vk->set_pools) * vk->set_pool_count);
}


private uint32_t
vk_get_memory(
	vk_t vk,
	uint32_t bits,
	VkMemoryPropertyFlags flags
	)
{
	for(uint32_t i = 0; i < vk->memory_properties.memoryTypeCount; ++i)
	{
		if(
			(bits & (1 << i)) &&
			(vk->memory_properties.memoryTypes[i].propertyFlags & flags) == flags
			)
		{
			return i;
		}
	}

	hard_assert_unreachable();
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

	VkResult result = vk->table.vkCreateBuffer(vk->device, &buffer_info, NULL, &buffer->buffer);
	hard_assert_eq(result, VK_SUCCESS);

	VkMemoryRequirements memory_requirements;
	vk->table.vkGetBufferMemoryRequirements(vk->device, buffer->buffer, &memory_requirements);

	uint32_t memory_type_index = vk_get_memory(vk, memory_requirements.memoryTypeBits, flags);

	VkMemoryAllocateInfo memory_info =
	{
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext = NULL,
		.allocationSize = memory_requirements.size,
		.memoryTypeIndex = memory_type_index
	};

	result = vk->table.vkAllocateMemory(vk->device, &memory_info, NULL, &buffer->memory);
	hard_assert_eq(result, VK_SUCCESS);

	result = vk->table.vkBindBufferMemory(vk->device, buffer->buffer, buffer->memory, 0);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
vk_free_buffer(
	vk_t vk,
	vk_buffer_t* buffer
	)
{
	assert_not_null(vk);

	vk->table.vkFreeMemory(vk->device, buffer->memory, NULL);
	vk->table.vkDestroyBuffer(vk->device, buffer->buffer, NULL);
}


private void
vk_init_staging_buffer(
	vk_t vk,
	vk_command_t* command,
	VkDeviceSize size
	)
{
	assert_not_null(vk);
	assert_not_null(command);

	VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	vk_init_buffer(vk, size, usage, flags, &command->staging_buffer);
}


private void
vk_free_staging_buffer(
	vk_t vk,
	vk_command_t* command
	)
{
	assert_not_null(vk);
	assert_not_null(command);

	vk_free_buffer(vk, &command->staging_buffer);
}


private void
vk_free_all_staging_buffers(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_command_t* command = vk->commands;
	vk_command_t* command_end = command + MACRO_ARRAY_LEN(vk->commands);

	while(command != command_end)
	{
		vk_free_staging_buffer(vk, command);

		++command;
	}
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

	VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	vk_init_buffer(vk, size, usage, flags, buffer);
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

	VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
	VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	vk_init_buffer(vk, size, usage, flags, buffer);
}


private void
vk_init_ubo_buffer(
	vk_t vk,
	VkDeviceSize size,
	vk_descriptor_set_layout_t set_layout,
	vk_frame_buffer_t* frame_buffer
	)
{
	assert_not_null(vk);
	assert_not_null(set_layout.layout);
	assert_not_null(frame_buffer);

	VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	vk_init_buffer(vk, size, usage, flags, &frame_buffer->buffer);

	if(!frame_buffer->set)
	{
		vk_get_descriptor_sets(vk, set_layout, &frame_buffer->set, 1);
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

	vk->table.vkUpdateDescriptorSets(vk->device, 1, &write_set, 0, NULL);
}


private void
vk_free_ubo_buffer(
	vk_t vk,
	vk_frame_buffer_t* frame_buffer
	)
{
	assert_not_null(vk);
	assert_not_null(frame_buffer);

	vk_free_buffer(vk, &frame_buffer->buffer);
}


private void
vk_init_vert_ubo_buffer(
	vk_t vk,
	VkDeviceSize size,
	vk_frame_buffer_t* frame_buffer
	)
{
	assert_not_null(vk);
	assert_not_null(frame_buffer);

	vk_init_ubo_buffer(vk, size, vk->vert_ubo_set_layout, frame_buffer);
}


private void
vk_init_frag_ubo_buffer(
	vk_t vk,
	VkDeviceSize size,
	vk_frame_buffer_t* frame_buffer
	)
{
	assert_not_null(vk);
	assert_not_null(frame_buffer);

	vk_init_ubo_buffer(vk, size, vk->frag_ubo_set_layout, frame_buffer);
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

	vk_command_t* command = vk_get_command(vk);

	vk_free_staging_buffer(vk, command);
	vk_init_staging_buffer(vk, command, size);

	void* mapped_data;
	VkResult result = vk->table.vkMapMemory(vk->device, command->staging_buffer.memory, 0, size, 0, &mapped_data);
	hard_assert_eq(result, VK_SUCCESS);

	memcpy(mapped_data, data, size);

	vk->table.vkUnmapMemory(vk->device, command->staging_buffer.memory);

	VkBufferCopy buffer_copy =
	{
		.srcOffset = 0,
		.dstOffset = 0,
		.size = size
	};

	vk->table.vkCmdCopyBuffer(command->buffer, command->staging_buffer.buffer, buffer->buffer, 1, &buffer_copy);

	vk_run_command(vk, command);
}


private void
vk_read_from_buffer(
	vk_t vk,
	vk_buffer_t* buffer,
	void* data,
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

	vk_command_t* command = vk_get_command(vk);

	vk_free_staging_buffer(vk, command);
	vk_init_staging_buffer(vk, command, size);

	VkBufferCopy buffer_copy =
	{
		.srcOffset = 0,
		.dstOffset = 0,
		.size = size
	};

	vk->table.vkCmdCopyBuffer(command->buffer, buffer->buffer, command->staging_buffer.buffer, 1, &buffer_copy);

	vk_run_command(vk, command);
	vk_wait_command(vk, command);

	void* mapped_data;
	VkResult result = vk->table.vkMapMemory(vk->device, command->staging_buffer.memory, 0, size, 0, &mapped_data);
	hard_assert_eq(result, VK_SUCCESS);

	memcpy(data, mapped_data, size);

	vk->table.vkUnmapMemory(vk->device, command->staging_buffer.memory);
}


private void
vk_copy_texture_to_image(
	vk_t vk,
	vk_image_t* image
	)
{
	assert_not_null(vk);
	assert_not_null(image);

	vk_command_t* command = vk_get_command(vk);

	vk_free_staging_buffer(vk, command);
	vk_init_staging_buffer(vk, command, image->size);

	void* mapped_data;
	VkResult result = vk->table.vkMapMemory(vk->device,
		command->staging_buffer.memory, 0, image->size, 0, &mapped_data);
	hard_assert_eq(result, VK_SUCCESS);

	memcpy(mapped_data, image->data, image->size);

	vk->table.vkUnmapMemory(vk->device, command->staging_buffer.memory);

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

	vk->table.vkCmdCopyBufferToImage(command->buffer, command->staging_buffer.buffer,
		image->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, count, buffer_image_copies);

	vk_run_command(vk, command);
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

	vk_command_t* command = vk_get_command(vk);

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

	vk->table.vkCmdPipelineBarrier(command->buffer, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1, &barrier);

	vk_run_command(vk, command);
}


private void
vk_copy_data_to_image(
	vk_t vk,
	vk_image_t* image
	)
{
	assert_not_null(vk);
	assert_not_null(image);

	vk_transition_image_layout(vk, image,
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

	vk_copy_texture_to_image(vk, image);

	vk_transition_image_layout(vk, image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}


private void
vk_init_image(
	vk_t vk,
	vk_image_t* image
	)
{
	assert_not_null(vk);
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
				image->format = vk->format;
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
		image->samples = vk->samples;
	}
	else
	{
		image->samples = VK_SAMPLE_COUNT_1_BIT;
	}

	if(!(image->type & VK_IMAGE_TYPE_CUSTOM_SIZE_BIT))
	{
		image->width = vk->screen_extent.width;
		image->height = vk->screen_extent.height;
	}

	bool image_backed_texture =
		(image->type & VK_IMAGE_TYPE_TEXTURE_BIT) && image->path && !str_is_empty(image->path);

	if(image->type & VK_IMAGE_TYPE_TEXTURE_BIT)
	{
		if(image_backed_texture)
		{
			assert_eq(image->type & VK_IMAGE_TYPE_CUSTOM_SIZE_BIT, 0);

			bool cube = image->type & VK_IMAGE_TYPE_CUBE_BIT;
			simulation_texture_t* texture = simulation_get_texture(vk->simulation, image->path, cube);
			image->data = texture->data;
			image->size = texture->size;
			image->width = texture->width;
			image->height = texture->height;
			image->levels = 1 + MACRO_MIN(
				MACRO_LOG2(MACRO_MAX(image->width, image->height)),
				vk->options.mipmap_levels
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
		image->layers = 1;
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

	VkResult result = vk->table.vkCreateImage(vk->device, &image_info, NULL, &image->image);
	hard_assert_eq(result, VK_SUCCESS);

	VkMemoryRequirements memory_requirements;
	vk->table.vkGetImageMemoryRequirements(vk->device, image->image, &memory_requirements);

	uint32_t memory_type_index = vk_get_memory(vk,
		memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	VkMemoryAllocateInfo memory_info =
	{
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext = NULL,
		.allocationSize = memory_requirements.size,
		.memoryTypeIndex = memory_type_index
	};

	result = vk->table.vkAllocateMemory(
		vk->device, &memory_info, NULL, &image->memory);
	hard_assert_eq(result, VK_SUCCESS);

	result = vk->table.vkBindImageMemory(
		vk->device, image->image, image->memory, 0);
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

	result = vk->table.vkCreateImageView(vk->device, &image_view_info, NULL, &image->view);
	hard_assert_eq(result, VK_SUCCESS);

	if(image_backed_texture)
	{
		vk_copy_data_to_image(vk, image);
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

	vk->table.vkDestroyImageView(vk->device, image->view, NULL);
	vk->table.vkDestroyImage(vk->device, image->image, NULL);
	vk->table.vkFreeMemory(vk->device, image->memory, NULL);
}


private void
vk_write_images_to_set(
	vk_t vk,
	VkDescriptorSet set,
	vk_image_t* images,
	uint32_t count
	)
{
	assert_not_null(vk);
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
			sampler = vk->depth_sampler;
		}
		else
		{
			layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			sampler = vk->image_sampler;
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

	vk->table.vkUpdateDescriptorSets(vk->device, count, write_sets, 0, NULL);
}


private void
vk_init_frame_image(
	vk_t vk,
	vk_frame_image_t* frame_image
	)
{
	assert_not_null(vk);
	assert_not_null(frame_image);

	vk_init_image(vk, &frame_image->image);

	if(!frame_image->set)
	{
		vk_get_descriptor_sets(vk, vk->sampler_set_layout, &frame_image->set, 1);
	}

	vk_write_images_to_set(vk, frame_image->set, &frame_image->image, 1);
}


private void
vk_free_frame_image(
	vk_t vk,
	vk_frame_image_t* frame_image
	)
{
	assert_not_null(vk);
	assert_not_null(frame_image);

	vk_free_image(vk, &frame_image->image);
}


private void
vk_init_timing(
	vk_t vk,
	vk_timing_t* timing,
	VkCommandBuffer command_buffer,
	uint32_t count
	)
{
	assert_not_null(vk);
	assert_not_null(timing);

	if(!vk->timing_enabled)
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

	VkResult result = vk->table.vkCreateQueryPool(vk->device, &query_pool_info, NULL, &timing->pool);
	hard_assert_eq(result, VK_SUCCESS);

	VkSemaphoreCreateInfo semaphore_info =
	{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0
	};

	result = vk->table.vkCreateSemaphore(vk->device, &semaphore_info, NULL, &timing->semaphore);
	hard_assert_eq(result, VK_SUCCESS);

	VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	vk_init_buffer(vk, sizeof(uint64_t) * count * 2, usage, flags, &timing->buffer);

	timing->results = alloc_malloc(sizeof(*timing->results) * timing->count * 2);
	assert_not_null(timing->results);

	timing->index_map = alloc_malloc(sizeof(*timing->index_map) * timing->count);
	assert_not_null(timing->index_map);

	timing->current = 0;
	timing->first_reset = true;
}


private void
vk_free_timing(
	vk_t vk,
	vk_timing_t* timing
	)
{
	assert_not_null(vk);

	if(!vk->timing_enabled)
	{
		return;
	}

	alloc_free(timing->index_map, sizeof(*timing->index_map) * timing->count);
	alloc_free(timing->results, sizeof(*timing->results) * timing->count * 2);

	vk_free_buffer(vk, &timing->buffer);

	vk->table.vkDestroySemaphore(vk->device, timing->semaphore, NULL);
	vk->table.vkDestroyQueryPool(vk->device, timing->pool, NULL);
}


private void
vk_timing_start(
	vk_t vk,
	vk_timing_t* timing,
	uint32_t index
	)
{
	assert_not_null(vk);

	if(!vk->timing_enabled)
	{
		return;
	}

	assert_lt(index, timing->count);

	uint32_t query_index = timing->current;
	assert_lt(query_index, timing->count * 2 - 1);

	vk->table.vkCmdWriteTimestamp(timing->command_buffer,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timing->pool, query_index);

	timing->index_map[index] = query_index;
}


private void
vk_timing_end(
	vk_t vk,
	vk_timing_t* timing,
	uint32_t index
	)
{
	assert_not_null(vk);

	if(!vk->timing_enabled)
	{
		return;
	}

	assert_lt(index, timing->count);

	uint32_t end_query_index = timing->index_map[index] + 1;

	vk->table.vkCmdWriteTimestamp(timing->command_buffer,
		VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing->pool, end_query_index);

	timing->current = end_query_index + 1;
}


private void
vk_timing_query(
	vk_t vk,
	vk_timing_t* timing
	)
{
	assert_not_null(vk);

	if(!vk->timing_enabled)
	{
		return;
	}

	vk->table.vkCmdCopyQueryPoolResults(timing->command_buffer,
		timing->pool, 0, timing->count * 2, timing->buffer.buffer, 0,
		sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
}


private void
vk_timing_load(
	vk_t vk,
	vk_timing_t* timing
	)
{
	assert_not_null(vk);

	if(!vk->timing_enabled)
	{
		return;
	}

	vk_read_from_buffer(vk, &timing->buffer, timing->results, sizeof(*timing->results) * timing->count * 2);
}


private uint64_t
vk_timing_get(
	vk_t vk,
	vk_timing_t* timing,
	uint32_t index
	)
{
	assert_not_null(vk);

	if(!vk->timing_enabled)
	{
		return 0;
	}

	assert_lt(index, timing->count);

	uint32_t start_query_index = timing->index_map[index];
	uint32_t end_query_index = start_query_index + 1;

	return (timing->results[end_query_index] - timing->results[start_query_index]) * vk->timestamp_period;
}


private void
vk_timing_reset(
	vk_t vk,
	vk_timing_t* timing
	)
{
	assert_not_null(vk);

	if(!vk->timing_enabled || (!timing->current && !timing->first_reset))
	{
		return;
	}

	vk->table.vkCmdResetQueryPool(timing->command_buffer, timing->pool, 0, timing->count * 2);
	timing->current = 0;
	timing->first_reset = false;
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

	VkShaderModuleCreateInfo shader_info =
	{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.codeSize = file.len,
		.pCode = (void*) file.data
	};

	VkShaderModule shader_module;
	VkResult result = vk->table.vkCreateShaderModule(vk->device, &shader_info, NULL, &shader_module);
	hard_assert_eq(result, VK_SUCCESS);

	file_free(file);

	return shader_module;
}


private void
vk_destroy_shader(
	vk_t vk,
	VkShaderModule shader
	)
{
	assert_not_null(vk);
	assert_not_null(shader);

	vk->table.vkDestroyShaderModule(vk->device, shader, NULL);
}


private VkPipelineCache
vk_init_pipeline_cache(
	vk_t vk,
	const char* path
	)
{
	assert_not_null(vk);
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

	VkResult result = vk->table.vkCreatePipelineCache(
		vk->device, &pipeline_cache_info, NULL, &pipeline_cache);
	hard_assert_eq(result, VK_SUCCESS);

	file_free(file);

	return pipeline_cache;
}


private void
vk_free_pipeline_cache(
	vk_t vk,
	const char* path,
	VkPipelineCache pipeline_cache
	)
{
	assert_not_null(vk);

	file_t file;
	VkResult result = vk->table.vkGetPipelineCacheData(vk->device, pipeline_cache, &file.len, NULL);
	hard_assert_eq(result, VK_SUCCESS);

	file.data = alloc_malloc(file.len);
	assert_ptr(file.data, file.len);

	result = vk->table.vkGetPipelineCacheData(vk->device, pipeline_cache, &file.len, file.data);
	hard_assert_eq(result, VK_SUCCESS);

	bool status = file_write(path, file);
	if(!status)
	{
		hard_assert_log("file_write(\"%s\")", path);
	}

	file_free(file);

	vk->table.vkDestroyPipelineCache(vk->device, pipeline_cache, NULL);

}


private void
vk_free_sampler(
	vk_t vk,
	VkSampler sampler
	)
{
	assert_not_null(vk);

	vk->table.vkDestroySampler(vk->device, sampler, NULL);
}


private void
vk_init_depth_sampler(
	vk_t vk
	)
{
	assert_not_null(vk);

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

	VkResult result = vk->table.vkCreateSampler(vk->device, &sampler_info, NULL, &vk->depth_sampler);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
vk_init_image_sampler(
	vk_t vk
	)
{
	assert_not_null(vk);

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
		.anisotropyEnable = !!vk->anisotropy,
		.maxAnisotropy = vk->anisotropy,
		.compareEnable = VK_FALSE,
		.compareOp = VK_COMPARE_OP_ALWAYS,
		.minLod = 0.0f,
		.maxLod = VK_LOD_CLAMP_NONE,
		.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
		.unnormalizedCoordinates = VK_FALSE
	};

	VkResult result = vk->table.vkCreateSampler(
		vk->device, &sampler_info, NULL, &vk->image_sampler);
	hard_assert_eq(result, VK_SUCCESS);
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

	vk_free_sampler(vk, vk->image_sampler);
	vk_free_sampler(vk, vk->depth_sampler);
}


private void
vk_init_shadow_render_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

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

	VkRenderPassCreateInfo render_pass_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.attachmentCount = MACRO_ARRAY_LEN(attachments),
		.pAttachments = attachments,
		.subpassCount = MACRO_ARRAY_LEN(subpasses),
		.pSubpasses = subpasses,
		.dependencyCount = MACRO_ARRAY_LEN(subpass_dependencies),
		.pDependencies = subpass_dependencies
	};

	VkResult result = vk->table.vkCreateRenderPass(vk->device,
		&render_pass_info, NULL, &vk->shadow.render_pass);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
vk_free_shadow_render_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->table.vkDestroyRenderPass(vk->device, vk->shadow.render_pass, NULL);
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

	VkPipelineShaderStageCreateInfo shader_stages[] =
	{
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vk_create_shader(vk, "shaders/shadow.vert.spv"),
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
		.pViewports = &vk->shadow_extent.viewport,
		.scissorCount = 1,
		.pScissors = &vk->shadow_extent.scissor
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
		.pAttachments = &vk_no_blending_attachment,
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

	VkResult result = vk->table.vkCreatePipelineLayout(vk->device,
		&pipeline_layout_info, NULL, &vk->shadow.pipeline_layout);
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
		.layout = vk->shadow.pipeline_layout,
		.renderPass = vk->shadow.render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	static const char* pipeline_cache_path = "cache/vk_shadow_pipeline.bin";
	VkPipelineCache pipeline_cache = vk_init_pipeline_cache(vk, pipeline_cache_path);

	result = vk->table.vkCreateGraphicsPipelines(vk->device,
		pipeline_cache, 1, &pipeline_info, NULL, &vk->shadow.pipeline);
	hard_assert_eq(result, VK_SUCCESS);

	vk_free_pipeline_cache(vk, pipeline_cache_path, pipeline_cache);

	for(uint32_t i = 0; i < MACRO_ARRAY_LEN(shader_stages); ++i)
	{
		vk_destroy_shader(vk, shader_stages[i].module);
	}
}


private void
vk_free_shadow_pipeline(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->table.vkDestroyPipeline(vk->device, vk->shadow.pipeline, NULL);
	vk->table.vkDestroyPipelineLayout(vk->device, vk->shadow.pipeline_layout, NULL);
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

	VkAttachmentDescription attachments[] =
	{
		{
			.flags = 0,
			.format = VK_FORMAT_D32_SFLOAT,
			.samples = vk->samples,
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
			.samples = vk->samples,
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
			.samples = vk->samples,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
		},
		{
			.flags = 0,
			.format = vk->format,
			.samples = vk->samples,
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
			.format = vk->format,
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

	VkRenderPassCreateInfo render_pass_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.attachmentCount = MACRO_ARRAY_LEN(attachments),
		.pAttachments = attachments,
		.subpassCount = MACRO_ARRAY_LEN(subpasses),
		.pSubpasses = subpasses,
		.dependencyCount = MACRO_ARRAY_LEN(subpass_dependencies),
		.pDependencies = subpass_dependencies
	};

	VkResult result = vk->table.vkCreateRenderPass(vk->device,
		&render_pass_info, NULL, &vk->scene.render_pass);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
vk_free_scene_render_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->table.vkDestroyRenderPass(vk->device, vk->scene.render_pass, NULL);
}


private void
vk_init_scene_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_scene_render_pass(vk);
}


private void
vk_free_scene_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_scene_render_pass(vk);
}


private void
vk_init_scene_pipeline(
	vk_t vk
	)
{
	assert_not_null(vk);

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
		.enable_depth_shadows = vk->options.enable_depth_shadows,
		.enable_backface_shadows = vk->options.enable_backface_shadows,
		.enable_specular = vk->options.enable_specular,
		.shadow_value = vk->options.shadow_value,
		.lambert_start_angle = vk->options.lambert_start_angle
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
			.module = vk_create_shader(vk, "shaders/scene.vert.spv"),
			.pName = "main",
			.pSpecializationInfo = NULL
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = vk_create_shader(vk, "shaders/scene.frag.spv"),
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
		.rasterizationSamples = vk->samples,
		.sampleShadingEnable = vk->options.sample_shading,
		.minSampleShading = vk->options.min_sample_shading,
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
		*color_blend_attachment = vk_no_blending_attachment;
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
		vk->vert_ubo_set_layout.layout,
		vk->sampler_set_layout.layout,
		vk->sampler_set_layout.layout
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

	VkResult result = vk->table.vkCreatePipelineLayout(vk->device,
		&pipeline_layout_info, NULL, &vk->scene.pipeline_layout);
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
		.layout = vk->scene.pipeline_layout,
		.renderPass = vk->scene.render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	static const char* pipeline_cache_path = "cache/vk_scene_pipeline.bin";
	VkPipelineCache pipeline_cache = vk_init_pipeline_cache(vk, pipeline_cache_path);

	result = vk->table.vkCreateGraphicsPipelines(vk->device,
		pipeline_cache, 1, &pipeline_info, NULL, &vk->scene.pipeline);
	hard_assert_eq(result, VK_SUCCESS);

	vk_free_pipeline_cache(vk, pipeline_cache_path, pipeline_cache);

	for(uint32_t i = 0; i < MACRO_ARRAY_LEN(shader_stages); ++i)
	{
		vk_destroy_shader(vk, shader_stages[i].module);
	}
}


private void
vk_free_scene_pipeline(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->table.vkDestroyPipeline(vk->device, vk->scene.pipeline, NULL);
	vk->table.vkDestroyPipelineLayout(vk->device, vk->scene.pipeline_layout, NULL);
}


private void
vk_init_scene_consts(
	vk_t vk
	)
{
	assert_not_null(vk);
}


private void
vk_free_scene_consts(
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
	vk_init_scene_pipeline(vk);
	vk_init_scene_consts(vk);
}


private void
vk_free_scene(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_scene_consts(vk);
	vk_free_scene_pipeline(vk);
	vk_free_scene_pass(vk);
}


private void
vk_init_ssao_render_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

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

	VkRenderPassCreateInfo render_pass_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.attachmentCount = MACRO_ARRAY_LEN(attachments),
		.pAttachments = attachments,
		.subpassCount = MACRO_ARRAY_LEN(subpasses),
		.pSubpasses = subpasses,
		.dependencyCount = MACRO_ARRAY_LEN(subpass_dependencies),
		.pDependencies = subpass_dependencies
	};

	VkResult result = vk->table.vkCreateRenderPass(vk->device,
		&render_pass_info, NULL, &vk->ssao.render_pass);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
vk_free_ssao_render_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->table.vkDestroyRenderPass(vk->device, vk->ssao.render_pass, NULL);
}


private void
vk_init_ssao_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_ssao_render_pass(vk);
}


private void
vk_free_ssao_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_ssao_render_pass(vk);
}


private void
vk_init_ssao_pipeline(
	vk_t vk
	)
{
	assert_not_null(vk);

	typedef struct vk_ssao_frag_specialization
	{
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
	vk_ssao_frag_specialization_t;

	vk_ssao_frag_specialization_t frag_specialization_data =
	{
		.ssao_kernel_size = vk->options.ssao_kernel_size,
		.ssao_noise_size = vk->options.ssao_noise_size,
		.ssao_radius = vk->options.ssao_radius,
		.ssao_bias = vk->options.ssao_bias,
		.ssao_power = vk->options.ssao_power,
		.ssao_range_check = vk->options.ssao_range_check,
		.ssao_depth_k = vk->options.ssao_depth_k,
		.ssao_depth_gamma = vk->options.ssao_depth_gamma,
		.ssao_debug = vk->options.ssao_debug
	};

	VkSpecializationMapEntry frag_map_entries[] =
	{
		{
			.constantID = 0,
			.offset = offsetof(vk_ssao_frag_specialization_t, ssao_kernel_size),
			.size = sizeof(frag_specialization_data.ssao_kernel_size)
		},
		{
			.constantID = 1,
			.offset = offsetof(vk_ssao_frag_specialization_t, ssao_noise_size),
			.size = sizeof(frag_specialization_data.ssao_noise_size)
		},
		{
			.constantID = 2,
			.offset = offsetof(vk_ssao_frag_specialization_t, ssao_radius),
			.size = sizeof(frag_specialization_data.ssao_radius)
		},
		{
			.constantID = 3,
			.offset = offsetof(vk_ssao_frag_specialization_t, ssao_bias),
			.size = sizeof(frag_specialization_data.ssao_bias)
		},
		{
			.constantID = 4,
			.offset = offsetof(vk_ssao_frag_specialization_t, ssao_power),
			.size = sizeof(frag_specialization_data.ssao_power)
		},
		{
			.constantID = 5,
			.offset = offsetof(vk_ssao_frag_specialization_t, ssao_range_check),
			.size = sizeof(frag_specialization_data.ssao_range_check)
		},
		{
			.constantID = 6,
			.offset = offsetof(vk_ssao_frag_specialization_t, ssao_depth_k),
			.size = sizeof(frag_specialization_data.ssao_depth_k)
		},
		{
			.constantID = 7,
			.offset = offsetof(vk_ssao_frag_specialization_t, ssao_depth_gamma),
			.size = sizeof(frag_specialization_data.ssao_depth_gamma)
		},
		{
			.constantID = 8,
			.offset = offsetof(vk_ssao_frag_specialization_t, ssao_debug),
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
			.module = vk_create_shader(vk, "shaders/fullscreen.vert.spv"),
			.pName = "main",
			.pSpecializationInfo = NULL
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = vk_create_shader(vk, "shaders/ssao.frag.spv"),
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
		.pAttachments = &vk_no_blending_attachment,
		.blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f }
	};

	VkDescriptorSetLayout set_layouts[] =
	{
		vk->scene_set_layout.layout,
		vk->sampler_set_layout.layout,
		vk->frag_ubo_set_layout.layout,
		vk->frag_ubo_set_layout.layout
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

	VkResult result = vk->table.vkCreatePipelineLayout(vk->device,
		&pipeline_layout_info, NULL, &vk->ssao.pipeline_layout);
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
		.layout = vk->ssao.pipeline_layout,
		.renderPass = vk->ssao.render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	static const char* pipeline_cache_path = "cache/vk_ssao_pipeline.bin";
	VkPipelineCache pipeline_cache = vk_init_pipeline_cache(vk, pipeline_cache_path);

	result = vk->table.vkCreateGraphicsPipelines(vk->device,
		pipeline_cache, 1, &pipeline_info, NULL, &vk->ssao.pipeline);
	hard_assert_eq(result, VK_SUCCESS);

	vk_free_pipeline_cache(vk, pipeline_cache_path, pipeline_cache);

	for(uint32_t i = 0; i < MACRO_ARRAY_LEN(shader_stages); ++i)
	{
		vk_destroy_shader(vk, shader_stages[i].module);
	}
}


private void
vk_free_ssao_pipeline(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->table.vkDestroyPipeline(vk->device, vk->ssao.pipeline, NULL);
	vk->table.vkDestroyPipelineLayout(vk->device, vk->ssao.pipeline_layout, NULL);
}


private void
vk_init_ssao_noise_const(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_image_t* image = &vk->ssao.noise.image;
	*image =
	(vk_image_t)
	{
		.path = NULL,
		.type = VK_IMAGE_TYPE_TEXTURE_2D_BITS | VK_IMAGE_TYPE_CUSTOM_FORMAT_BIT | VK_IMAGE_TYPE_CUSTOM_SIZE_BIT,
		.format = VK_FORMAT_R32G32B32A32_SFLOAT,
		.width = vk->options.ssao_noise_size,
		.height = vk->options.ssao_noise_size
	};
	vk_init_frame_image(vk, &vk->ssao.noise);

	uint32_t size = vk->options.ssao_noise_size * vk->options.ssao_noise_size;
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
	vk_copy_data_to_image(vk, image);
}


private void
vk_free_ssao_noise_const(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_frame_image(vk, &vk->ssao.noise);
}


private void
vk_init_ssao_kernel_const(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_frag_ubo_buffer(vk, sizeof(vec4) * vk->options.ssao_kernel_size, &vk->ssao.kernel_ubo);

	uint32_t size = vk->options.ssao_kernel_size;
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

	vk_copy_to_buffer(vk, &vk->ssao.kernel_ubo.buffer, data, sizeof(*data) * size);
}


private void
vk_free_ssao_kernel_const(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_ubo_buffer(vk, &vk->ssao.kernel_ubo);
}


private void
vk_init_ssao_consts(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_ssao_noise_const(vk);
	vk_init_ssao_kernel_const(vk);
}


private void
vk_free_ssao_consts(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_ssao_kernel_const(vk);
	vk_free_ssao_noise_const(vk);
}


private void
vk_init_ssao(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_ssao_pass(vk);
	vk_init_ssao_pipeline(vk);
	vk_init_ssao_consts(vk);
}


private void
vk_free_ssao(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_ssao_consts(vk);
	vk_free_ssao_pipeline(vk);
	vk_free_ssao_pass(vk);
}


private void
vk_init_ssao_blur_render_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	assert_not_null(vk);

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

	VkRenderPassCreateInfo render_pass_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.attachmentCount = MACRO_ARRAY_LEN(attachments),
		.pAttachments = attachments,
		.subpassCount = MACRO_ARRAY_LEN(subpasses),
		.pSubpasses = subpasses,
		.dependencyCount = MACRO_ARRAY_LEN(subpass_dependencies),
		.pDependencies = subpass_dependencies
	};

	VkResult result = vk->table.vkCreateRenderPass(vk->device,
		&render_pass_info, NULL, &vk->ssao_blur.render_pass);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
vk_free_ssao_blur_render_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->table.vkDestroyRenderPass(vk->device, vk->ssao_blur.render_pass, NULL);
}


private void
vk_init_ssao_blur_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_ssao_blur_render_pass(vk);
}


private void
vk_free_ssao_blur_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_ssao_blur_render_pass(vk);
}


private void
vk_init_ssao_blur_pipeline(
	vk_t vk
	)
{
	assert_not_null(vk);

	typedef struct vk_ssao_blur_frag_specialization
	{
		float ssao_blur_radius;
		float ssao_blur_falloff;
		float ssao_blur_depth_tolerance;
	}
	vk_ssao_blur_frag_specialization_t;

	vk_ssao_blur_frag_specialization_t frag_specialization_data =
	{
		.ssao_blur_radius = vk->options.ssao_blur_radius,
		.ssao_blur_falloff = vk->options.ssao_blur_falloff,
		.ssao_blur_depth_tolerance = vk->options.ssao_blur_depth_tolerance
	};

	VkSpecializationMapEntry frag_map_entries[] =
	{
		{
			.constantID = 0,
			.offset = offsetof(vk_ssao_blur_frag_specialization_t, ssao_blur_radius),
			.size = sizeof(frag_specialization_data.ssao_blur_radius)
		},
		{
			.constantID = 1,
			.offset = offsetof(vk_ssao_blur_frag_specialization_t, ssao_blur_falloff),
			.size = sizeof(frag_specialization_data.ssao_blur_falloff)
		},
		{
			.constantID = 2,
			.offset = offsetof(vk_ssao_blur_frag_specialization_t, ssao_blur_depth_tolerance),
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
			.module = vk_create_shader(vk, "shaders/fullscreen.vert.spv"),
			.pName = "main",
			.pSpecializationInfo = NULL
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = vk_create_shader(vk, "shaders/ssao_blur.frag.spv"),
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
		.pAttachments = &vk_no_blending_attachment,
		.blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f }
	};

	VkDescriptorSetLayout set_layouts[] =
	{
		vk->scene_set_layout.layout,
		vk->sampler_set_layout.layout
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

	VkResult result = vk->table.vkCreatePipelineLayout(vk->device,
		&pipeline_layout_info, NULL, &vk->ssao_blur.pipeline_layout);
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
		.layout = vk->ssao_blur.pipeline_layout,
		.renderPass = vk->ssao_blur.render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	const char* pipeline_cache_path = "cache/vk_ssao_blur_pipeline.bin";
	VkPipelineCache pipeline_cache = vk_init_pipeline_cache(vk, pipeline_cache_path);

	result = vk->table.vkCreateGraphicsPipelines(
		vk->device, pipeline_cache, 1, &pipeline_info, NULL, &vk->ssao_blur.pipeline);
	hard_assert_eq(result, VK_SUCCESS);

	vk_free_pipeline_cache(vk, pipeline_cache_path, pipeline_cache);

	for(uint32_t i = 0; i < MACRO_ARRAY_LEN(shader_stages); ++i)
	{
		vk_destroy_shader(vk, shader_stages[i].module);
	}
}


private void
vk_free_ssao_blur_pipeline(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->table.vkDestroyPipeline(vk->device, vk->ssao_blur.pipeline, NULL);
	vk->table.vkDestroyPipelineLayout(vk->device, vk->ssao_blur.pipeline_layout, NULL);
}


private void
vk_init_ssao_blur_consts(
	vk_t vk
	)
{
	assert_not_null(vk);
}


private void
vk_free_ssao_blur_consts(
	vk_t vk
	)
{
	assert_not_null(vk);
}


private void
vk_init_ssao_blur(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_ssao_blur_pass(vk);
	vk_init_ssao_blur_pipeline(vk);
	vk_init_ssao_blur_consts(vk);
}


private void
vk_free_ssao_blur(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_ssao_blur_consts(vk);
	vk_free_ssao_blur_pipeline(vk);
	vk_free_ssao_blur_pass(vk);
}


private void
vk_init_output_render_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkAttachmentDescription attachments[] =
	{
		{
			.flags = 0,
			.format = vk->format,
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

	VkRenderPassCreateInfo render_pass_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.attachmentCount = MACRO_ARRAY_LEN(attachments),
		.pAttachments = attachments,
		.subpassCount = MACRO_ARRAY_LEN(subpasses),
		.pSubpasses = subpasses,
		.dependencyCount = MACRO_ARRAY_LEN(subpass_dependencies),
		.pDependencies = subpass_dependencies
	};

	VkResult result = vk->table.vkCreateRenderPass(vk->device,
		&render_pass_info, NULL, &vk->output.render_pass);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
vk_free_output_render_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->table.vkDestroyRenderPass(vk->device, vk->output.render_pass, NULL);
}


private void
vk_init_output_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_output_render_pass(vk);
}


private void
vk_free_output_pass(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_output_render_pass(vk);
}


private void
vk_init_skybox_pipeline(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkPipelineShaderStageCreateInfo shader_stages[] =
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
		.pAttachments = &vk_no_blending_attachment,
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
		vk->sampler_set_layout.layout
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

	VkResult result = vk->table.vkCreatePipelineLayout(vk->device,
		&pipeline_layout_info, NULL, &vk->output.skybox.pipeline_layout);
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
		.layout = vk->output.skybox.pipeline_layout,
		.renderPass = vk->output.render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	static const char* pipeline_cache_path = "cache/vk_skybox_pipeline.bin";
	VkPipelineCache pipeline_cache = vk_init_pipeline_cache(vk, pipeline_cache_path);

	result = vk->table.vkCreateGraphicsPipelines(vk->device,
		pipeline_cache, 1, &pipeline_info, NULL, &vk->output.skybox.pipeline);
	hard_assert_eq(result, VK_SUCCESS);

	vk_free_pipeline_cache(vk, pipeline_cache_path, pipeline_cache);

	for(uint32_t i = 0; i < MACRO_ARRAY_LEN(shader_stages); ++i)
	{
		vk_destroy_shader(vk, shader_stages[i].module);
	}
}


private void
vk_free_skybox_pipeline(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->table.vkDestroyPipeline(vk->device, vk->output.skybox.pipeline, NULL);
	vk->table.vkDestroyPipelineLayout(vk->device, vk->output.skybox.pipeline_layout, NULL);
}


private void
vk_init_skybox_consts(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->output.skybox.sky.image =
	(vk_image_t)
	{
		.path = simulation_get_skybox_path(vk->simulation),
		.type = VK_IMAGE_TYPE_TEXTURE_CUBE_BITS
	};
	vk_init_frame_image(vk, &vk->output.skybox.sky);

	vk_init_vertex_buffer(vk, sizeof(vk_skybox_vertex_data), &vk->output.skybox.vertex_buffer);

	vk_copy_to_buffer(vk, &vk->output.skybox.vertex_buffer,
		vk_skybox_vertex_data, sizeof(vk_skybox_vertex_data));

	vk_init_index_buffer(vk, sizeof(vk_skybox_index_data), &vk->output.skybox.index_buffer);

	vk_copy_to_buffer(vk, &vk->output.skybox.index_buffer,
		vk_skybox_index_data, sizeof(vk_skybox_index_data));
}


private void
vk_free_skybox_consts(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_buffer(vk, &vk->output.skybox.index_buffer);
	vk_free_buffer(vk, &vk->output.skybox.vertex_buffer);

	vk_free_frame_image(vk, &vk->output.skybox.sky);
}


private void
vk_init_compose_pipeline(
	vk_t vk
	)
{
	assert_not_null(vk);

	typedef struct vk_compose_frag_specialization
	{
		int32_t enable_ssao;
	}
	vk_compose_frag_specialization_t;

	vk_compose_frag_specialization_t frag_specialization_data =
	{
		.enable_ssao = vk->options.enable_ssao
	};

	VkSpecializationMapEntry frag_map_entries[] =
	{
		{
			.constantID = 0,
			.offset = offsetof(vk_compose_frag_specialization_t, enable_ssao),
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
			.module = vk_create_shader(vk, "shaders/fullscreen.vert.spv"),
			.pName = "main",
			.pSpecializationInfo = NULL
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = vk_create_shader(vk, "shaders/compose.frag.spv"),
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
		.pAttachments = &vk_blending_attachment,
		.blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f }
	};

	VkDescriptorSetLayout set_layouts[] =
	{
		vk->sampler_set_layout.layout,
		vk->sampler_set_layout.layout
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

	VkResult result = vk->table.vkCreatePipelineLayout(vk->device,
		&pipeline_layout_info, NULL, &vk->output.compose.pipeline_layout);
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
		.layout = vk->output.compose.pipeline_layout,
		.renderPass = vk->output.render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	static const char* pipeline_cache_path = "cache/vk_compose_pipeline.bin";
	VkPipelineCache pipeline_cache = vk_init_pipeline_cache(vk, pipeline_cache_path);

	result = vk->table.vkCreateGraphicsPipelines(vk->device,
		pipeline_cache, 1, &pipeline_info, NULL, &vk->output.compose.pipeline);
	hard_assert_eq(result, VK_SUCCESS);

	vk_free_pipeline_cache(vk, pipeline_cache_path, pipeline_cache);

	for(uint32_t i = 0; i < MACRO_ARRAY_LEN(shader_stages); ++i)
	{
		vk_destroy_shader(vk, shader_stages[i].module);
	}
}


private void
vk_free_compose_pipeline(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->table.vkDestroyPipeline(vk->device, vk->output.compose.pipeline, NULL);
	vk->table.vkDestroyPipelineLayout(vk->device, vk->output.compose.pipeline_layout, NULL);
}


private void
vk_init_compose_consts(
	vk_t vk
	)
{
	assert_not_null(vk);
}


private void
vk_free_compose_consts(
	vk_t vk
	)
{
	assert_not_null(vk);
}


private void
vk_init_preview_pipeline(
	vk_t vk,
	const char* frag_shader_path,
	const char* pipeline_cache_path,
	VkPipelineLayout* pipeline_layout,
	VkPipeline* pipeline
	)
{
	assert_not_null(vk);
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
			.module = vk_create_shader(vk, "shaders/fullscreen.vert.spv"),
			.pName = "main",
			.pSpecializationInfo = NULL
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext = NULL,
			.flags = 0,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = vk_create_shader(vk, frag_shader_path),
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
		.pAttachments = &vk_no_blending_attachment,
		.blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f }
	};

	VkDescriptorSetLayout set_layouts[] =
	{
		vk->sampler_set_layout.layout
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

	VkResult result = vk->table.vkCreatePipelineLayout(
		vk->device, &pipeline_layout_info, NULL, pipeline_layout);
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
		.renderPass = vk->output.render_pass,
		.subpass = 0,
		.basePipelineHandle = VK_NULL_HANDLE,
		.basePipelineIndex = -1
	};

	VkPipelineCache pipeline_cache = vk_init_pipeline_cache(vk, pipeline_cache_path);

	result = vk->table.vkCreateGraphicsPipelines(
		vk->device, pipeline_cache, 1, &pipeline_info, NULL, pipeline);
	hard_assert_eq(result, VK_SUCCESS);

	vk_free_pipeline_cache(vk, pipeline_cache_path, pipeline_cache);

	for(uint32_t i = 0; i < MACRO_ARRAY_LEN(shader_stages); ++i)
	{
		vk_destroy_shader(vk, shader_stages[i].module);
	}
}


private void
vk_free_preview_pipeline(
	vk_t vk,
	VkPipelineLayout pipeline_layout,
	VkPipeline pipeline
	)
{
	assert_not_null(vk);
	assert_not_null(pipeline_layout);
	assert_not_null(pipeline);

	vk->table.vkDestroyPipeline(vk->device, pipeline, NULL);
	vk->table.vkDestroyPipelineLayout(vk->device, pipeline_layout, NULL);
}


private void
vk_init_preview_pipelines(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_preview_pipeline(vk, "shaders/preview_depth.frag.spv", "cache/vk_preview_depth_pipeline.bin",
		&vk->output.preview.depth.pipeline_layout, &vk->output.preview.depth.pipeline);

	vk_init_preview_pipeline(vk, "shaders/preview_image.frag.spv", "cache/vk_preview_image_pipeline.bin",
		&vk->output.preview.image.pipeline_layout, &vk->output.preview.image.pipeline);
}


private void
vk_free_preview_pipelines(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_preview_pipeline(vk, vk->output.preview.image.pipeline_layout, vk->output.preview.image.pipeline);
	vk_free_preview_pipeline(vk, vk->output.preview.depth.pipeline_layout, vk->output.preview.depth.pipeline);
}


private void
vk_init_preview_consts(
	vk_t vk
	)
{
	assert_not_null(vk);
}


private void
vk_free_preview_consts(
	vk_t vk
	)
{
	assert_not_null(vk);
}


private void
vk_init_output(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_output_pass(vk);
	vk_init_skybox_pipeline(vk);
	vk_init_skybox_consts(vk);
	vk_init_compose_pipeline(vk);
	vk_init_compose_consts(vk);
	vk_init_preview_pipelines(vk);
	vk_init_preview_consts(vk);
}


private void
vk_free_output(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_preview_consts(vk);
	vk_free_preview_pipelines(vk);
	vk_free_compose_consts(vk);
	vk_free_compose_pipeline(vk);
	vk_free_skybox_consts(vk);
	vk_free_skybox_pipeline(vk);
	vk_free_output_pass(vk);
}


private void
vk_init_pipelines(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_samplers(vk);
	vk_init_shadow(vk);
	vk_init_scene(vk);
	vk_init_ssao(vk);
	vk_init_ssao_blur(vk);
	vk_init_output(vk);
}


private void
vk_free_pipelines(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_output(vk);
	vk_free_ssao_blur(vk);
	vk_free_ssao(vk);
	vk_free_scene(vk);
	vk_free_shadow(vk);
	vk_free_samplers(vk);
}


private void
vk_init_models(
	vk_t vk
	)
{
	assert_not_null(vk);

	simulation_model_info_t info = simulation_get_model_info(vk->simulation);
	vk->model_count = info.model_count;
	vk->material_count = info.material_count;

	VkDescriptorImageInfo descriptor_images[vk->material_count];
	VkDescriptorImageInfo* descriptor_image = descriptor_images;

	VkWriteDescriptorSet descriptor_writes[vk->material_count];
	VkWriteDescriptorSet* descriptor_write = descriptor_writes;

	VkDescriptorSet sets[vk->material_count];
	VkDescriptorSet* set = sets;

	vk_get_descriptor_sets(vk, vk->sampler_set_layout, sets, vk->material_count);



	vk->materials = alloc_malloc(sizeof(*vk->materials) * vk->material_count);
	assert_ptr(vk->materials, sizeof(*vk->materials) * vk->material_count);

	vk->models = alloc_malloc(sizeof(*vk->models) * vk->model_count);
	assert_ptr(vk->models, sizeof(*vk->models) * vk->model_count);

	vk_material_t* material = vk->materials;
	vk_model_t* model = vk->models;

	for(uint32_t i = 0; i < vk->model_count; ++i)
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
			mesh->material_idx = material - vk->materials + sim_mesh->material_idx;
			mesh->vertex_count = sim_mesh->vertex_count;
			mesh->index_count = sim_mesh->index_count;

			vk_init_vertex_buffer(vk, sizeof(vk_shadow_vertex_data_t) *
				sim_mesh->vertex_count, &mesh->shadow_vertex_buffer);
			vk_copy_to_buffer(vk, &mesh->shadow_vertex_buffer,
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

			vk_init_vertex_buffer(vk, sizeof(*vertex_data) *
				sim_mesh->vertex_count, &mesh->scene_vertex_buffer);
			vk_copy_to_buffer(vk, &mesh->scene_vertex_buffer,
				vertex_data, sizeof(*vertex_data) * sim_mesh->vertex_count);

			alloc_free(vertex_data, sizeof(*vertex_data) * sim_mesh->vertex_count);

			vk_init_index_buffer(vk, sizeof(*sim_mesh->indexes) *
				sim_mesh->index_count, &mesh->index_buffer);
			vk_copy_to_buffer(vk, &mesh->index_buffer,
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

			vk_init_image(vk, &material->texture);

			glm_vec3_copy(sim_material->ambient, material->constant_data.ambient);
			glm_vec3_copy(sim_material->diffuse, material->constant_data.diffuse);
			material->constant_data.shininess = sim_material->shininess;
			material->constant_data.shininess_strength = sim_material->shininess_strength;

			*descriptor_image =
			(VkDescriptorImageInfo)
			{
				.sampler = vk->image_sampler,
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

		vk_init_vertex_buffer(vk, sizeof(vk_model_instance_data_t) *
			VK_MAX_INSTANCES, &model->instance_buffer);

		++model;
	}

	vk->table.vkUpdateDescriptorSets(vk->device, vk->material_count, descriptor_writes, 0, NULL);
}


private void
vk_free_models(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_model_t* model = vk->models;
	vk_model_t* model_end = model + vk->model_count;

	while(model < model_end)
	{
		vk_free_buffer(vk, &model->instance_buffer);

		vk_mesh_t* mesh = model->meshes;
		vk_mesh_t* mesh_end = mesh + model->mesh_count;

		while(mesh < mesh_end)
		{
			vk_free_buffer(vk, &mesh->index_buffer);
			vk_free_buffer(vk, &mesh->scene_vertex_buffer);
			vk_free_buffer(vk, &mesh->shadow_vertex_buffer);

			++mesh;
		}

		alloc_free(model->meshes, sizeof(*model->meshes) * model->mesh_count);

		++model;
	}

	alloc_free(vk->models, sizeof(*vk->models) * vk->model_count);

	vk_material_t* material = vk->materials;
	vk_material_t* material_end = material + vk->material_count;

	while(material < material_end)
	{
		vk_free_image(vk, &material->texture);
		str_free(material->texture.path);
		++material;
	}

	alloc_free(vk->materials, sizeof(*vk->materials) * vk->material_count);
}


private void
vk_free_swapchain(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->table.vkDestroySwapchainKHR(vk->device, vk->swapchain, NULL);
}


private void
vk_init_swapchain(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkSwapchainCreateInfoKHR swapchain_info =
	{
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.pNext = NULL,
		.flags = 0,
		.surface = vk->surface,
		.minImageCount = vk->image_count,
		.imageFormat = vk->format,
		.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
		.imageExtent = vk->screen_extent.extent,
		.imageArrayLayers = 1,
		.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
		.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = 0,
		.pQueueFamilyIndices = NULL,
		.preTransform = vk->transform,
		.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		.presentMode = vk->present_mode,
		.clipped = VK_TRUE,
		.oldSwapchain = vk->swapchain
	};

	VkSwapchainKHR swapchain;
	VkResult result = vk->table.vkCreateSwapchainKHR(vk->device, &swapchain_info, NULL, &swapchain);
	hard_assert_eq(result, VK_SUCCESS);

	vk_free_swapchain(vk);
	vk->swapchain = swapchain;
}


private void
vk_init_frames(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkCommandBuffer command_buffers[MACRO_ARRAY_LEN(vk->barriers)];
	VkCommandBuffer* command_buffer = command_buffers;

	VkCommandBufferAllocateInfo command_buffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.pNext = NULL,
		.commandPool = vk->command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = MACRO_ARRAY_LEN(command_buffers)
	};

	VkResult result = vk->table.vkAllocateCommandBuffers(
		vk->device, &command_buffer_info, command_buffers);
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

	vk_barrier_t* barrier = vk->barriers;
	vk_barrier_t* barrier_end = barrier + MACRO_ARRAY_LEN(vk->barriers);

	while(barrier < barrier_end)
	{
		result = vk->table.vkCreateSemaphore(vk->device, &semaphore_info, NULL, &barrier->semaphore);
		hard_assert_eq(result, VK_SUCCESS);

		result = vk->table.vkCreateFence(vk->device, &fence_info, NULL, &barrier->fence);
		hard_assert_eq(result, VK_SUCCESS);

		barrier->command_buffer = *command_buffer;

		vk_init_timing(vk, &barrier->timing, barrier->command_buffer, VK_BARRIER_TIMING_IDX__COUNT);

		++barrier;
		++command_buffer;
	}

	vk->barrier = vk->barriers;
}


private void
vk_free_frames(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkCommandBuffer command_buffers[MACRO_ARRAY_LEN(vk->barriers)];
	VkCommandBuffer* command_buffer = command_buffers;

	vk_barrier_t* barrier = vk->barriers;
	vk_barrier_t* barrier_end = barrier + MACRO_ARRAY_LEN(vk->barriers);

	while(barrier < barrier_end)
	{
		vk_free_timing(vk, &barrier->timing);

		*command_buffer = barrier->command_buffer;

		vk->table.vkDestroyFence(vk->device, barrier->fence, NULL);
		vk->table.vkDestroySemaphore(vk->device, barrier->semaphore, NULL);

		++barrier;
		++command_buffer;
	}

	vk->table.vkFreeCommandBuffers(vk->device,
		vk->command_pool, MACRO_ARRAY_LEN(vk->barriers), command_buffers);
}


private void
vk_init_shadow_framebuffer(
	vk_t vk,
	vk_frame_t* frame
	)
{
	assert_not_null(vk);
	assert_not_null(frame);

	vk_image_t* image = &frame->shadow.map.image;
	image->type = VK_IMAGE_TYPE_ATTACHMENT_BIT | VK_IMAGE_TYPE_DEPTH_BIT |
		VK_IMAGE_TYPE_SAMPLED_BIT | VK_IMAGE_TYPE_CUSTOM_SIZE_BIT;
	image->width = vk->options.shadow_map_size;
	image->height = vk->options.shadow_map_size;
	vk_init_frame_image(vk, &frame->shadow.map);

	VkImageView attachments[] =
	{
		image->view
	};

	VkFramebufferCreateInfo framebuffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.renderPass = vk->shadow.render_pass,
		.attachmentCount = MACRO_ARRAY_LEN(attachments),
		.pAttachments = attachments,
		.width = image->width,
		.height = image->height,
		.layers = 1
	};

	VkResult result = vk->table.vkCreateFramebuffer(vk->device,
		&framebuffer_info, NULL, &frame->shadow.framebuffer);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
vk_free_shadow_framebuffer(
	vk_t vk,
	vk_frame_t* frame
	)
{
	assert_not_null(vk);
	assert_not_null(frame);

	vk->table.vkDestroyFramebuffer(vk->device, frame->shadow.framebuffer, NULL);

	vk_free_frame_image(vk, &frame->shadow.map);
}


private void
vk_init_scene_framebuffer(
	vk_t vk,
	vk_frame_t* frame
	)
{
	assert_not_null(vk);
	assert_not_null(frame);

	vk_init_vert_ubo_buffer(vk, sizeof(vk_scene_vert_ubo_data_t), &frame->scene.vert_ubo);

	frame->scene.position_ms.image.type = VK_IMAGE_TYPE_ATTACHMENT_BIT |
		VK_IMAGE_TYPE_SAMPLED_BIT | VK_IMAGE_TYPE_MULTISAMPLED_BIT | VK_IMAGE_TYPE_CUSTOM_FORMAT_BIT;
	frame->scene.position_ms.image.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	vk_init_frame_image(vk, &frame->scene.position_ms);

	frame->scene.normal_ms.image.type = VK_IMAGE_TYPE_ATTACHMENT_BIT |
		VK_IMAGE_TYPE_SAMPLED_BIT | VK_IMAGE_TYPE_MULTISAMPLED_BIT | VK_IMAGE_TYPE_CUSTOM_FORMAT_BIT;
	frame->scene.normal_ms.image.format = VK_FORMAT_R8G8B8A8_UNORM;
	vk_init_frame_image(vk, &frame->scene.normal_ms);

	frame->scene.map_ms.image.type = VK_IMAGE_TYPE_ATTACHMENT_BIT |
		VK_IMAGE_TYPE_SAMPLED_BIT | VK_IMAGE_TYPE_MULTISAMPLED_BIT;
	vk_init_frame_image(vk, &frame->scene.map_ms);

	frame->scene.position.image.type = VK_IMAGE_TYPE_ATTACHMENT_BIT |
		VK_IMAGE_TYPE_SAMPLED_BIT | VK_IMAGE_TYPE_CUSTOM_FORMAT_BIT;
	frame->scene.position.image.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	vk_init_frame_image(vk, &frame->scene.position);

	frame->scene.normal.image.type = VK_IMAGE_TYPE_ATTACHMENT_BIT |
		VK_IMAGE_TYPE_SAMPLED_BIT | VK_IMAGE_TYPE_CUSTOM_FORMAT_BIT;
	frame->scene.normal.image.format = VK_FORMAT_R8G8B8A8_UNORM;
	vk_init_frame_image(vk, &frame->scene.normal);

	frame->scene.map.image.type = VK_IMAGE_TYPE_ATTACHMENT_BIT | VK_IMAGE_TYPE_SAMPLED_BIT;
	vk_init_frame_image(vk, &frame->scene.map);

	vk_image_t images[] =
	{
		frame->scene.position.image,
		frame->scene.normal.image
	};

	vk_get_descriptor_sets(vk, vk->scene_set_layout, &frame->scene.set, 1);

	vk_write_images_to_set(vk, frame->scene.set, images, MACRO_ARRAY_LEN(images));

	frame->scene.depth.type = VK_IMAGE_TYPE_ATTACHMENT_BIT |
		VK_IMAGE_TYPE_DEPTH_BIT | VK_IMAGE_TYPE_MULTISAMPLED_BIT | VK_IMAGE_TYPE_TRANSIENT_BIT;
	vk_init_image(vk, &frame->scene.depth);

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
		.renderPass = vk->scene.render_pass,
		.attachmentCount = MACRO_ARRAY_LEN(attachments),
		.pAttachments = attachments,
		.width = vk->screen_extent.width,
		.height = vk->screen_extent.height,
		.layers = 1
	};

	VkResult result = vk->table.vkCreateFramebuffer(vk->device,
		&framebuffer_info, NULL, &frame->scene.framebuffer);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
vk_free_scene_framebuffer(
	vk_t vk,
	vk_frame_t* frame
	)
{
	assert_not_null(vk);
	assert_not_null(frame);

	vk->table.vkDestroyFramebuffer(vk->device, frame->scene.framebuffer, NULL);

	vk_free_image(vk, &frame->scene.depth);
	vk_free_frame_image(vk, &frame->scene.map);
	vk_free_frame_image(vk, &frame->scene.normal);
	vk_free_frame_image(vk, &frame->scene.position);
	vk_free_frame_image(vk, &frame->scene.map_ms);
	vk_free_frame_image(vk, &frame->scene.normal_ms);
	vk_free_frame_image(vk, &frame->scene.position_ms);
	vk_free_ubo_buffer(vk, &frame->scene.vert_ubo);
}


private void
vk_init_ssao_framebuffer(
	vk_t vk,
	vk_frame_t* frame
	)
{
	assert_not_null(vk);
	assert_not_null(frame);

	vk_init_frag_ubo_buffer(vk, sizeof(vk_ssao_frag_ubo_data_t), &frame->ssao.frag_ubo);

	frame->ssao.map.image =
	(vk_image_t)
	{
		.type = VK_IMAGE_TYPE_ATTACHMENT_BIT | VK_IMAGE_TYPE_SAMPLED_BIT |
			VK_IMAGE_TYPE_CUSTOM_FORMAT_BIT | VK_IMAGE_TYPE_CUSTOM_SIZE_BIT,
		.format = VK_FORMAT_R8_UNORM,
		.width = vk->ssao_extent.width,
		.height = vk->ssao_extent.height
	};
	vk_init_frame_image(vk, &frame->ssao.map);

	VkImageView attachments[] =
	{
		frame->ssao.map.image.view
	};

	VkFramebufferCreateInfo framebuffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.renderPass = vk->ssao.render_pass,
		.attachmentCount = MACRO_ARRAY_LEN(attachments),
		.pAttachments = attachments,
		.width = vk->ssao_extent.width,
		.height = vk->ssao_extent.height,
		.layers = 1
	};

	VkResult result = vk->table.vkCreateFramebuffer(vk->device,
		&framebuffer_info, NULL, &frame->ssao.framebuffer);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
vk_free_ssao_framebuffer(
	vk_t vk,
	vk_frame_t* frame
	)
{
	assert_not_null(vk);
	assert_not_null(frame);

	vk->table.vkDestroyFramebuffer(vk->device, frame->ssao.framebuffer, NULL);

	vk_free_frame_image(vk, &frame->ssao.map);
	vk_free_ubo_buffer(vk, &frame->ssao.frag_ubo);
}


private void
vk_init_ssao_blur_framebuffer(
	vk_t vk,
	vk_frame_t* frame
	)
{
	assert_not_null(vk);
	assert_not_null(frame);

	frame->ssao_blur.map.image =
	(vk_image_t)
	{
		.type = VK_IMAGE_TYPE_ATTACHMENT_BIT | VK_IMAGE_TYPE_SAMPLED_BIT |
			VK_IMAGE_TYPE_CUSTOM_FORMAT_BIT | VK_IMAGE_TYPE_CUSTOM_SIZE_BIT,
		.format = VK_FORMAT_R8_UNORM,
		.width = vk->ssao_extent.width,
		.height = vk->ssao_extent.height
	};
	vk_init_frame_image(vk, &frame->ssao_blur.map);

	VkImageView attachments[] =
	{
		frame->ssao_blur.map.image.view
	};

	VkFramebufferCreateInfo framebuffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.renderPass = vk->ssao_blur.render_pass,
		.attachmentCount = MACRO_ARRAY_LEN(attachments),
		.pAttachments = attachments,
		.width = vk->ssao_extent.width,
		.height = vk->ssao_extent.height,
		.layers = 1
	};

	VkResult result = vk->table.vkCreateFramebuffer(vk->device,
		&framebuffer_info, NULL, &frame->ssao_blur.framebuffer);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
vk_free_ssao_blur_framebuffer(
	vk_t vk,
	vk_frame_t* frame
	)
{
	assert_not_null(vk);
	assert_not_null(frame);

	vk->table.vkDestroyFramebuffer(vk->device, frame->ssao_blur.framebuffer, NULL);

	vk_free_frame_image(vk, &frame->ssao_blur.map);
}


private void
vk_init_output_framebuffer(
	vk_t vk,
	vk_frame_t* frame,
	VkImage* swapchain_image
	)
{
	assert_not_null(vk);
	assert_not_null(frame);

	frame->output.image = *swapchain_image;

	VkImageViewCreateInfo image_view_info =
	{
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.pNext = NULL,
		.flags = 0,
		.image = frame->output.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = vk->format,
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

	VkResult result = vk->table.vkCreateImageView(vk->device,
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
		.renderPass = vk->output.render_pass,
		.attachmentCount = MACRO_ARRAY_LEN(attachments),
		.pAttachments = attachments,
		.width = vk->screen_extent.width,
		.height = vk->screen_extent.height,
		.layers = 1
	};

	result = vk->table.vkCreateFramebuffer(vk->device,
		&framebuffer_info, NULL, &frame->output.framebuffer);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
vk_free_output_framebuffer(
	vk_t vk,
	vk_frame_t* frame
	)
{
	assert_not_null(vk);
	assert_not_null(frame);

	vk->table.vkDestroyFramebuffer(vk->device, frame->output.framebuffer, NULL);
	vk->table.vkDestroyImageView(vk->device, frame->output.image_view, NULL);
}


private void
vk_init_framebuffers(
	vk_t vk
	)
{
	assert_not_null(vk);

	uint32_t image_count;
	VkResult result = vk->table.vkGetSwapchainImagesKHR(
		vk->device, vk->swapchain, &image_count, NULL);
	hard_assert_eq(result, VK_SUCCESS);

	assert_lt(image_count, VK_MAX_IMAGES);
	assert_ge(image_count, vk->image_count);

	vk->image_count = image_count;

	VkImage images[image_count];
	result = vk->table.vkGetSwapchainImagesKHR(
		vk->device, vk->swapchain, &image_count, images);
	hard_assert_eq(result, VK_SUCCESS);


	VkSemaphoreCreateInfo semaphore_info =
	{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
		.pNext = NULL,
		.flags = 0
	};

	vk_frame_t* frame = vk->frames;
	vk_frame_t* frame_end = frame + vk->image_count;

	VkImage* image = images;

	while(frame < frame_end)
	{
		vk_init_shadow_framebuffer(vk, frame);
		vk_init_scene_framebuffer(vk, frame);
		vk_init_ssao_framebuffer(vk, frame);
		vk_init_ssao_blur_framebuffer(vk, frame);
		vk_init_output_framebuffer(vk, frame, image);


		result = vk->table.vkCreateSemaphore(vk->device, &semaphore_info, NULL, &frame->semaphore);
		hard_assert_eq(result, VK_SUCCESS);

		frame->barrier = NULL;


		++frame;
		++image;
	}
}


private void
vk_free_framebuffers(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_frame_t* frame = vk->frames;
	vk_frame_t* frame_end = frame + vk->image_count;

	while(frame < frame_end)
	{
		vk->table.vkDestroySemaphore(vk->device, frame->semaphore, NULL);

		vk_free_output_framebuffer(vk, frame);
		vk_free_ssao_blur_framebuffer(vk, frame);
		vk_free_ssao_framebuffer(vk, frame);
		vk_free_scene_framebuffer(vk, frame);
		vk_free_shadow_framebuffer(vk, frame);

		++frame;
	}
}


private void
vk_device_wait_idle(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkResult result = vk->table.vkDeviceWaitIdle(vk->device);
	hard_assert_eq(result, VK_SUCCESS);
}


private void
vk_recreate_swapchain(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_device_wait_idle(vk);

	vk_get_extent(vk);

	vk_free_framebuffers(vk);
	vk_init_swapchain(vk);
	vk_init_framebuffers(vk);

	atomic_store_rel(&vk->window.resize.boolean, false);
}


#define VK_FOR_EACH_MODEL(entities_per_model, ...)									\
do																					\
{																					\
	vk_entities_per_model_t* entities_per_model = entity_data;						\
	vk_entities_per_model_t* entities_per_model##_end =								\
		entities_per_model + vk->model_count;										\
																					\
	__VA_OPT__(vk_model_t* __VA_ARGS__ = vk->models;)								\
																					\
	while(entities_per_model < entities_per_model##_end)							\
	{																				\
		if(entities_per_model->entities_used != 0)									\
		{																			\
			hard_assert_le(entities_per_model->entities_used, VK_MAX_INSTANCES);

#define VK_FOR_EACH_MODEL_END(entities_per_model, ...)	\
		}												\
														\
		++entities_per_model;							\
		__VA_OPT__(++__VA_ARGS__;)						\
	}													\
}														\
while(0)


private void
vk_draw_shadow(
	vk_t vk,
	vk_frame_t* frame,
	simulation_camera_t* camera,
	simulation_transform_t* transform,
	vk_entities_per_model_t* entity_data
	)
{
	assert_not_null(vk);
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
		.renderPass = vk->shadow.render_pass,
		.framebuffer = frame->shadow.framebuffer,
		.renderArea = vk->shadow_extent.scissor,
		.clearValueCount = MACRO_ARRAY_LEN(clear_values),
		.pClearValues = clear_values
	};

	vk->table.vkCmdBeginRenderPass(vk->barrier->command_buffer,
		&render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

	vk->table.vkCmdBindPipeline(vk->barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, vk->shadow.pipeline);

	vk_shadow_vert_constant_data_t shadow_vert_constant_data;
	glm_mat4_copy(transform->light_transform, shadow_vert_constant_data.transform);

	vk->table.vkCmdPushConstants(vk->barrier->command_buffer,
		vk->shadow.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
		0, sizeof(shadow_vert_constant_data), &shadow_vert_constant_data);

	VK_FOR_EACH_MODEL(entities_per_model, model)
	{
		vk->table.vkCmdBindVertexBuffers(vk->barrier->command_buffer,
			1, 1, &model->instance_buffer.buffer, (VkDeviceSize[]){0});

		vk_mesh_t* mesh = model->meshes;
		vk_mesh_t* mesh_end = mesh + model->mesh_count;

		while(mesh < mesh_end)
		{
			vk->table.vkCmdBindVertexBuffers(vk->barrier->command_buffer,
				0, 1, &mesh->shadow_vertex_buffer.buffer, (VkDeviceSize[]){0});

			vk->table.vkCmdBindIndexBuffer(vk->barrier->command_buffer,
				mesh->index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

			vk->table.vkCmdDrawIndexed(vk->barrier->command_buffer,
				mesh->index_count, entities_per_model->entities_used, 0, 0, 0);

			++mesh;
		}
	}
	VK_FOR_EACH_MODEL_END(entities_per_model, model);

	vk->table.vkCmdEndRenderPass(vk->barrier->command_buffer);
}


private void
vk_draw_scene(
	vk_t vk,
	vk_frame_t* frame,
	simulation_camera_t* camera,
	simulation_transform_t* transform,
	vk_entities_per_model_t* entity_data
	)
{
	assert_not_null(vk);
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
		.renderPass = vk->scene.render_pass,
		.framebuffer = frame->scene.framebuffer,
		.renderArea = vk->screen_extent.scissor,
		.clearValueCount = MACRO_ARRAY_LEN(clear_values),
		.pClearValues = clear_values
	};

	vk->table.vkCmdBeginRenderPass(vk->barrier->command_buffer,
		&render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

	vk->table.vkCmdSetViewport(vk->barrier->command_buffer, 0, 1, &vk->screen_extent.viewport);
	vk->table.vkCmdSetScissor(vk->barrier->command_buffer, 0, 1, &vk->screen_extent.scissor);

	vk_scene_vert_ubo_data_t scene_vert_ubo_data;
	glm_mat4_copy(transform->projection, scene_vert_ubo_data.projection);
	glm_mat4_copy(transform->view, scene_vert_ubo_data.view);
	glm_mat4_copy(transform->light_transform, scene_vert_ubo_data.light_transform);
	glm_vec4_copy(transform->light_direction, scene_vert_ubo_data.light_direction);
	glm_vec3_copy(camera->pos, scene_vert_ubo_data.camera_position);
	scene_vert_ubo_data.camera_position[3] = 1.0f;

	vk_copy_to_buffer(vk, &frame->scene.vert_ubo.buffer,
		&scene_vert_ubo_data, sizeof(scene_vert_ubo_data));

	vk_scene_frag_constant_data_t scene_frag_constant_data =
	{
		.near = camera->near
	};

	vk->table.vkCmdBindPipeline(vk->barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, vk->scene.pipeline);

	vk->table.vkCmdBindDescriptorSets(vk->barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, vk->scene.pipeline_layout,
		0, 1, &frame->scene.vert_ubo.set, 0, NULL);

	vk->table.vkCmdBindDescriptorSets(vk->barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, vk->scene.pipeline_layout,
		2, 1, &frame->shadow.map.set, 0, NULL);

	VK_FOR_EACH_MODEL(entities_per_model, model)
	{
		vk->table.vkCmdBindVertexBuffers(vk->barrier->command_buffer,
			1, 1, &model->instance_buffer.buffer, (VkDeviceSize[]){0});

		vk_mesh_t* mesh = model->meshes;
		vk_mesh_t* mesh_end = mesh + model->mesh_count;

		while(mesh < mesh_end)
		{
			vk_material_t* material = vk->materials + mesh->material_idx;

			glm_vec4_copy(material->constant_data.diffuse, scene_frag_constant_data.diffuse);
			glm_vec4_copy(material->constant_data.ambient, scene_frag_constant_data.ambient);
			scene_frag_constant_data.shininess = material->constant_data.shininess;
			scene_frag_constant_data.shininess_strength = material->constant_data.shininess_strength;

			vk->table.vkCmdPushConstants(vk->barrier->command_buffer,
				vk->scene.pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT,
				0, sizeof(scene_frag_constant_data), &scene_frag_constant_data);

			vk->table.vkCmdBindVertexBuffers(vk->barrier->command_buffer,
				0, 1, &mesh->scene_vertex_buffer.buffer, (VkDeviceSize[]){0});

			vk->table.vkCmdBindIndexBuffer(vk->barrier->command_buffer,
				mesh->index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

			vk->table.vkCmdBindDescriptorSets(vk->barrier->command_buffer,
				VK_PIPELINE_BIND_POINT_GRAPHICS, vk->scene.pipeline_layout,
				1, 1, &material->set, 0, NULL);

			vk->table.vkCmdDrawIndexed(vk->barrier->command_buffer,
				mesh->index_count, entities_per_model->entities_used, 0, 0, 0);

			++mesh;
		}
	}
	VK_FOR_EACH_MODEL_END(entities_per_model, model);

	vk->table.vkCmdEndRenderPass(vk->barrier->command_buffer);
}


private void
vk_draw_ssao(
	vk_t vk,
	vk_frame_t* frame,
	simulation_camera_t* camera,
	simulation_transform_t* transform,
	vk_entities_per_model_t* entity_data
	)
{
	assert_not_null(vk);
	assert_not_null(frame);

	VkRenderPassBeginInfo render_pass_begin_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.pNext = NULL,
		.renderPass = vk->ssao.render_pass,
		.framebuffer = frame->ssao.framebuffer,
		.renderArea = vk->ssao_extent.scissor,
		.clearValueCount = 0,
		.pClearValues = NULL
	};

	vk->table.vkCmdBeginRenderPass(vk->barrier->command_buffer,
		&render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

	vk->table.vkCmdSetViewport(vk->barrier->command_buffer, 0, 1, &vk->ssao_extent.viewport);
	vk->table.vkCmdSetScissor(vk->barrier->command_buffer, 0, 1, &vk->ssao_extent.scissor);

	vk_ssao_frag_ubo_data_t ssao_frag_ubo_data;
	glm_mat4_copy(transform->projection, ssao_frag_ubo_data.projection);

	vk_copy_to_buffer(vk, &frame->ssao.frag_ubo.buffer,
		&ssao_frag_ubo_data, sizeof(ssao_frag_ubo_data));

	vk->table.vkCmdBindPipeline(vk->barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, vk->ssao.pipeline);

	VkDescriptorSet sets[] =
	{
		frame->scene.set,
		vk->ssao.noise.set,
		frame->ssao.frag_ubo.set,
		vk->ssao.kernel_ubo.set
	};

	vk->table.vkCmdBindDescriptorSets(vk->barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, vk->ssao.pipeline_layout,
		0, MACRO_ARRAY_LEN(sets), sets, 0, NULL);

	vk->table.vkCmdDraw(vk->barrier->command_buffer, 6, 1, 0, 0);

	vk->table.vkCmdEndRenderPass(vk->barrier->command_buffer);
}


private void
vk_draw_ssao_blur(
	vk_t vk,
	vk_frame_t* frame,
	simulation_camera_t* camera,
	simulation_transform_t* transform,
	vk_entities_per_model_t* entity_data
	)
{
	assert_not_null(vk);
	assert_not_null(frame);

	VkRenderPassBeginInfo render_pass_begin_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.pNext = NULL,
		.renderPass = vk->ssao_blur.render_pass,
		.framebuffer = frame->ssao_blur.framebuffer,
		.renderArea = vk->ssao_extent.scissor,
		.clearValueCount = 0,
		.pClearValues = NULL
	};

	vk->table.vkCmdBeginRenderPass(vk->barrier->command_buffer,
		&render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

	vk->table.vkCmdSetViewport(vk->barrier->command_buffer, 0, 1, &vk->ssao_extent.viewport);
	vk->table.vkCmdSetScissor(vk->barrier->command_buffer, 0, 1, &vk->ssao_extent.scissor);

	vk->table.vkCmdBindPipeline(vk->barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, vk->ssao_blur.pipeline);

	VkDescriptorSet sets[] =
	{
		frame->scene.set,
		frame->ssao.map.set
	};

	vk->table.vkCmdBindDescriptorSets(vk->barrier->command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS, vk->ssao_blur.pipeline_layout,
		0, MACRO_ARRAY_LEN(sets), sets, 0, NULL);

	vk->table.vkCmdDraw(vk->barrier->command_buffer, 6, 1, 0, 0);

	vk->table.vkCmdEndRenderPass(vk->barrier->command_buffer);
}


private void
vk_draw_output(
	vk_t vk,
	vk_frame_t* frame,
	simulation_camera_t* camera,
	simulation_transform_t* transform,
	vk_entities_per_model_t* entity_data
	)
{
	assert_not_null(vk);
	assert_not_null(frame);

	VkRenderPassBeginInfo render_pass_begin_info =
	{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.pNext = NULL,
		.renderPass = vk->output.render_pass,
		.framebuffer = frame->output.framebuffer,
		.renderArea = vk->screen_extent.scissor,
		.clearValueCount = 0,
		.pClearValues = NULL
	};

	vk->table.vkCmdBeginRenderPass(vk->barrier->command_buffer,
		&render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

	vk->table.vkCmdSetViewport(vk->barrier->command_buffer, 0, 1, &vk->screen_extent.viewport);
	vk->table.vkCmdSetScissor(vk->barrier->command_buffer, 0, 1, &vk->screen_extent.scissor);


	switch(vk->options.preview)
	{

	case VK_PREVIEW_NONE:
	{
		vk_skybox_constant_data_t skybox_constant_data;
		glm_mat4_copy(transform->projection, skybox_constant_data.transform);

		mat4 view;
		glm_mat4_copy(transform->view, view);
		view[3][0] = 0.0f;
		view[3][1] = 0.0f;
		view[3][2] = 0.0f;
		glm_mat4_mul(skybox_constant_data.transform, view, skybox_constant_data.transform);

		vk->table.vkCmdBindPipeline(vk->barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, vk->output.skybox.pipeline);

		vk->table.vkCmdPushConstants(vk->barrier->command_buffer,
			vk->output.skybox.pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
			0, sizeof(skybox_constant_data), &skybox_constant_data);

		vk->table.vkCmdBindVertexBuffers(vk->barrier->command_buffer,
			0, 1, &vk->output.skybox.vertex_buffer.buffer, (VkDeviceSize[]){0});

		vk->table.vkCmdBindIndexBuffer(vk->barrier->command_buffer,
			vk->output.skybox.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT16);

		vk->table.vkCmdBindDescriptorSets(vk->barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, vk->output.skybox.pipeline_layout,
			0, 1, &vk->output.skybox.sky.set, 0, NULL);

		vk->table.vkCmdDrawIndexed(vk->barrier->command_buffer,
			MACRO_ARRAY_LEN(vk_skybox_index_data), 1, 0, 0, 0);


		vk->table.vkCmdBindPipeline(vk->barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, vk->output.compose.pipeline);

		VkDescriptorSet sets[] =
		{
			frame->scene.map.set,
			frame->ssao_blur.map.set
		};

		vk->table.vkCmdBindDescriptorSets(vk->barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, vk->output.compose.pipeline_layout,
			0, MACRO_ARRAY_LEN(sets), sets, 0, NULL);

		vk->table.vkCmdDraw(vk->barrier->command_buffer, 6, 1, 0, 0);

		break;
	}

	case VK_PREVIEW_SHADOW_MAP:
	{
		vk->table.vkCmdBindPipeline(vk->barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, vk->output.preview.depth.pipeline);

		vk->table.vkCmdBindDescriptorSets(vk->barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, vk->output.preview.depth.pipeline_layout,
			0, 1, &frame->shadow.map.set, 0, NULL);

		vk->table.vkCmdDraw(vk->barrier->command_buffer, 6, 1, 0, 0);

		break;
	}

	case VK_PREVIEW_SCENE_POSITION_MAP:
	{
		vk->table.vkCmdBindPipeline(vk->barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, vk->output.preview.image.pipeline);

		vk->table.vkCmdBindDescriptorSets(vk->barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, vk->output.preview.image.pipeline_layout,
			0, 1, &frame->scene.position.set, 0, NULL);

		vk->table.vkCmdDraw(vk->barrier->command_buffer, 6, 1, 0, 0);

		break;
	}

	case VK_PREVIEW_SCENE_NORMAL_MAP:
	{
		vk->table.vkCmdBindPipeline(vk->barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, vk->output.preview.image.pipeline);

		vk->table.vkCmdBindDescriptorSets(vk->barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, vk->output.preview.image.pipeline_layout,
			0, 1, &frame->scene.normal.set, 0, NULL);

		vk->table.vkCmdDraw(vk->barrier->command_buffer, 6, 1, 0, 0);

		break;
	}

	case VK_PREVIEW_SCENE_MAP:
	{
		vk->table.vkCmdBindPipeline(vk->barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, vk->output.preview.image.pipeline);

		vk->table.vkCmdBindDescriptorSets(vk->barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, vk->output.preview.image.pipeline_layout,
			0, 1, &frame->scene.map.set, 0, NULL);

		vk->table.vkCmdDraw(vk->barrier->command_buffer, 6, 1, 0, 0);

		break;
	}

	case VK_PREVIEW_SSAO_MAP:
	{
		vk->table.vkCmdBindPipeline(vk->barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, vk->output.preview.image.pipeline);

		vk->table.vkCmdBindDescriptorSets(vk->barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, vk->output.preview.image.pipeline_layout,
			0, 1, &frame->ssao.map.set, 0, NULL);

		vk->table.vkCmdDraw(vk->barrier->command_buffer, 6, 1, 0, 0);

		break;
	}

	case VK_PREVIEW_SSAO_BLUR_MAP:
	{
		vk->table.vkCmdBindPipeline(vk->barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, vk->output.preview.image.pipeline);

		vk->table.vkCmdBindDescriptorSets(vk->barrier->command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS, vk->output.preview.image.pipeline_layout,
			0, 1, &frame->ssao_blur.map.set, 0, NULL);

		vk->table.vkCmdDraw(vk->barrier->command_buffer, 6, 1, 0, 0);

		break;
	}

	default: assert_unreachable();

	}


	vk->table.vkCmdEndRenderPass(vk->barrier->command_buffer);
}


private void
vk_draw(
	vk_t vk
	)
{
	assert_not_null(vk);

	VkResult result = vk->table.vkWaitForFences(vk->device, 1, &vk->barrier->fence, VK_TRUE, UINT64_MAX);
	hard_assert_eq(result, VK_SUCCESS);

	uint64_t frame_time = 0;
	if(vk->barrier->timing.current)
	{
		vk_timing_load(vk, &vk->barrier->timing);

		uint64_t shadow_time = vk_timing_get(vk, &vk->barrier->timing, VK_BARRIER_TIMING_IDX_SHADOW);
		stats_log(vk->stats, "vk_barrier_timing_shadow", shadow_time);
		frame_time += shadow_time;

		uint64_t scene_time = vk_timing_get(vk, &vk->barrier->timing, VK_BARRIER_TIMING_IDX_SCENE);
		stats_log(vk->stats, "vk_barrier_timing_scene", scene_time);
		frame_time += scene_time;

		uint64_t ssao_time = vk_timing_get(vk, &vk->barrier->timing, VK_BARRIER_TIMING_IDX_SSAO);
		stats_log(vk->stats, "vk_barrier_timing_ssao", ssao_time);
		frame_time += ssao_time;

		uint64_t ssao_blur_time = vk_timing_get(vk, &vk->barrier->timing, VK_BARRIER_TIMING_IDX_SSAO_BLUR);
		stats_log(vk->stats, "vk_barrier_timing_ssao_blur", ssao_blur_time);
		frame_time += ssao_blur_time;

		uint64_t output_time = vk_timing_get(vk, &vk->barrier->timing, VK_BARRIER_TIMING_IDX_OUTPUT);
		stats_log(vk->stats, "vk_barrier_timing_output", output_time);
		frame_time += output_time;
	}

	uint32_t image_idx;
	result = vk->table.vkAcquireNextImageKHR(vk->device, vk->swapchain,
		UINT64_MAX, vk->barrier->semaphore, VK_NULL_HANDLE, &image_idx);
	if(result == VK_ERROR_OUT_OF_DATE_KHR)
	{
		vk_recreate_swapchain(vk);
		return;
	}

	if(result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
	{
		hard_assert_log("%d\n", result);
	}

	vk_frame_t* frame = vk->frames + image_idx;

	if(frame->barrier)
	{
		result = vk->table.vkWaitForFences(vk->device, 1, &frame->barrier->fence, VK_TRUE, UINT64_MAX);
		hard_assert_eq(result, VK_SUCCESS);
	}
	frame->barrier = vk->barrier;

	result = vk->table.vkResetFences(vk->device, 1, &vk->barrier->fence);
	hard_assert_eq(result, VK_SUCCESS);



	uint64_t start_time = time_get();
	simulation_update(vk->simulation);

	result = vk->table.vkResetCommandBuffer(vk->barrier->command_buffer, 0);
	hard_assert_eq(result, VK_SUCCESS);

	VkCommandBufferBeginInfo command_buffer_info =
	{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.pNext = NULL,
		.flags = 0,
		.pInheritanceInfo = NULL
	};

	result = vk->table.vkBeginCommandBuffer(vk->barrier->command_buffer, &command_buffer_info);
	hard_assert_eq(result, VK_SUCCESS);

	simulation_camera_t camera = simulation_get_camera(vk->simulation);
	simulation_transform_t transform = simulation_get_transform(
		vk->simulation, vk->screen_extent.width, vk->screen_extent.height);

	uint32_t sim_entity_count;
	simulation_entity_data_t* sim_entity_data =
		simulation_get_entity_data(vk->simulation, &sim_entity_count);

	simulation_entity_data_t* sim_entity = sim_entity_data;
	simulation_entity_data_t* sim_entity_end = sim_entity + sim_entity_count;

	vk_entities_per_model_t* entity_data = alloc_calloc(sizeof(*entity_data) * vk->model_count);
	assert_ptr(entity_data, sizeof(*entity_data) * vk->model_count);

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

	VK_FOR_EACH_MODEL(entities_per_model, model)
	{
		simulation_entity_data_t** entity = entities_per_model->entities;
		simulation_entity_data_t** entity_end = entity + entities_per_model->entities_used;

		uint64_t instance_data_size =
			sizeof(vk_model_instance_data_t) * entities_per_model->entities_used;
		vk_model_instance_data_t* instance_data = alloc_malloc(instance_data_size);
		assert_ptr(instance_data, instance_data_size);

		vk_model_instance_data_t* instance = instance_data;

		while(entity < entity_end)
		{
			glm_mat4_copy((*entity)->transform, instance->transform);

			++entity;
			++instance;
		}

		vk_copy_to_buffer(vk, &model->instance_buffer, instance_data, instance_data_size);
		alloc_free(instance_data, instance_data_size);
	}
	VK_FOR_EACH_MODEL_END(entities_per_model, model);

	vk_timing_reset(vk, &vk->barrier->timing);

	vk_timing_start(vk, &vk->barrier->timing, VK_BARRIER_TIMING_IDX_SHADOW);
	vk_draw_shadow(vk, frame, &camera, &transform, entity_data);
	vk_timing_end(vk, &vk->barrier->timing, VK_BARRIER_TIMING_IDX_SHADOW);

	vk_timing_start(vk, &vk->barrier->timing, VK_BARRIER_TIMING_IDX_SCENE);
	vk_draw_scene(vk, frame, &camera, &transform, entity_data);
	vk_timing_end(vk, &vk->barrier->timing, VK_BARRIER_TIMING_IDX_SCENE);

	vk_timing_start(vk, &vk->barrier->timing, VK_BARRIER_TIMING_IDX_SSAO);
	vk_draw_ssao(vk, frame, &camera, &transform, entity_data);
	vk_timing_end(vk, &vk->barrier->timing, VK_BARRIER_TIMING_IDX_SSAO);

	vk_timing_start(vk, &vk->barrier->timing, VK_BARRIER_TIMING_IDX_SSAO_BLUR);
	vk_draw_ssao_blur(vk, frame, &camera, &transform, entity_data);
	vk_timing_end(vk, &vk->barrier->timing, VK_BARRIER_TIMING_IDX_SSAO_BLUR);

	vk_timing_start(vk, &vk->barrier->timing, VK_BARRIER_TIMING_IDX_OUTPUT);
	vk_draw_output(vk, frame, &camera, &transform, entity_data);
	vk_timing_end(vk, &vk->barrier->timing, VK_BARRIER_TIMING_IDX_OUTPUT);

	vk_timing_query(vk, &vk->barrier->timing);

	VK_FOR_EACH_MODEL(entities_per_model)
	{
		alloc_free(entities_per_model->entities,
			sizeof(*entities_per_model->entities) * entities_per_model->entities_size);
	}
	VK_FOR_EACH_MODEL_END(entities_per_model);

	alloc_free(entity_data, sizeof(*entity_data) * vk->model_count);

	simulation_free_entity_data(sim_entity_data, sim_entity_count);

	result = vk->table.vkEndCommandBuffer(vk->barrier->command_buffer);
	hard_assert_eq(result, VK_SUCCESS);


	VkPipelineStageFlags wait_stages[] =
	{
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
	};

	VkSubmitInfo submit_info =
	{
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.pNext = NULL,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &vk->barrier->semaphore,
		.pWaitDstStageMask = wait_stages,
		.commandBufferCount = 1,
		.pCommandBuffers = &vk->barrier->command_buffer,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &frame->semaphore
	};

	result = vk->table.vkQueueSubmit(vk->queue, 1, &submit_info, vk->barrier->fence);
	hard_assert_eq(result, VK_SUCCESS);

	VkPresentInfoKHR present_info =
	{
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.pNext = NULL,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &frame->semaphore,
		.swapchainCount = 1,
		.pSwapchains = &vk->swapchain,
		.pImageIndices = &image_idx,
		.pResults = NULL
	};

	result = vk->table.vkQueuePresentKHR(vk->queue, &present_info);
	if(
		result == VK_ERROR_OUT_OF_DATE_KHR ||
		result == VK_SUBOPTIMAL_KHR ||
		atomic_load_acq(&vk->window.resize.boolean)
		)
	{
		vk_recreate_swapchain(vk);
	}
	else
	{
		hard_assert_eq(result, VK_SUCCESS);
	}

	uint64_t end_time = time_get();
	stats_log(vk->stats, "vk_command_record_time", end_time - start_time);

	frame_time += end_time - start_time;
	stats_log(vk->stats, "vk_frame_time", frame_time);

	if(++vk->barrier >= vk->barriers + MACRO_ARRAY_LEN(vk->barriers))
	{
		vk->barrier = vk->barriers;
	}
}


private void
vk_thread_fn(
	vk_t vk
	)
{
	assert_not_null(vk);

	while(atomic_load_acq(&vk->is_running))
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

	atomic_init(&vk->is_running, true);

	thread_data_t thread_data =
	{
		.fn = (void*) vk_thread_fn,
		.data = vk
	};
	thread_init(&vk->thread, thread_data);
}


private void
vk_free_thread(
	vk_t vk
	)
{
	assert_not_null(vk);

	atomic_store_rel(&vk->is_running, false);
	thread_join(vk->thread);

	thread_free(&vk->thread);
}


private void
vk_init_vk(
	vk_t vk
	)
{
	assert_not_null(vk);

	srand48(time(NULL));

	vk_init_stats(vk);
	vk_init_instance(vk);
	vk_init_surface(vk);
	vk_init_device(vk);
	vk_init_commands(vk);
	vk_init_sets(vk);
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

	vk_free_all_staging_buffers(vk);

	vk_free_framebuffers(vk);
	vk_free_frames(vk);
	vk_free_swapchain(vk);
	vk_free_models(vk);
	vk_free_pipelines(vk);
	vk_free_sets(vk);
	vk_free_commands(vk);
	vk_free_device(vk);
	vk_free_surface(vk);
	vk_free_instance(vk);
	vk_free_stats(vk);
}


private void
vk_window_close_once_fn(
	vk_t vk,
	window_close_event_data_t* event_data
	)
{
	assert_not_null(vk);

	vk->window.close_once_listener = NULL;
	window_close(vk->window.handle);
}


private void
vk_window_free_once_fn(
	vk_t vk,
	window_free_event_data_t* event_data
	)
{
	assert_not_null(vk);

	vk_free_vk(vk);

	window_event_table_t* table = window_get_event_table(vk->window.handle);

	event_target_del(&table->key_up_target, vk->window.key_up_listener);
	event_target_del(&table->key_down_target, vk->window.key_down_listener);
	event_target_del(&table->mouse_move_target, vk->window.mouse_move_listener);
	event_target_del(&table->mouse_up_target, vk->window.mouse_up_listener);
	event_target_del(&table->mouse_down_target, vk->window.mouse_down_listener);
	event_target_del(&table->resize_target, vk->window.resize_listener);
	event_target_del_once(&table->close_target, vk->window.close_once_listener);

	simulation_stop(vk->simulation);
}


private void
vk_window_init_once_fn(
	vk_t vk,
	window_init_event_data_t* event_data
	)
{
	assert_not_null(vk);

	window_show(vk->window.handle);

	vk_init_vk(vk);
}


private void
vk_window_resize_fn(
	vk_t vk,
	window_resize_event_data_t* event_data
	)
{
	assert_not_null(vk);

	if(!atomic_exchange_acq_rel(&vk->window.resize.boolean, false, true))
	{
		sync_cond_wake(&vk->window.resize.cond);
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
		vk->window.mouse_holding = true;
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
		vk->window.mouse_holding = false;
	}
}


private void
vk_window_mouse_move_fn(
	vk_t vk,
	window_mouse_move_event_data_t* event_data
	)
{
	assert_not_null(vk);

	bool modify_angle = vk->window.mouse_holding;

	if(!modify_angle)
	{
		window_info_t info;
		window_get_info(vk->window.handle, &info);

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
		window_close(vk->window.handle);
	}
	else if(event_data->key == WINDOW_KEY_F11)
	{
		window_toggle_fullscreen(vk->window.handle);
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

	window_manager_run(vk->window.manager);
}


private void
vk_init_window(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->window.manager = window_manager_init();
	vk->window.handle = window_init();

	window_history_t history =
	{
		.extent =
		{
			.x = -1,
			.y = -1,
			.w = vk->options.window_width,
			.h = vk->options.window_height
		},
		.fullscreen = vk->options.window_fullscreen,
		.rel_mouse_in_fullscreen = true
	};
	window_manager_add(vk->window.manager, vk->window.handle, "Thesis", &history);

	atomic_init(&vk->window.resize.boolean, false);
	sync_mtx_init(&vk->window.resize.mtx);
	sync_cond_init(&vk->window.resize.cond);

	vk->window.mouse_holding = false;

	window_event_table_t* table = window_get_event_table(vk->window.handle);

	event_listener_data_t close_once_data =
	{
		.fn = (void*) vk_window_close_once_fn,
		.data = vk
	};
	vk->window.close_once_listener = event_target_once(&table->close_target, close_once_data);

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
	vk->window.resize_listener = event_target_add(&table->resize_target, resize_data);

	event_listener_data_t mouse_down_data =
	{
		.fn = (void*) vk_window_mouse_down_fn,
		.data = vk
	};
	vk->window.mouse_down_listener = event_target_add(&table->mouse_down_target, mouse_down_data);

	event_listener_data_t mouse_up_data =
	{
		.fn = (void*) vk_window_mouse_up_fn,
		.data = vk
	};
	vk->window.mouse_up_listener = event_target_add(&table->mouse_up_target, mouse_up_data);

	event_listener_data_t mouse_move_data =
	{
		.fn = (void*) vk_window_mouse_move_fn,
		.data = vk
	};
	vk->window.mouse_move_listener = event_target_add(&table->mouse_move_target, mouse_move_data);

	event_listener_data_t key_down_data =
	{
		.fn = (void*) vk_window_key_down_fn,
		.data = vk
	};
	vk->window.key_down_listener = event_target_add(&table->key_down_target, key_down_data);

	event_listener_data_t key_up_data =
	{
		.fn = (void*) vk_window_key_up_fn,
		.data = vk
	};
	vk->window.key_up_listener = event_target_add(&table->key_up_target, key_up_data);

	thread_data_t thread_data =
	{
		.fn = (void*) vk_window_thread_fn,
		.data = vk
	};
	thread_init(&vk->window.thread, thread_data);
}


private void
vk_free_window(
	vk_t vk
	)
{
	assert_not_null(vk);

	thread_free(&vk->window.thread);

	sync_cond_free(&vk->window.resize.cond);
	sync_mtx_free(&vk->window.resize.mtx);

	window_manager_free(vk->window.manager);
}


private void
vk_free(
	vk_t vk,
	simulation_free_event_data_t* event_data
	)
{
	assert_not_null(vk);

	if(!vk->options.window_enable)
	{
		alloc_free(vk, sizeof(*vk));
		return;
	}

	window_manager_stop_running(vk->window.manager);
	thread_join(vk->window.thread);

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

	if(!vk->options.window_enable)
	{
		return vk;
	}

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
