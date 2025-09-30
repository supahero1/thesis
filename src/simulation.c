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

#include <thesis/stb.h>
#include <thesis/hash.h>
#include <thesis/time.h>
#include <thesis/debug.h>
#include <thesis/collider.h>
#include <thesis/alloc_ext.h>
#include <thesis/simulation.h>

#include <vips/vips.h>

#include <stdatomic.h>

#define SIMULATION_STATS_SIZE 64


typedef struct simulation_texture_info
{
	uint32_t model_idx;
	simulation_texture_t* texture;
}
simulation_texture_info_t;

typedef struct simulation_entity
{
	uint32_t model_idx;
	vec3 translation;
	vec3 rotation;
	vec3 center;
	bool dynamic;
}
simulation_entity_t;

struct simulation
{
	sync_mtx_t mutex;
	uint64_t last_update;

	simulation_camera_t camera;
	simulation_light_t light;

	simulation_model_info_t info;
	hash_table_t model_table;

	str_t skybox_path;

	simulation_entity_t** entities;
	uint32_t entity_count;

	stats_t stats;
	collider_t collider;

	atomic_flag stopped;

	simulation_event_table_t event_table;
};


private void
simulation_model_table_value_free_fn(
	simulation_texture_info_t* info
	)
{
	assert_not_null(info);

	if(info->texture)
	{
		alloc_free(info->texture->data, info->texture->size);
		alloc_free(info->texture, sizeof(*info->texture));
	}

	alloc_free(info, sizeof(*info));
}


simulation_t
simulation_init(
	simulation_camera_t camera,
	simulation_light_t light,
	const char* skybox_path
	)
{
	int status = VIPS_INIT("thesis");
	hard_assert_false(status);

	simulation_t simulation = alloc_calloc(sizeof(*simulation));
	assert_not_null(simulation);

	sync_mtx_init(&simulation->mutex);

	camera.fov = glm_rad(camera.fov);
	camera.angle[0] = glm_rad(camera.angle[0]);
	camera.angle[1] = glm_rad(camera.angle[1]);
	camera.angle[2] = glm_rad(camera.angle[2]);
	simulation->camera = camera;

	simulation->light = light;

	simulation->model_table = hash_table_init(8, NULL, (void*) simulation_model_table_value_free_fn);

	simulation->skybox_path = str_init_copy_cstr(skybox_path);

	simulation->stats = stats_init();
	simulation->collider = collider_init(simulation->stats);

	stats_add(simulation->stats, "simulation_update", SIMULATION_STATS_SIZE);

	atomic_flag_clear(&simulation->stopped);

	event_target_init(&simulation->event_table.free_target);

	return simulation;
}


void
simulation_free(
	simulation_t simulation
	)
{
	assert_not_null(simulation);

	simulation_free_event_data_t event_data =
	{
		.simulation = simulation
	};
	event_target_fire(&simulation->event_table.free_target, &event_data);

	event_target_free(&simulation->event_table.free_target);

	stats_del(simulation->stats, "simulation_update");

	collider_free(simulation->collider);
	stats_free(simulation->stats);

	for(uint32_t i = 0; i < simulation->entity_count; ++i)
	{
		alloc_free(simulation->entities[i], sizeof(*simulation->entities[i]));
	}
	alloc_free(simulation->entities, sizeof(*simulation->entities) * simulation->entity_count);

	str_free(simulation->skybox_path);

	hash_table_free(simulation->model_table);

	for(uint32_t i = 0; i < simulation->info.model_count; ++i)
	{
		model_free(simulation->info.models[i]);
	}
	alloc_free(simulation->info.models, sizeof(*simulation->info.models) * simulation->info.model_count);

	sync_mtx_free(&simulation->mutex);

	alloc_free(simulation, sizeof(*simulation));

	vips_shutdown();
}


simulation_event_table_t*
simulation_get_event_table(
	simulation_t simulation
	)
{
	assert_not_null(simulation);

	return &simulation->event_table;
}


stats_t
simulation_get_stats(
	simulation_t simulation
	)
{
	assert_not_null(simulation);

	return simulation->stats;
}


private void
simulation_add_triangle_collider_entity(
	simulation_t simulation,
	vec3 translation,
	vec3 v0,
	vec3 v1,
	vec3 v2
	)
{
	assert_not_null(simulation);

	rect_extent_3d_t extent;
	extent.min.x = MACRO_MIN(v0[0], MACRO_MIN(v1[0], v2[0]));
	extent.min.y = MACRO_MIN(v0[1], MACRO_MIN(v1[1], v2[1]));
	extent.min.z = MACRO_MIN(v0[2], MACRO_MIN(v1[2], v2[2]));
	extent.max.x = MACRO_MAX(v0[0], MACRO_MAX(v1[0], v2[0]));
	extent.max.y = MACRO_MAX(v0[1], MACRO_MAX(v1[1], v2[1]));
	extent.max.z = MACRO_MAX(v0[2], MACRO_MAX(v1[2], v2[2]));

	extent.min = triplet_add(extent.min, *(triplet_t*) translation);
	extent.max = triplet_add(extent.max, *(triplet_t*) translation);

	collider_entity_t collider_entity =
	{
		.type = COLLIDER_ENTITY_TYPE_TRIANGLE,

		.rect_extent = extent,

		.v0 = {{ v0[0], v0[1], v0[2] }},
		.v1 = {{ v1[0], v1[1], v1[2] }},
		.v2 = {{ v2[0], v2[1], v2[2] }}
	};
	collider_add(simulation->collider, &collider_entity);
}


private void
simulation_add_mesh_collider_entity(
	simulation_t simulation,
	simulation_entity_t* entity
	)
{
	assert_not_null(simulation);
	assert_not_null(entity);

	model_t* model = simulation->info.models[entity->model_idx];
	for(uint32_t i = 0; i < model->mesh_count; ++i)
	{
		mesh_t* mesh = &model->meshes[i];

		uint32_t* index = &mesh->indexes[0];
		uint32_t* index_end = index + mesh->index_count;

		while(index != index_end)
		{
			simulation_add_triangle_collider_entity(
				simulation,
				entity->translation,
				mesh->vertices[index[0]],
				mesh->vertices[index[1]],
				mesh->vertices[index[2]]
				);
			index += 3;
		}
	}

	glm_vec3_zero(entity->translation);
}


private triplet_t
simulation_add_collider_entity(
	simulation_t simulation,
	simulation_entity_t* entity
	)
{
	assert_not_null(simulation);
	assert_not_null(entity);

	if(!entity->dynamic)
	{
		simulation_add_mesh_collider_entity(simulation, entity);
		return (triplet_t){0};
	}

	rect_extent_3d_t extent =
	{
		.min = {{ FLT_MAX, FLT_MAX, FLT_MAX }},
		.max = {{ -FLT_MAX, -FLT_MAX, -FLT_MAX }}
	};

	model_t* model = simulation->info.models[entity->model_idx];
	for(uint32_t i = 0; i < model->mesh_count; ++i)
	{
		mesh_t* mesh = &model->meshes[i];

		for(uint32_t v = 0; v < mesh->vertex_count; ++v)
		{
			extent.min.x = MACRO_MIN(extent.min.x, mesh->vertices[v][0]);
			extent.min.y = MACRO_MIN(extent.min.y, mesh->vertices[v][1]);
			extent.min.z = MACRO_MIN(extent.min.z, mesh->vertices[v][2]);
			extent.max.x = MACRO_MAX(extent.max.x, mesh->vertices[v][0]);
			extent.max.y = MACRO_MAX(extent.max.y, mesh->vertices[v][1]);
			extent.max.z = MACRO_MAX(extent.max.z, mesh->vertices[v][2]);
		}
	}

	triplet_t center =
	{
		.x = (extent.min.x + extent.max.x) * 0.5f,
		.y = (extent.min.y + extent.max.y) * 0.5f,
		.z = (extent.min.z + extent.max.z) * 0.5f
	};

	float r = (extent.max.x - extent.min.x) * 0.5f;
	extent =
	(rect_extent_3d_t)
	{
		.min = {{ -r, -r, -r }},
		.max = {{  r,  r,  r }}
	};

	extent.min = triplet_add(extent.min, *(triplet_t*) entity->translation);
	extent.max = triplet_add(extent.max, *(triplet_t*) entity->translation);

	glm_vec3_copy((void*) &center, entity->translation);
	glm_vec3_negate(entity->translation);

	triplet_t v = {{ (float) rand() / RAND_MAX - 0.5f,
		(float) rand() / RAND_MAX - 0.5f,
		(float) rand() / RAND_MAX - 0.5f }};
	v = triplet_scale(triplet_normalize(v), 0.5f);

	triplet_t w;
	glm_vec3_copy((void*) &entity->rotation, (void*) &w);
	glm_vec3_zero(entity->rotation);

	collider_entity_t collider_entity =
	{
		.type = COLLIDER_ENTITY_TYPE_SPHERE,

		.rect_extent = extent,

		.pos_external = (void*) &entity->translation,
		.rotation_external = (void*) &entity->rotation,
		.pos_diff_count = 0,
		// .vw_diff_count = 0,
		.pos_diff = {{ 0.0f, 0.0f, 0.0f }},
		.v_force = {{ 0.0f, 0.0f, 0.0f }},
		// .w_diff = {{ 0.0f, 0.0f, 0.0f }},
		.v = v,
		.w = w
	};
	collider_add(simulation->collider, &collider_entity);

	return center;
}


void
simulation_add_entity(
	simulation_t simulation,
	simulation_entity_init_t entity_init
	)
{
	assert_not_null(simulation);

	simulation->entities = alloc_remalloc(
		simulation->entities,
		sizeof(*simulation->entities) * simulation->entity_count,
		sizeof(*simulation->entities) * (simulation->entity_count + 1)
		);
	assert_not_null(simulation->entities);

	simulation_entity_t* entity = alloc_malloc(sizeof(*entity));
	assert_not_null(entity);

	simulation->entities[simulation->entity_count++] = entity;

	simulation_texture_info_t* info = hash_table_get(
		simulation->model_table,
		entity_init.model_path
		);

	if(!info)
	{
		simulation->info.models = alloc_remalloc(
			simulation->info.models,
			sizeof(*simulation->info.models) * simulation->info.model_count,
			sizeof(*simulation->info.models) * (simulation->info.model_count + 1)
			);
		assert_not_null(simulation->info.models);

		model_t* model = model_init(entity_init.model_path, entity_init.scale);
		simulation->info.material_count += model->material_count;
		simulation->info.models[simulation->info.model_count] = model;

		info = alloc_calloc(sizeof(*info));
		assert_not_null(info);

		info->model_idx = simulation->info.model_count++;

		hash_table_set(simulation->model_table, entity_init.model_path, info);
	}

	entity->model_idx = info->model_idx;

	glm_vec3_copy(entity_init.translation, entity->translation);
	glm_vec3_copy(entity_init.rotation, entity->rotation);
	entity->dynamic = entity_init.dynamic;

	triplet_t center = simulation_add_collider_entity(simulation, entity);
	glm_vec3_copy((void*) &center, entity->center);
}


simulation_entity_data_t*
simulation_get_entity_data(
	simulation_t simulation,
	uint32_t* data_count
	)
{
	assert_not_null(simulation);

	if(data_count)
	{
		*data_count = simulation->entity_count;
	}

	simulation_entity_data_t* data = alloc_malloc(sizeof(*data) * simulation->entity_count);
	assert_ptr(data, sizeof(*data) * simulation->entity_count);

	for(uint32_t i = 0; i < simulation->entity_count; ++i)
	{
		simulation_entity_data_t* cur_data = &data[i];
		simulation_entity_t* entity = simulation->entities[i];

		cur_data->model_idx = entity->model_idx;

		vec3 neg_center;
		glm_vec3_negate_to(entity->center, neg_center);

		glm_mat4_identity(cur_data->transform);
		glm_translate(cur_data->transform, entity->translation);
		glm_rotate_z(cur_data->transform, entity->rotation[2], cur_data->transform);
		glm_rotate_y(cur_data->transform, entity->rotation[1], cur_data->transform);
		glm_rotate_x(cur_data->transform, entity->rotation[0], cur_data->transform);
		glm_translate(cur_data->transform, neg_center);
	}

	return data;
}


void
simulation_free_entity_data(
	simulation_entity_data_t* data,
	uint32_t data_count
	)
{
	alloc_free(data, sizeof(*data) * data_count);
}


simulation_camera_t
simulation_get_camera(
	simulation_t simulation
	)
{
	assert_not_null(simulation);

	return simulation->camera;
}


simulation_transform_t
simulation_get_transform(
	simulation_t simulation,
	float width,
	float height
	)
{
	assert_not_null(simulation);

	simulation_transform_t transform;

	glm_mat4_zero(transform.projection);
	float f = 1.0f / tanf(simulation->camera.fov * 0.5f);
	transform.projection[0][0] = f / (width / height);
	transform.projection[1][1] = f;
	transform.projection[2][3] = 1.0f;
	transform.projection[3][2] = simulation->camera.near;

	glm_mat4_inv(transform.projection, transform.inverse_projection);

	glm_mat4_identity(transform.view);
	glm_euler_xyz(simulation->camera.angle, transform.view);
	glm_translate(transform.view, (vec3){ -simulation->camera.pos[0],
		-simulation->camera.pos[1], -simulation->camera.pos[2] });

	glm_mat4_inv(transform.view, transform.inverse_view);

	glm_mat4_identity(transform.light_transform);

	vec3 light_up = { 0.0f, -1.0f, 0.0f };
	mat4 light_view;
	glm_lookat(simulation->light.pos, simulation->light.target, light_up, light_view);

	mat4 light_proj;
	glm_ortho(simulation->light.left, simulation->light.right, simulation->light.bottom,
		simulation->light.top, simulation->light.near, simulation->light.far, light_proj);

	glm_mat4_mul(light_proj, light_view, transform.light_transform);

	glm_vec3_sub(simulation->light.pos, simulation->light.target, transform.light_direction);
	glm_vec3_normalize(transform.light_direction);
	transform.light_direction[3] = 0.0f;

	return transform;
}


simulation_model_info_t
simulation_get_model_info(
	simulation_t simulation
	)
{
	assert_not_null(simulation);

	return simulation->info;
}


private simulation_texture_t*
simulation_load_texture(
	str_t path,
	bool is_cube_map
	)
{
	assert_not_null(path);

	simulation_texture_t* texture = alloc_calloc(sizeof(*texture));
	assert_not_null(texture);

	int width;
	int height;
	void* data;

	char full_path[256];
	memcpy(full_path, path->str, path->len + 1);
	char* full_path_end = full_path + path->len;

	stbi_set_flip_vertically_on_load(1);

	if(!is_cube_map)
	{
		texture->layers = 1;
	}
	else
	{
		memcpy(full_path_end, "/px.png", 8);
		texture->layers = 6;
	}

	data = stbi_load(full_path, &width, &height, NULL, 4);
	hard_assert_not_null(data,
		{
			stbi_print_failure();
			fprintf(stderr, "Failed to load texture: '%s'\n", full_path);
		}
		);

	texture->width = width;
	texture->height = height;
	texture->levels = 1 + MACRO_LOG2(MACRO_MAX(width, height));

	uint32_t size = 0;
	for(uint32_t i = 0; i < texture->levels; ++i)
	{
		size += width * height;
		width = MACRO_MAX(width >> 1, 1);
		height = MACRO_MAX(height >> 1, 1);
	}
	size *= texture->layers * 4;
	texture->size = size;

	texture->data = alloc_malloc(texture->size);
	assert_not_null(texture->data);

	uint32_t mip_size = texture->width * texture->height * 4;
	memcpy(texture->data, data, mip_size);
	stbi_image_free(data);

	if(is_cube_map)
	{
		void* image_data = texture->data + mip_size;

		memcpy(full_path_end, "/nx.png", 8);
		data = stbi_load(full_path, &width, &height, NULL, 4);
		hard_assert_not_null(data, stbi_print_failure());

		memcpy(image_data, data, mip_size);
		image_data += mip_size;
		stbi_image_free(data);

		memcpy(full_path_end, "/ny.png", 8);
		data = stbi_load(full_path, &width, &height, NULL, 4);
		hard_assert_not_null(data, stbi_print_failure());

		memcpy(image_data, data, mip_size);
		image_data += mip_size;
		stbi_image_free(data);

		memcpy(full_path_end, "/py.png", 8);
		data = stbi_load(full_path, &width, &height, NULL, 4);
		hard_assert_not_null(data, stbi_print_failure());

		memcpy(image_data, data, mip_size);
		image_data += mip_size;
		stbi_image_free(data);

		memcpy(full_path_end, "/pz.png", 8);
		data = stbi_load(full_path, &width, &height, NULL, 4);
		hard_assert_not_null(data, stbi_print_failure());

		memcpy(image_data, data, mip_size);
		image_data += mip_size;
		stbi_image_free(data);

		memcpy(full_path_end, "/nz.png", 8);
		data = stbi_load(full_path, &width, &height, NULL, 4);
		hard_assert_not_null(data, stbi_print_failure());

		memcpy(image_data, data, mip_size);
		stbi_image_free(data);
	}

	uint32_t mip_width = texture->width;
	uint32_t mip_height = texture->height;
	data = texture->data;

	for(uint32_t level = 1; level < texture->levels; ++level)
	{
		uint32_t next_mip_width = MACRO_MAX(mip_width >> 1, 1);
		uint32_t next_mip_height = MACRO_MAX(mip_height >> 1, 1);
		uint32_t next_mip_size = next_mip_width * next_mip_height * 4;
		void* next_data = data + mip_size * texture->layers;

		for(uint32_t layer = 0; layer < texture->layers; ++layer)
		{
			void* src_data = data + layer * mip_size;
			void* dest_data = next_data + layer * next_mip_size;

			VipsImage* src = vips_image_new_from_memory(
				src_data, mip_size, mip_width,
				mip_height, 4, VIPS_FORMAT_UCHAR
				);

			double scale_x = (double) next_mip_width / mip_width;
			double scale_y = (double) next_mip_height / mip_height;

			VipsImage* resized = NULL;
			int status = vips_resize(src, &resized, scale_x, "vscale", scale_y, NULL);
			assert_false(status, fprintf(stderr, "%s\n", vips_error_buffer()));

			size_t out_size;
			void* out_mem = vips_image_write_to_memory(resized, &out_size);
			assert_not_null(out_mem);
			assert_eq(out_size, next_mip_size);

			memcpy(dest_data, out_mem, out_size);
			g_free(out_mem);

			g_object_unref(src);
			g_object_unref(resized);
		}

		mip_width = next_mip_width;
		mip_height = next_mip_height;
		mip_size = next_mip_size;
		data = next_data;
	}

	return texture;
}


simulation_texture_t*
simulation_get_texture(
	simulation_t simulation,
	str_t path,
	bool is_cube_map
	)
{
	assert_not_null(simulation);
	assert_not_null(path);

	simulation_texture_info_t* info = hash_table_get(simulation->model_table, path->str);

	if(!info)
	{
		info = alloc_calloc(sizeof(*info));
		assert_not_null(info);

		hash_table_set(simulation->model_table, path->str, info);
	}

	if(!info->texture)
	{
		info->texture = simulation_load_texture(path, is_cube_map);
	}

	return info->texture;
}


const str_t
simulation_get_skybox_path(
	simulation_t simulation
	)
{
	assert_not_null(simulation);

	return simulation->skybox_path;
}


void
simulation_modify_position(
	simulation_t simulation,
	vec3 pos
	)
{
	assert_not_null(simulation);

	mat4 rotation;
	glm_mat4_identity(rotation);

	glm_rotate(rotation, simulation->camera.angle[1], (vec3){ 0.0f, 1.0f, 0.0f });
	glm_rotate(rotation, simulation->camera.angle[0], (vec3){ 1.0f, 0.0f, 0.0f });

	vec3 rotated_pos;
	glm_mat4_mulv3(rotation, pos, 1.0f, rotated_pos);

	glm_vec3_add(simulation->camera.pos, rotated_pos, simulation->camera.pos);
}


void
simulation_modify_angle(
	simulation_t simulation,
	vec3 angle
	)
{
	assert_not_null(simulation);

	glm_vec3_add(simulation->camera.angle, angle, simulation->camera.angle);
	simulation->camera.angle[0] = glm_clamp(simulation->camera.angle[0], -M_PI_2, M_PI_2);
}


void
simulation_stop(
	simulation_t simulation
	)
{
	assert_not_null(simulation);

	bool stopped = atomic_flag_test_and_set(&simulation->stopped);
	if(stopped)
	{
		return;
	}

	simulation_stop_event_data_t event_data =
	{
		.simulation = simulation
	};
	event_target_fire(&simulation->event_table.stop_target, &event_data);
}


void
simulation_update(
	simulation_t simulation
	)
{
	assert_not_null(simulation);

	sync_mtx_lock(&simulation->mutex);

	uint64_t now = time_get();
	if(!simulation->last_update)
	{
		simulation->last_update = now;
		sync_mtx_unlock(&simulation->mutex);
		return;
	}

	float delta = (float)(now - simulation->last_update) / time_ms_to_ns(1) / 16.66667f;
	simulation->last_update = now;

	collider_update(simulation->collider, delta);

	stats_log(simulation->stats, "simulation_update", time_get() - now);

	sync_mtx_unlock(&simulation->mutex);
}
