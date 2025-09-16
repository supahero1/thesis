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

#include <thesis/time.h>
#include <thesis/debug.h>
#include <thesis/collider.h>
#include <thesis/alloc_ext.h>

#include <thesis/octree.h>

#include <stddef.h>

#define COLLIDER_MAP_SIZE 1000000.0f
#define COLLIDER_STATS_SIZE 64
#define COLLIDER_RESTITUTION 0.8f
#define COLLISION_IMPULSE_FACTOR -(1.0f + COLLIDER_RESTITUTION)


struct collider
{
	stats_t stats;
	octree_t octree;
	float delta;

	collider_entity_t** query_entities;
	uint32_t query_entities_used;
	uint32_t query_entities_size;
};


collider_t
collider_init(
	stats_t stats
	)
{
	collider_t collider = alloc_calloc(sizeof(*collider));
	assert_not_null(collider);

	collider->stats = stats;
	stats_add(collider->stats, "collider_normalize", COLLIDER_STATS_SIZE);
	stats_add(collider->stats, "collider_collision_update", COLLIDER_STATS_SIZE);
	stats_add(collider->stats, "collider_resolution_update", COLLIDER_STATS_SIZE);

	collider->octree.half_extent =
	(half_extent_3d_t)
	{
		.w = COLLIDER_MAP_SIZE,
		.h = COLLIDER_MAP_SIZE,
		.d = COLLIDER_MAP_SIZE,
	};
	collider->octree.rect_extent = half_to_rect_3d_extent(collider->octree.half_extent);
	collider->octree.min_size = 64.0f;
	octree_init(&collider->octree);

	return collider;
}


void
collider_free(
	collider_t collider
	)
{
	assert_not_null(collider);

	stats_del(collider->stats, "collider_resolution_update");
	stats_del(collider->stats, "collider_collision_update");
	stats_del(collider->stats, "collider_normalize");

	octree_free(&collider->octree);

	alloc_free(collider, sizeof(*collider));
}


void
collider_add(
	collider_t collider,
	const collider_entity_t* entity
	)
{
	assert_not_null(collider);
	assert_not_null(entity);

	octree_insert(&collider->octree, entity);
}


private void
collider_query_entity(
	collider_t collider,
	collider_entity_t* entity
	)
{
	assert_not_null(collider);
	assert_not_null(entity);

	if(collider->query_entities_used >= collider->query_entities_size)
	{
		uint32_t new_size = (collider->query_entities_size << 1) | 1;

		collider->query_entities = alloc_remalloc(
			collider->query_entities,
			sizeof(*collider->query_entities) * collider->query_entities_size,
			sizeof(*collider->query_entities) * new_size
			);
		assert_not_null(collider->query_entities);

		collider->query_entities_size = new_size;
	}

	collider->query_entities[collider->query_entities_used++] = entity;
}


private void
collider_query_fn(
	octree_t* ot,
	uint32_t entity_idx,
	collider_entity_t* entity
	)
{
	assert_not_null(ot);
	assert_not_null(entity);

	(void) entity_idx;

	collider_t collider = MACRO_CONTAINER_OF(ot, struct collider, octree);
	collider_query_entity(collider, entity);
}


private void
collider_query(
	collider_t collider,
	rect_extent_3d_t extent
	)
{
	assert_not_null(collider);

	collider->query_entities_used = 0;

	octree_query(&collider->octree, extent, (void*) collider_query_fn);

	if(collider->query_entities_used < collider->query_entities_size / 4)
	{
		uint32_t new_size = collider->query_entities_size >> 1;

		collider->query_entities = alloc_remalloc(
			collider->query_entities,
			sizeof(*collider->query_entities) * collider->query_entities_size,
			sizeof(*collider->query_entities) * new_size
			);
		assert_not_null(collider->query_entities);

		collider->query_entities_size = new_size;
	}
}


private triplet_t
collider_closest_point_on_triangle(
	triplet_t p,
	triplet_t a,
	triplet_t b,
	triplet_t c
	)
{
	triplet_t ab = triplet_sub(b, a);
	triplet_t ac = triplet_sub(c, a);
	triplet_t ap = triplet_sub(p, a);

	float d1 = triplet_dot(ab, ap);
	float d2 = triplet_dot(ac, ap);
	if(d1 <= 0.0f && d2 <= 0.0f)
	{
		return a;
	}

	triplet_t bp = triplet_sub(p, b);

	float d3 = triplet_dot(ab, bp);
	float d4 = triplet_dot(ac, bp);
	if(d3 >= 0.0f && d4 <= d3)
	{
		return b;
	}

	triplet_t cp = triplet_sub(p, c);

	float d5 = triplet_dot(ab, cp);
	float d6 = triplet_dot(ac, cp);
	if(d6 >= 0.0f && d5 <= d6)
	{
		return c;
	}

	float vc = d1 * d4 - d3 * d2;
	if(vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
	{
		float v = d1 / (d1 - d3);
		return triplet_add(a, triplet_scale(ab, v));
	}

	float vb = d5 * d2 - d1 * d6;
	if(vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
	{
		float v = d2 / (d2 - d6);
		return triplet_add(a, triplet_scale(ac, v));
	}

	float va = d3 * d6 - d5 * d4;
	if(va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
	{
		float v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return triplet_add(b, triplet_scale(triplet_sub(c, b), v));
	}

	float denom = 1.0f / (va + vb + vc);
	float v = vb * denom;
	float w = vc * denom;

	return triplet_add(a, triplet_add(triplet_scale(ab, v), triplet_scale(ac, w)));
}


private void
collider_collision_update_entity(
	collider_t collider,
	uint32_t entity_idx,
	collider_entity_t* entity_a
	)
{
	assert_not_null(collider);
	assert_not_null(entity_a);

	(void) entity_idx;

	if(entity_a->type == COLLIDER_ENTITY_TYPE_TRIANGLE)
	{
		return;
	}

	collider_query(collider, entity_a->rect_extent);

	triplet_t center_a =
	{
		.x = (entity_a->rect_extent.min.x + entity_a->rect_extent.max.x) * 0.5f,
		.y = (entity_a->rect_extent.min.y + entity_a->rect_extent.max.y) * 0.5f,
		.z = (entity_a->rect_extent.min.z + entity_a->rect_extent.max.z) * 0.5f
	};

	for(uint32_t i = 0; i < collider->query_entities_used; ++i)
	{
		collider_entity_t* entity_b = collider->query_entities[i];
		if(entity_a == entity_b)
		{
			continue;
		}

		switch(entity_b->type)
		{

		case COLLIDER_ENTITY_TYPE_TRIANGLE:
		{
			triplet_t closest_point = collider_closest_point_on_triangle(
				center_a, entity_b->v0, entity_b->v1, entity_b->v2);

			triplet_t penetration = triplet_sub(center_a, closest_point);
			if(triplet_length(penetration) >= entity_a->r)
			{
				break;
			}

			triplet_t normal = triplet_normalize(penetration);

			float penetration_depth = entity_a->r - triplet_length(penetration);
			triplet_t correction = triplet_scale(normal, penetration_depth);

			entity_a->correction = triplet_add(entity_a->correction, correction);
			++entity_a->collision_count;

			float relative_velocity = triplet_dot(entity_a->v, normal);
			if(relative_velocity >= 0.0f)
			{
				break;
			}

			triplet_t impulse = triplet_scale(normal, COLLISION_IMPULSE_FACTOR * relative_velocity);
			entity_a->v = triplet_add(entity_a->v, impulse);

			break;
		}

		case COLLIDER_ENTITY_TYPE_SPHERE:
		{
			triplet_t center_b =
			{
				.x = (entity_b->rect_extent.min.x + entity_b->rect_extent.max.x) * 0.5f,
				.y = (entity_b->rect_extent.min.y + entity_b->rect_extent.max.y) * 0.5f,
				.z = (entity_b->rect_extent.min.z + entity_b->rect_extent.max.z) * 0.5f
			};

			triplet_t penetration = triplet_sub(center_a, center_b);

			float distance = triplet_length(penetration);
			if(distance >= entity_a->r + entity_b->r)
			{
				break;
			}

			triplet_t normal = triplet_normalize(penetration);

			float penetration_depth = entity_a->r + entity_b->r - distance;
			triplet_t correction = triplet_scale(normal, penetration_depth * 0.5f);

			entity_a->correction = triplet_add(entity_a->correction, correction);
			++entity_a->collision_count;

			entity_b->correction = triplet_sub(entity_b->correction, correction);
			++entity_b->collision_count;

			triplet_t velocity_diff = triplet_sub(entity_a->v, entity_b->v);
			float relative_velocity = triplet_dot(velocity_diff, normal);
			if(relative_velocity >= 0.0f)
			{
				break;
			}

			triplet_t impulse = triplet_scale(normal, COLLISION_IMPULSE_FACTOR * relative_velocity);

			entity_a->v = triplet_add(entity_a->v, impulse);
			entity_b->v = triplet_sub(entity_b->v, impulse);

			break;
		}

		default: assert_unreachable();

		}
	}
}


private octree_status_t
collider_collision_update_fn(
	octree_t* ot,
	uint32_t entity_idx,
	collider_entity_t* entity
	)
{
	assert_not_null(ot);
	assert_not_null(entity);

	collider_t collider = MACRO_CONTAINER_OF(ot, struct collider, octree);
	collider_collision_update_entity(collider, entity_idx, entity);

	return OCTREE_STATUS_NOT_CHANGED;
}


private bool
collider_resolution_update_entity(
	collider_t collider,
	uint32_t entity_idx,
	collider_entity_t* entity
	)
{
	assert_not_null(collider);
	assert_not_null(entity);

	(void) entity_idx;

	if(entity->type == COLLIDER_ENTITY_TYPE_TRIANGLE)
	{
		return false;
	}

	if(!entity->collision_count)
	{
		return false;
	}

	triplet_t pos =
	{
		.x = (entity->rect_extent.min.x + entity->rect_extent.max.x) * 0.5f,
		.y = (entity->rect_extent.min.y + entity->rect_extent.max.y) * 0.5f,
		.z = (entity->rect_extent.min.z + entity->rect_extent.max.z) * 0.5f
	};

	entity->correction = triplet_scale(entity->correction, 1.0f / entity->collision_count);
	entity->collision_count = 0;

	pos = triplet_add(pos, entity->correction);
	entity->correction = (triplet_t){{ 0.0f, 0.0f, 0.0f }};

	triplet_t a = (triplet_t){{ 0.0f, -9.81f, 0.0f }};
	entity->v = triplet_add(entity->v, triplet_scale(a, collider->delta));
	pos = triplet_add(pos, triplet_scale(entity->v, collider->delta));

	entity->rect_extent =
	(rect_extent_3d_t)
	{
		.min = {{ pos.x - entity->r, pos.y - entity->r, pos.z - entity->r }},
		.max = {{ pos.x + entity->r, pos.y + entity->r, pos.z + entity->r }}
	};

	if(entity->external)
	{
		*entity->external = pos;
	}

	return true;
}


private octree_status_t
collider_resolution_update_fn(
	octree_t* ot,
	uint32_t entity_idx,
	collider_entity_t* entity
	)
{
	assert_not_null(ot);
	assert_not_null(entity);

	collider_t collider = MACRO_CONTAINER_OF(ot, struct collider, octree);
	bool status = collider_resolution_update_entity(collider, entity_idx, entity);

	return status ? OCTREE_STATUS_CHANGED : OCTREE_STATUS_NOT_CHANGED;
}


void
collider_update(
	collider_t collider,
	float delta
	)
{
	collider->delta = delta;

	uint64_t start = time_get();
	octree_normalize(&collider->octree);
	stats_log(collider->stats, "collider_normalize", time_get() - start);

	start = time_get();
	octree_update(&collider->octree, collider_collision_update_fn);
	stats_log(collider->stats, "collider_collision_update", time_get() - start);

	start = time_get();
	octree_update(&collider->octree, collider_resolution_update_fn);
	stats_log(collider->stats, "collider_resolution_update", time_get() - start);
}
