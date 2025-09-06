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

#define octree_entity_data collider_entity_t
#define octree_get_entity_data_rect_extent(entity) (entity).rect_extent
#include <thesis/octree.h>

#include <stddef.h>

#define COLLIDER_MAP_SIZE 16384.0f
#define COLLIDER_STATS_SIZE 60


struct collider
{
	stats_t stats;
	octree_t octree;
	float delta;
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
	stats_add(collider->stats, "collider_update", COLLIDER_STATS_SIZE);

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

	stats_del(collider->stats, "collider_update");
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
collider_update_entity(
	collider_t collider,
	uint32_t entity_idx,
	collider_entity_t* entity
	)
{
}


private octree_status_t
collider_type_to_status(
	collider_entity_type_t type
	)
{
	switch(type)
	{

	case COLLIDER_ENTITY_TYPE_TRIANGLE: return OCTREE_STATUS_NOT_CHANGED;
	case COLLIDER_ENTITY_TYPE_SPHERE: return OCTREE_STATUS_CHANGED;
	default: assert_unreachable();

	}
}


private octree_status_t
collider_octree_update_fn(
	octree_t* ot,
	uint32_t entity_idx,
	collider_entity_t* entity
	)
{
	assert_not_null(ot);
	assert_not_null(entity);

	collider_t collider = MACRO_CONTAINER_OF(ot, struct collider, octree);
	collider_update_entity(collider, entity_idx, entity);

	return collider_type_to_status(entity->type);
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
	octree_update(&collider->octree, collider_octree_update_fn);
	stats_log(collider->stats, "collider_update", time_get() - start);
}
