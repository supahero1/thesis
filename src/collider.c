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
#include <thesis/collider.h>
#include <thesis/alloc_ext.h>

#include <thesis/octree.h>

#include <math.h>
#include <stddef.h>

#define COLLIDER_MAP_SIZE 1000000.0f
#define COLLIDER_STATS_SIZE 64
#define COLLIDER_RESTITUTION 0.8f
#define COLLISION_IMPULSE_FACTOR -(1.0f + COLLIDER_RESTITUTION)
#define COLLIDER_VELOCITY_LIMIT 15.0f


struct collider
{
	stats_t stats;
	octree_t octree;
	float delta;

	collider_entity_t* balls;
	uint32_t balls_count;

	collider_entity_t** query_entities;
	uint32_t query_entities_used;
	uint32_t query_entities_size;

	float ball_mass;
	float inv_ball_mass;
	float ball_radius;
	float ball_diameter;
	float ball_moment_inertia;
};


collider_t
collider_init(
	stats_t stats
	)
{
	collider_t collider = alloc_calloc(collider, 1);
	assert_not_null(collider);

	collider->stats = stats;
	stats_add(collider->stats, "collider_normalize", COLLIDER_STATS_SIZE);
	stats_add(collider->stats, "collider_collide", COLLIDER_STATS_SIZE);
	stats_add(collider->stats, "collider_resolve", COLLIDER_STATS_SIZE);

	collider->octree.half_extent =
	(half_extent_3d_t)
	{
		.w = COLLIDER_MAP_SIZE,
		.h = COLLIDER_MAP_SIZE,
		.d = COLLIDER_MAP_SIZE,
	};
	collider->octree.rect_extent = half_to_rect_3d_extent(collider->octree.half_extent);
	collider->octree.min_size = 32.0f;
	octree_init(&collider->octree);

	collider->ball_mass = 1.0f;
	collider->inv_ball_mass = 1.0f / collider->ball_mass;

	return collider;
}


void
collider_free(
	collider_t collider
	)
{
	assert_not_null(collider);

	alloc_free(collider->query_entities, collider->query_entities_size);

	alloc_free(collider->balls, collider->balls_count);

	stats_del(collider->stats, "collider_resolve");
	stats_del(collider->stats, "collider_collide");
	stats_del(collider->stats, "collider_normalize");

	octree_free(&collider->octree);

	alloc_free(collider, 1);
}


void
collider_add(
	collider_t collider,
	const collider_entity_t* entity
	)
{
	assert_not_null(collider);
	assert_not_null(entity);

	if(entity->type == COLLIDER_ENTITY_TYPE_TRIANGLE)
	{
		octree_insert(&collider->octree, entity);
		return;
	}

	collider->balls = alloc_remalloc(collider->balls, collider->balls_count, collider->balls_count + 1);
	assert_not_null(collider->balls);

	collider->balls[collider->balls_count++] = *entity;

	if(collider->ball_radius == 0.0f)
	{
		collider->ball_radius = (entity->rect_extent.max.x - entity->rect_extent.min.x) * 0.5f;
		collider->ball_diameter = collider->ball_radius * 2.0f;
		collider->ball_moment_inertia = 0.4f * collider->ball_mass * collider->ball_radius * collider->ball_radius;
	}

	assert_lt(collider->ball_radius - (entity->rect_extent.max.x - entity->rect_extent.min.x) * 0.5f, 0.01f);
	assert_lt(collider->ball_radius - (entity->rect_extent.max.y - entity->rect_extent.min.y) * 0.5f, 0.01f);
	assert_lt(collider->ball_radius - (entity->rect_extent.max.z - entity->rect_extent.min.z) * 0.5f, 0.01f);
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

		collider->query_entities = alloc_remalloc(collider->query_entities, collider->query_entities_size, new_size);
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

		collider->query_entities = alloc_remalloc(collider->query_entities, collider->query_entities_size, new_size);
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


private triplet_t
collider_get_surface_velocity(
	triplet_t center,
	triplet_t velocity,
	triplet_t angular_velocity,
	triplet_t contact_point
	)
{
	triplet_t radius_vec = triplet_sub(contact_point, center);
	triplet_t surface_vel = triplet_cross(angular_velocity, radius_vec);
	return triplet_add(velocity, surface_vel);
}


private void
collider_ball_collide_entity(
	collider_t collider,
	collider_entity_t* entity_a,
	collider_entity_t* entity_b
	)
{
	assert_not_null(entity_a);
	assert_not_null(entity_b);

	if(entity_a == entity_b)
	{
		return;
	}

	triplet_t center_a =
	{
		.x = (entity_a->rect_extent.min.x + entity_a->rect_extent.max.x) * 0.5f,
		.y = (entity_a->rect_extent.min.y + entity_a->rect_extent.max.y) * 0.5f,
		.z = (entity_a->rect_extent.min.z + entity_a->rect_extent.max.z) * 0.5f
	};

	if(entity_b->type == COLLIDER_ENTITY_TYPE_TRIANGLE)
	{
		triplet_t closest_point = collider_closest_point_on_triangle(
			center_a, entity_b->v0, entity_b->v1, entity_b->v2);

		triplet_t penetration = triplet_sub(center_a, closest_point);
		if(triplet_length(penetration) >= collider->ball_radius)
		{
			return;
		}

		entity_a->hit = true;

		triplet_t normal = triplet_normalize(penetration);

		float penetration_depth = collider->ball_radius - triplet_length(penetration);
		triplet_t correction = triplet_scale(normal, penetration_depth);

		entity_a->pos_diff = triplet_add(entity_a->pos_diff, correction);
		++entity_a->pos_diff_count;

		float relative_velocity = triplet_dot(entity_a->v, normal);
		if(relative_velocity >= 0.0f)
		{
			return;
		}

		triplet_t impulse = triplet_scale(normal, COLLISION_IMPULSE_FACTOR * relative_velocity);
		entity_a->v_force = triplet_add(entity_a->v_force, impulse);
		++entity_a->force_count;
	}
	else /* COLLIDER_ENTITY_TYPE_SPHERE */
	{
		triplet_t center_b =
		{
			.x = (entity_b->rect_extent.min.x + entity_b->rect_extent.max.x) * 0.5f,
			.y = (entity_b->rect_extent.min.y + entity_b->rect_extent.max.y) * 0.5f,
			.z = (entity_b->rect_extent.min.z + entity_b->rect_extent.max.z) * 0.5f
		};

		triplet_t penetration = triplet_sub(center_a, center_b);

		float distance = triplet_length(penetration);
		if(distance >= collider->ball_diameter)
		{
			return;
		}

		triplet_t normal = triplet_normalize(penetration);

		float penetration_depth = collider->ball_diameter - distance;
		triplet_t pos_diff = triplet_scale(normal, penetration_depth * 0.5f);

		entity_a->pos_diff = triplet_add(entity_a->pos_diff, pos_diff);
		++entity_a->pos_diff_count;

		entity_b->pos_diff = triplet_sub(entity_b->pos_diff, pos_diff);
		++entity_b->pos_diff_count;

		triplet_t velocity_diff = triplet_sub(entity_a->v, entity_b->v);
		float relative_velocity = triplet_dot(velocity_diff, normal);
		if(relative_velocity >= 0.0f)
		{
			return;
		}

		triplet_t v_force = triplet_scale(normal, COLLISION_IMPULSE_FACTOR * relative_velocity * 0.5f);
		entity_a->v_force = triplet_add(entity_a->v_force, v_force);
		++entity_a->force_count;
		entity_b->v_force = triplet_sub(entity_b->v_force, v_force);
		++entity_b->force_count;
	}
}


private bool
collider_ball_resolve(
	collider_t collider,
	collider_entity_t* entity
	)
{
	assert_not_null(collider);
	assert_not_null(entity);

	triplet_t pos =
	{
		.x = (entity->rect_extent.min.x + entity->rect_extent.max.x) * 0.5f,
		.y = (entity->rect_extent.min.y + entity->rect_extent.max.y) * 0.5f,
		.z = (entity->rect_extent.min.z + entity->rect_extent.max.z) * 0.5f
	};

	if(entity->pos_diff_count)
	{
		entity->pos_diff = triplet_scale(entity->pos_diff, 1.0f / entity->pos_diff_count);
		entity->pos_diff_count = 0;
	}

	pos = triplet_add(pos, entity->pos_diff);
	entity->pos_diff = (triplet_t){{ 0.0f, 0.0f, 0.0f }};

	if(entity->force_count)
	{
		entity->v_force = triplet_scale(entity->v_force, 1.0f / entity->force_count);
		entity->force_count = 0;
	}

	entity->v = triplet_add(entity->v, entity->v_force);
	entity->v_force = (triplet_t){{ 0.0f, 0.0f, 0.0f }};

	triplet_t a = (triplet_t){{ 0.0f, -0.01f, 0.0f }};
	entity->v = triplet_add(entity->v, triplet_scale(a, collider->delta));

	float speed = triplet_length(entity->v);
	if(speed > COLLIDER_VELOCITY_LIMIT)
	{
		entity->v = triplet_scale(entity->v, COLLIDER_VELOCITY_LIMIT / speed);
	}

	pos = triplet_add(pos, triplet_scale(entity->v, collider->delta));

	entity->rect_extent =
	(rect_extent_3d_t)
	{
		.min = {{ pos.x - collider->ball_radius, pos.y - collider->ball_radius, pos.z - collider->ball_radius }},
		.max = {{ pos.x + collider->ball_radius, pos.y + collider->ball_radius, pos.z + collider->ball_radius }}
	};

	if(entity->pos_external)
	{
		*entity->pos_external = pos;
	}

	if(entity->rotation_external)
	{
		triplet_t rotation = *entity->rotation_external;
		rotation = triplet_add(rotation, triplet_scale(entity->w, collider->delta));
		rotation.x = fmodf(rotation.x, M_PI * 2.0f);
		rotation.y = fmodf(rotation.y, M_PI * 2.0f);
		rotation.z = fmodf(rotation.z, M_PI * 2.0f);
		*entity->rotation_external = rotation;
	}

	return true;
}


private void
collider_collide(
	collider_t collider
	)
{
	assert_not_null(collider);

	collider_entity_t* ball = collider->balls;
	collider_entity_t* ball_end = ball + collider->balls_count;

	while(ball != ball_end)
	{
		ball->hit = false;

		collider_query(collider, ball->rect_extent);

		for(uint32_t i = 0; i < collider->query_entities_used; ++i)
		{
			collider_ball_collide_entity(collider, ball, collider->query_entities[i]);
		}

		collider_entity_t* other_ball = ball + 1;

		while(other_ball != ball_end)
		{
			collider_ball_collide_entity(collider, ball, other_ball);

			++other_ball;
		}

		++ball;
	}
}


private void
collider_resolve(
	collider_t collider
	)
{
	assert_not_null(collider);

	collider_entity_t* ball = collider->balls;
	collider_entity_t* ball_end = ball + collider->balls_count;

	while(ball != ball_end)
	{
		collider_ball_resolve(collider, ball);

		++ball;
	}
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
	collider_collide(collider);
	stats_log(collider->stats, "collider_collide", time_get() - start);

	start = time_get();
	collider_resolve(collider);
	stats_log(collider->stats, "collider_resolve", time_get() - start);
}
